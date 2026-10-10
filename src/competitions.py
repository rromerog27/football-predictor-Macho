"""Registro de ligas y copas, y carga de sus datos para el modelo.

- Las 6 ligas de Understat usan xG real; las demás, el xG aproximado con tiros
  de ESPN (o goles donde ESPN no publica tiros).
- Las copas internacionales (Champions, Libertadores...) se modelan junto con
  las ligas de sus participantes (su "pool"): así la fuerza de un equipo sale
  sobre todo de su liga, y los cruces entre países de la copa calibran unas
  ligas frente a otras.
- El calendario del día y las cuotas salen de ESPN en todas las competiciones
  (en las de Understat, las cuotas se cruzan por hora y nombre de equipo).
- Córners y tarjetas (para src/corners_cards.py) también salen de ESPN: en las ligas
  de Understat se cruzan partido a partido por fecha y nombres.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pandas as pd
import requests

from src import espn_source, lineups, market_signal, understat_source
from src import football_data_source as fd
from src.match_model import (
    STANDARD_COLUMNS,
    STAT_COLUMNS,
    LeagueData,
    LeagueModel,
    MatchPrediction,
    PredictionError,
    build_league_data,
    predict_match,
    normalize,
    team_similarity,
    utc_now,
)

UNDERSTAT_SEASONS = 5  # temporadas anteriores a la actual (la más antigua es solo historial previo)
ESPN_LEAGUE_YEARS = 4  # años naturales de una liga de ESPN (incluido el actual)
ESPN_CUP_YEARS = 2  # años naturales del pool de una copa (muchas ligas: se limita la descarga)
ESPN_STATS_YEARS = 3  # años naturales de córners y tarjetas de ESPN en las ligas de Understat
MAX_WORKERS = 8
MIN_MAPPING_SIMILARITY = 0.85  # nombres ESPN ↔ Understat del mismo club (p. ej. "Leeds United" ↔ "Leeds")


@dataclass(frozen=True)
class Competition:
    code: str  # identificador estable (slug de ESPN)
    name: str
    region: str
    understat: str | None = None  # código de Understat si hay xG real
    pool: tuple[str, ...] = ()  # copas: ligas (slugs de ESPN) que alimentan la fuerza de sus equipos
    aliases: tuple[str, ...] = ()
    lower: tuple[str, ...] = ()  # división inferior (ESPN): historial de los recién ascendidos

    @property
    def is_cup(self) -> bool:
        return bool(self.pool)


UEFA_POOL = ("eng.1", "esp.1", "ger.1", "ita.1", "fra.1", "por.1", "ned.1", "bel.1", "sco.1", "tur.1", "aut.1",
             "sui.1", "gre.1", "den.1", "nor.1", "swe.1", "cze.1", "cyp.1", "isr.1", "rou.1")
CONMEBOL_POOL = ("arg.1", "bra.1", "col.1", "chi.1", "per.1", "ecu.1", "uru.1", "par.1", "bol.1", "ven.1")
CONCACAF_POOL = ("mex.1", "usa.1", "crc.1", "hon.1", "gua.1")
# Las copas de una misma confederación se modelan juntas: más cruces entre países calibran mejor las ligas.
UEFA_CUPS = ("uefa.champions", "uefa.europa", "uefa.europa.conf")
CONMEBOL_CUPS = ("conmebol.libertadores", "conmebol.sudamericana")

COMPETITIONS = [
    # Europa — con xG real (Understat)
    Competition("eng.1", "Premier League", "Europa", "EPL", aliases=("premier", "epl", "inglaterra"), lower=("eng.2",)),
    Competition("esp.1", "LaLiga", "Europa", "La_liga", aliases=("la liga", "liga espanola", "espana"), lower=("esp.2",)),
    Competition("ger.1", "Bundesliga", "Europa", "Bundesliga", aliases=("alemania",), lower=("ger.2",)),
    Competition("ita.1", "Serie A", "Europa", "Serie_A", aliases=("italia",), lower=("ita.2",)),
    Competition("fra.1", "Ligue 1", "Europa", "Ligue_1", aliases=("francia",), lower=("fra.2",)),
    Competition("rus.1", "Liga rusa", "Europa", "RFPL", aliases=("rfpl", "rusia")),
    # Europa — xG aproximado (ESPN)
    Competition("eng.2", "Championship", "Europa", aliases=("inglaterra 2",), lower=("eng.3",)),
    Competition("esp.2", "LaLiga 2", "Europa", aliases=("segunda division", "laliga hypermotion")),
    Competition("ger.2", "2. Bundesliga", "Europa"),
    Competition("ita.2", "Serie B", "Europa", lower=("ita.3",)),
    Competition("fra.2", "Ligue 2", "Europa"),
    Competition("por.1", "Primeira Liga", "Europa", aliases=("portugal", "liga portugal")),
    Competition("ned.1", "Eredivisie", "Europa", aliases=("holanda", "paises bajos"), lower=("ned.2",)),
    Competition("bel.1", "Pro League", "Europa", aliases=("belgica",)),
    Competition("tur.1", "Süper Lig", "Europa", aliases=("turquia", "super lig"), lower=("tur.2",)),
    Competition("sco.1", "Premiership escocesa", "Europa", aliases=("escocia",), lower=("sco.2",)),
    # Copas de Europa
    Competition("uefa.champions", "Champions League", "Copas internacionales", pool=UEFA_POOL + UEFA_CUPS[1:],
                aliases=("champions", "ucl", "liga de campeones")),
    Competition("uefa.europa", "Europa League", "Copas internacionales",
                pool=UEFA_POOL + (UEFA_CUPS[0], UEFA_CUPS[2]), aliases=("uel",)),
    Competition("uefa.europa.conf", "Conference League", "Copas internacionales", pool=UEFA_POOL + UEFA_CUPS[:2],
                aliases=("conference", "uecl")),
    # América
    Competition("mex.1", "Liga MX", "América", aliases=("mexico",)),
    Competition("usa.1", "MLS", "América", aliases=("estados unidos",)),
    Competition("arg.1", "Liga Profesional Argentina", "América", aliases=("argentina",), lower=("arg.2",)),
    Competition("bra.1", "Brasileirão", "América", aliases=("brasil", "brasileirao"), lower=("bra.2",)),
    Competition("col.1", "Liga BetPlay (Colombia)", "América", aliases=("colombia",), lower=("col.2",)),
    Competition("chi.1", "Primera División de Chile", "América", aliases=("chile",), lower=("chi.2",)),
    Competition("per.1", "Liga 1 (Perú)", "América", aliases=("peru",)),
    Competition("ecu.1", "LigaPro (Ecuador)", "América", aliases=("ecuador",)),
    Competition("uru.1", "Liga AUF (Uruguay)", "América", aliases=("uruguay",)),
    Competition("par.1", "Primera División de Paraguay", "América", aliases=("paraguay",)),
    # Copas de América
    Competition("conmebol.libertadores", "Copa Libertadores", "Copas internacionales",
                pool=CONMEBOL_POOL + CONMEBOL_CUPS[1:],
                aliases=("libertadores",)),
    Competition("conmebol.sudamericana", "Copa Sudamericana", "Copas internacionales",
                pool=CONMEBOL_POOL + CONMEBOL_CUPS[:1],
                aliases=("sudamericana",)),
    Competition("concacaf.champions", "Concacaf Champions Cup", "Copas internacionales", pool=CONCACAF_POOL,
                aliases=("concachampions",)),
    # Asia
    Competition("jpn.1", "J1 League", "Asia", aliases=("japon",)),
]
BY_CODE = {c.code: c for c in COMPETITIONS}
REGIONS = ["Europa", "América", "Copas internacionales", "Asia"]
DEFAULT_CODES = ["eng.1", "esp.1", "ger.1", "ita.1", "fra.1", "por.1", "ned.1", "mex.1", "arg.1", "bra.1",
                 "usa.1", "uefa.champions", "uefa.europa", "conmebol.libertadores", "conmebol.sudamericana"]


def resolve_competition(name: str) -> Competition:
    key = normalize(name)
    for c in COMPETITIONS:
        names = {normalize(c.code), normalize(c.name), *(normalize(a) for a in c.aliases)}
        if c.understat:
            names.add(normalize(c.understat))
        if key in names:
            return c
    close = difflib.get_close_matches(key, [normalize(c.name) for c in COMPETITIONS], n=1, cutoff=0.6)
    if close:
        return next(c for c in COMPETITIONS if normalize(c.name) == close[0])
    raise PredictionError(f"Competición no reconocida: '{name}'. Disponibles: "
                          + ", ".join(c.name for c in COMPETITIONS) + ".")


def signal_label(comp: Competition) -> str:
    return "xG" if comp.understat else "xG aproximado (tiros)"


def _fetch_espn_years(slugs: list[str], years: list[int], refresh: bool,
                      progress: Callable[[str], None] | None) -> tuple[pd.DataFrame, list]:
    current_year = utc_now().year
    session = requests.Session()
    jobs = [(slug, year) for slug in slugs for year in years]
    if progress:
        progress(f"Descargando {len(jobs)} archivos de ESPN ({', '.join(slugs[:3])}{'…' if len(slugs) > 3 else ''})...")

    def job(args):
        slug, year = args
        events, info = espn_source.fetch_year(session, slug, year, current_year, refresh)
        return espn_source.events_frame(events, slug), info

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        results = list(pool.map(job, jobs))
    frames = [f for f, _ in results if not f.empty]
    if not frames:
        raise PredictionError(f"ESPN no devolvió partidos de {', '.join(slugs)}.")
    frame = espn_source.disambiguate_names(pd.concat(frames, ignore_index=True))
    return frame.drop_duplicates("id"), [info for _, info in results]


def _season_index(dates: pd.Series) -> pd.Series:
    """Temporada europea (julio-junio) de cada fecha: 2026-03 → 2025."""
    return dates.dt.year - (dates.dt.month < 7)


def map_team_names(lower: pd.DataFrame, upper: pd.DataFrame,
                   min_similarity: float = MIN_MAPPING_SIMILARITY) -> dict[str, str]:
    """Empareja equipos de la división inferior (ESPN) con los de la superior (Understat):
    nombre muy parecido y temporadas sin solaparse (un club no juega en las dos a la vez).
    Devuelve {nombre ESPN: nombre Understat}."""
    def team_seasons(df: pd.DataFrame) -> dict[str, set[int]]:
        long = pd.concat([df[["home", "datetime"]].rename(columns={"home": "team"}),
                          df[["away", "datetime"]].rename(columns={"away": "team"})])
        long = long.assign(season=_season_index(long["datetime"]))
        return long.groupby("team")["season"].agg(set).to_dict()

    upper_seasons, lower_seasons = team_seasons(upper), team_seasons(lower)
    candidates = []
    for e_name, e_seasons in lower_seasons.items():
        scored = [(team_similarity(e_name, u), u) for u in upper_seasons]
        sim, best = max(scored) if scored else (0.0, None)
        if best is not None and sim >= min_similarity and not (e_seasons & upper_seasons[best]):
            candidates.append((sim, e_name, best))
    mapping, used = {}, set()
    for sim, e_name, u_name in sorted(candidates, reverse=True):  # uno a uno, primero los más parecidos
        if u_name not in used:
            mapping[e_name] = u_name
            used.add(u_name)
    return mapping


def attach_espn_stats(matches: pd.DataFrame, espn: pd.DataFrame, competition: str,
                      max_gap: pd.Timedelta = pd.Timedelta(hours=36), min_similarity: float = 0.5) -> pd.DataFrame:
    """Copia córners y tarjetas (`STAT_COLUMNS`) de los partidos de ESPN a los de la competición en
    `matches` (de otra fuente, p. ej. Understat): misma fecha (±`max_gap`, por husos y horarios
    distintos) y los dos nombres parecidos. Las filas sin pareja quedan con NaN."""
    out = matches.copy()
    for col in STAT_COLUMNS:
        if col not in out:
            out[col] = float("nan")
    espn = espn[espn["played"] & espn[STAT_COLUMNS].notna().any(axis=1)].sort_values("datetime")
    target = out[(out["competition"] == competition) & out["played"]].sort_values("datetime")
    if espn.empty or target.empty:
        return out
    times = target["datetime"].to_numpy()
    sims: dict[tuple[str, str], float] = {}

    def sim(a: str, b: str) -> float:
        if (a, b) not in sims:
            sims[(a, b)] = team_similarity(a, b)
        return sims[(a, b)]

    taken = set()
    for e in espn.itertuples(index=False):
        lo = times.searchsorted((e.datetime - max_gap).to_datetime64())
        hi = times.searchsorted((e.datetime + max_gap).to_datetime64(), side="right")
        best, best_score = None, min_similarity
        for idx, row in zip(target.index[lo:hi], target.iloc[lo:hi].itertuples(index=False)):
            score = min(sim(e.home, row.home), sim(e.away, row.away))
            if score >= best_score and idx not in taken:
                best, best_score = idx, score
        if best is not None:
            taken.add(best)
            out.loc[best, STAT_COLUMNS] = [getattr(e, c) for c in STAT_COLUMNS]
    return out


def espn_odds_codes(comp: Competition) -> tuple[str, ...]:
    """Competiciones cuya señal de mercado sale de las cuotas pasadas de ESPN: la propia y su pool
    (en copas), si football-data no las cubre. Una petición por partido de los últimos ~13 meses
    la primera vez; después, solo los partidos nuevos. La división inferior no: con cuotas solo
    recientes en ella, empeoraba la validación (Argentina, Países Bajos, Serie B)."""
    return tuple(c for c in (comp.code, *comp.pool) if not fd.has_odds(c))


def load_competition(comp: Competition, refresh: bool = False, progress: Callable[[str], None] | None = None,
                     include_lower: bool = True, with_market: bool = True, with_stats: bool = True) -> LeagueData:
    """Descarga los datos de la competición y prepara el modelo (sin entrenarlo). Con `with_market`,
    cruza las cuotas de cierre de football-data.co.uk (y las cuotas pasadas de ESPN donde
    football-data no llega, ver `espn_odds_codes`) como señal de mercado. Con `with_stats`, en las
    ligas de Understat cruza los córners y tarjetas de ESPN (en las de ESPN ya vienen)."""
    now = utc_now()
    lower = comp.lower if include_lower else ()
    if comp.understat:
        matches, sources, current = understat_source.load(comp.understat, comp.code, UNDERSTAT_SEASONS, refresh,
                                                          now, progress)
        since = pd.Timestamp(f"{current - UNDERSTAT_SEASONS + 1}-07-01")  # la más antigua: historial previo
        matches = matches[STANDARD_COLUMNS]
        if lower:  # historial de los recién ascendidos en la división inferior (xG aproximado de ESPN)
            years = list(range(since.year - 1, now.year + 1))
            lower_matches, lower_sources = _fetch_espn_years(list(lower), years, refresh, progress)
            names = map_team_names(lower_matches, matches)
            lower_matches = lower_matches.assign(home=lower_matches["home"].replace(names),
                                                 away=lower_matches["away"].replace(names))
            matches = pd.concat([matches, lower_matches[STANDARD_COLUMNS + STAT_COLUMNS]], ignore_index=True)
            sources = sources + lower_sources
        if with_stats:
            try:
                stats_years = list(range(now.year - ESPN_STATS_YEARS + 1, now.year + 1))
                espn_matches, _ = _fetch_espn_years([comp.code], stats_years, refresh, progress)
                matches = attach_espn_stats(matches, espn_matches, comp.code)
            except PredictionError:
                pass  # sin córners ni tarjetas: el modelo de goles no los necesita
        if with_market:
            matches, market_sources = market_signal.attach(matches, progress, espn_odds_codes(comp))
            sources = sources + market_sources
        return build_league_data(comp.code, comp.name, matches, sources, "xG", since)

    years_back = ESPN_CUP_YEARS if comp.is_cup else ESPN_LEAGUE_YEARS
    years = list(range(now.year - years_back + 1, now.year + 1))
    slugs = [comp.code, *comp.pool, *lower]
    matches, sources = _fetch_espn_years(slugs, years, refresh, progress)
    played = matches[matches["played"]]
    coverage = played["h_sig"].notna().mean() if len(played) else 0.0
    signal = "xG aproximado (tiros)" if coverage >= 0.5 else "goles"
    since = pd.Timestamp(f"{years[0]}-01-01") + pd.Timedelta(days=180)  # medio año de historial previo
    cups = frozenset(c.code for c in COMPETITIONS if c.is_cup)
    if with_market:
        matches, market_sources = market_signal.attach(matches, progress, espn_odds_codes(comp))
        sources = sources + market_sources
    return build_league_data(comp.code, comp.name, matches, sources, signal, since, train_on_focus=not comp.is_cup,
                             kappa_exclude=cups)


# --------------------------------------------------------------------------
# Calendario del día y cuotas (ESPN)
# --------------------------------------------------------------------------


def espn_days(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[str]:
    """Fechas de ESPN (agrupa por día de la costa este de EE. UU.) que cubren [start, end)."""
    first = start_utc.tz_localize("UTC").tz_convert("America/New_York").date()
    last = (end_utc - pd.Timedelta(seconds=1)).tz_localize("UTC").tz_convert("America/New_York").date()
    return [d.strftime("%Y%m%d") for d in pd.date_range(first, last, freq="D")]


def day_fixtures(comp: Competition, start_utc: pd.Timestamp, end_utc: pd.Timestamp,
                 session: requests.Session | None = None) -> pd.DataFrame:
    """Partidos de la competición con inicio en [start_utc, end_utc) según ESPN (con cuotas)."""
    session = session or requests.Session()
    events = [e for day in espn_days(start_utc, end_utc) for e in espn_source.fetch_day(session, comp.code, day)]
    frame = espn_source.events_frame(events, comp.code).drop_duplicates("id")
    return frame[(frame["datetime"] >= start_utc) & (frame["datetime"] < end_utc)].reset_index(drop=True)


def next_kickoff(comp: Competition, after_utc: pd.Timestamp, session: requests.Session | None = None
                 ) -> pd.Timestamp | None:
    """Hora (UTC) del próximo partido de la competición desde `after_utc`, según la jornada
    actual de ESPN (None si ESPN no muestra ninguno)."""
    events = espn_source.fetch_current(session or requests.Session(), comp.code)
    times = espn_source.events_frame(events, comp.code)["datetime"]
    times = times[times >= after_utc]
    return times.min() if len(times) else None


def match_to_source(espn_rows: pd.DataFrame, source_rows: pd.DataFrame,
                    max_gap: pd.Timedelta = pd.Timedelta(hours=3)) -> dict[str, str]:
    """Empareja partidos de ESPN con los de otra fuente (Understat) por hora y nombres.
    Devuelve {id ESPN: id fuente}; solo parejas con ambos nombres razonablemente parecidos."""
    pairs = {}
    for e in espn_rows.itertuples(index=False):
        cand = source_rows[(source_rows["datetime"] - e.datetime).abs() <= max_gap]
        best, best_score = None, 0.0
        for s in cand.itertuples(index=False):
            score = min(team_similarity(e.home, s.home), team_similarity(e.away, s.away))
            if score > best_score:
                best, best_score = s.id, score
        if best is not None and best_score >= 0.5:
            pairs[e.id] = best
    return pairs


def fixture_rows_for_model(comp: Competition, data: LeagueData, espn_day: pd.DataFrame) -> pd.DataFrame:
    """Partidos del día listos para `predict_match`: en ligas de ESPN, las filas de ESPN;
    en las de Understat, la fila de Understat (equipos con su nombre) con las cuotas y los ids de ESPN.
    Los partidos de ESPN que no se emparejan se devuelven con su nombre de ESPN."""
    if not comp.understat:
        return espn_day
    us = data.focus_matches
    pairs = match_to_source(espn_day, us)
    odds_cols = ["odds_h", "odds_d", "odds_a", "odds_over25", "odds_under25"]
    rows = []
    for e in espn_day.itertuples(index=False):
        if e.id in pairs:
            row = us[us["id"] == pairs[e.id]].iloc[0].copy()
            for col in odds_cols:
                row[col] = getattr(e, col)
            # ids de ESPN para consultar las alineaciones del partido
            row["espn_id"], row["espn_home_id"], row["espn_away_id"] = e.id, e.home_id, e.away_id
            rows.append(row)
        else:
            rows.append(pd.Series(e._asdict()))
    return pd.DataFrame(rows).reset_index(drop=True) if rows else espn_day.iloc[0:0]


def match_url(match_id: str | None) -> str | None:
    """Página del partido en su fuente (Understat o ESPN)."""
    if not match_id:
        return None
    source, _, raw = str(match_id).partition(":")
    if source == "us":
        return f"https://understat.com/match/{raw}"
    if source == "espn":
        return f"https://www.espn.com/soccer/match/_/gameId/{raw}"
    return None


def understat_fixture_with_odds(comp: Competition, data: LeagueData, match_id: str) -> pd.Series | None:
    """Fila de Understat del partido con las cuotas de ESPN (None si ESPN no lo tiene o no responde)."""
    row = data.focus_matches[data.focus_matches["id"] == match_id]
    if row.empty:
        return None
    kickoff = row["datetime"].iloc[0]
    try:
        day = day_fixtures(comp, kickoff - pd.Timedelta(hours=3), kickoff + pd.Timedelta(hours=3))
    except PredictionError:
        return None
    rows = fixture_rows_for_model(comp, data, day)
    match = rows[rows["id"] == match_id]
    return match.iloc[0] if len(match) else None


def predict_fixture(comp: Competition, model: LeagueModel, home: str, away: str) -> MatchPrediction:
    """`predict_match` con el contexto del partido del calendario (próximos días): las cuotas de ESPN
    en las ligas de Understat y, si ESPN ya publicó las alineaciones, el ajuste por rotaciones."""
    pred = predict_match(model, home, away)
    if pred.match_id is None:
        return pred
    if comp.understat:
        fixture = understat_fixture_with_odds(comp, model.data, pred.match_id)
    else:
        rows = model.data.focus_matches[model.data.focus_matches["id"] == pred.match_id]
        fixture = rows.iloc[0] if len(rows) else None
    if fixture is None:
        return pred
    lineup = lineups.fixture_lineups(comp.code, comp.is_cup, fixture)
    return predict_match(model, pred.home, pred.away, fixture=fixture, lineups=lineup)


def coverage_note(comp: Competition, data: LeagueData) -> str:
    """Texto corto de la señal usada (y, en copas, cuántas ligas alimentan el modelo)."""
    if comp.is_cup:
        return f"{data.signal_name} · junto a {sum(c not in UEFA_CUPS + CONMEBOL_CUPS for c in comp.pool)} ligas"
    return data.signal_name
