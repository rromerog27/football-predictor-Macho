"""Registro de ligas y copas, y carga de sus datos para el modelo.

- Las 6 ligas de Understat usan xG real; las demás, el xG aproximado con tiros
  de ESPN (o goles donde ESPN no publica tiros).
- Las copas internacionales (Champions, Libertadores...) se modelan junto con
  las ligas de sus participantes (su "pool"): así la fuerza de un equipo sale
  sobre todo de su liga, y los cruces entre países de la copa calibran unas
  ligas frente a otras.
- El calendario del día y las cuotas salen de ESPN en todas las competiciones
  (en las de Understat, las cuotas se cruzan por hora y nombre de equipo).
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pandas as pd
import requests

from src import espn_source, understat_source
from src.match_model import (
    STANDARD_COLUMNS,
    LeagueData,
    PredictionError,
    build_league_data,
    normalize,
    utc_now,
)

UNDERSTAT_SEASONS = 5  # temporadas anteriores a la actual (la más antigua es solo historial previo)
ESPN_LEAGUE_YEARS = 4  # años naturales de una liga de ESPN (incluido el actual)
ESPN_CUP_YEARS = 2  # años naturales del pool de una copa (muchas ligas: se limita la descarga)
MAX_WORKERS = 8


@dataclass(frozen=True)
class Competition:
    code: str  # identificador estable (slug de ESPN)
    name: str
    region: str
    understat: str | None = None  # código de Understat si hay xG real
    pool: tuple[str, ...] = ()  # copas: ligas (slugs de ESPN) que alimentan la fuerza de sus equipos
    aliases: tuple[str, ...] = ()

    @property
    def is_cup(self) -> bool:
        return bool(self.pool)


UEFA_POOL = ("eng.1", "esp.1", "ger.1", "ita.1", "fra.1", "por.1", "ned.1", "bel.1", "sco.1", "tur.1", "aut.1",
             "sui.1", "gre.1", "den.1", "nor.1", "swe.1", "cze.1", "cyp.1", "isr.1", "rou.1")
CONMEBOL_POOL = ("arg.1", "bra.1", "col.1", "chi.1", "per.1", "ecu.1", "uru.1", "par.1", "bol.1", "ven.1")
CONCACAF_POOL = ("mex.1", "usa.1", "crc.1", "hon.1", "gua.1")

COMPETITIONS = [
    # Europa — con xG real (Understat)
    Competition("eng.1", "Premier League", "Europa", "EPL", aliases=("premier", "epl", "inglaterra")),
    Competition("esp.1", "LaLiga", "Europa", "La_liga", aliases=("la liga", "liga espanola", "espana")),
    Competition("ger.1", "Bundesliga", "Europa", "Bundesliga", aliases=("alemania",)),
    Competition("ita.1", "Serie A", "Europa", "Serie_A", aliases=("italia",)),
    Competition("fra.1", "Ligue 1", "Europa", "Ligue_1", aliases=("francia",)),
    Competition("rus.1", "Liga rusa", "Europa", "RFPL", aliases=("rfpl", "rusia")),
    # Europa — xG aproximado (ESPN)
    Competition("eng.2", "Championship", "Europa", aliases=("inglaterra 2",)),
    Competition("esp.2", "LaLiga 2", "Europa", aliases=("segunda division", "laliga hypermotion")),
    Competition("ger.2", "2. Bundesliga", "Europa"),
    Competition("ita.2", "Serie B", "Europa"),
    Competition("fra.2", "Ligue 2", "Europa"),
    Competition("por.1", "Primeira Liga", "Europa", aliases=("portugal", "liga portugal")),
    Competition("ned.1", "Eredivisie", "Europa", aliases=("holanda", "paises bajos")),
    Competition("bel.1", "Pro League", "Europa", aliases=("belgica",)),
    Competition("tur.1", "Süper Lig", "Europa", aliases=("turquia", "super lig")),
    Competition("sco.1", "Premiership escocesa", "Europa", aliases=("escocia",)),
    # Copas de Europa
    Competition("uefa.champions", "Champions League", "Copas internacionales", pool=UEFA_POOL,
                aliases=("champions", "ucl", "liga de campeones")),
    Competition("uefa.europa", "Europa League", "Copas internacionales", pool=UEFA_POOL, aliases=("uel",)),
    Competition("uefa.europa.conf", "Conference League", "Copas internacionales", pool=UEFA_POOL,
                aliases=("conference", "uecl")),
    # América
    Competition("mex.1", "Liga MX", "América", aliases=("mexico",)),
    Competition("usa.1", "MLS", "América", aliases=("estados unidos",)),
    Competition("arg.1", "Liga Profesional Argentina", "América", aliases=("argentina",)),
    Competition("bra.1", "Brasileirão", "América", aliases=("brasil", "brasileirao")),
    Competition("col.1", "Liga BetPlay (Colombia)", "América", aliases=("colombia",)),
    Competition("chi.1", "Primera División de Chile", "América", aliases=("chile",)),
    Competition("per.1", "Liga 1 (Perú)", "América", aliases=("peru",)),
    Competition("ecu.1", "LigaPro (Ecuador)", "América", aliases=("ecuador",)),
    Competition("uru.1", "Liga AUF (Uruguay)", "América", aliases=("uruguay",)),
    Competition("par.1", "Primera División de Paraguay", "América", aliases=("paraguay",)),
    # Copas de América
    Competition("conmebol.libertadores", "Copa Libertadores", "Copas internacionales", pool=CONMEBOL_POOL,
                aliases=("libertadores",)),
    Competition("conmebol.sudamericana", "Copa Sudamericana", "Copas internacionales", pool=CONMEBOL_POOL,
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


def load_competition(comp: Competition, refresh: bool = False,
                     progress: Callable[[str], None] | None = None) -> LeagueData:
    """Descarga los datos de la competición y prepara el modelo (sin entrenarlo)."""
    now = utc_now()
    if comp.understat:
        matches, sources, current = understat_source.load(comp.understat, comp.code, UNDERSTAT_SEASONS, refresh,
                                                          now, progress)
        since = pd.Timestamp(f"{current - UNDERSTAT_SEASONS + 1}-07-01")  # la más antigua: historial previo
        return build_league_data(comp.code, comp.name, matches[STANDARD_COLUMNS], sources, "xG", since)

    years_back = ESPN_CUP_YEARS if comp.is_cup else ESPN_LEAGUE_YEARS
    years = list(range(now.year - years_back + 1, now.year + 1))
    slugs = [comp.code, *comp.pool]
    matches, sources = _fetch_espn_years(slugs, years, refresh, progress)
    played = matches[matches["played"]]
    coverage = played["h_sig"].notna().mean() if len(played) else 0.0
    signal = "xG aproximado (tiros)" if coverage >= 0.5 else "goles"
    since = pd.Timestamp(f"{years[0]}-01-01") + pd.Timedelta(days=180)  # medio año de historial previo
    return build_league_data(comp.code, comp.name, matches, sources, signal, since)


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


def _name_score(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_to_source(espn_rows: pd.DataFrame, source_rows: pd.DataFrame,
                    max_gap: pd.Timedelta = pd.Timedelta(hours=3)) -> dict[str, str]:
    """Empareja partidos de ESPN con los de otra fuente (Understat) por hora y nombres.
    Devuelve {id ESPN: id fuente}; solo parejas con ambos nombres razonablemente parecidos."""
    pairs = {}
    for e in espn_rows.itertuples(index=False):
        cand = source_rows[(source_rows["datetime"] - e.datetime).abs() <= max_gap]
        best, best_score = None, 0.0
        for s in cand.itertuples(index=False):
            score = min(_name_score(e.home, s.home), _name_score(e.away, s.away))
            if score > best_score:
                best, best_score = s.id, score
        if best is not None and best_score >= 0.5:
            pairs[e.id] = best
    return pairs


def fixture_rows_for_model(comp: Competition, data: LeagueData, espn_day: pd.DataFrame) -> pd.DataFrame:
    """Partidos del día listos para `predict_match`: en ligas de ESPN, las filas de ESPN;
    en las de Understat, la fila de Understat (equipos con su nombre) con las cuotas de ESPN.
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


def coverage_note(comp: Competition, data: LeagueData) -> str:
    """Texto corto de la señal usada (y, en copas, cuántas ligas alimentan el modelo)."""
    if comp.is_cup:
        return f"{data.signal_name} · junto a {len(comp.pool)} ligas"
    return data.signal_name
