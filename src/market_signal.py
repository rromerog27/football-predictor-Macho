"""Señal de mercado: goles esperados implícitos en las cuotas de cierre de partidos ya jugados.

Las cuotas de cierre resumen lo que el mercado sabe antes de cada partido
(fichajes, lesiones, alineaciones probables, motivación). De cada partido
jugado con cuotas en football-data.co.uk se despejan los goles esperados
λ local / visitante que reproducen su 1X2 (y su Over/Under 2.5 si lo hay) con
un Poisson con corrección de Dixon-Coles. Esos λ son una tercera señal para la
fuerza de cada equipo, junto al xG y los goles.

Solo entran cuotas de partidos anteriores al que se predice (las del propio
partido nunca): no hay fuga de información.

Los nombres de football-data ("Man United", "Ath Madrid") se emparejan con los
de la fuente del modelo por resultados: cada partido con la misma fecha (±1
día) y el mismo marcador es un "voto" para emparejar sus equipos, y cada club
acumula decenas de votos con su pareja correcta.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src import espn_source
from src import football_data_source as fd
from src.match_model import PredictionError, SourceInfo, utc_now

SOLVER_RHO = -0.05  # Dixon-Coles del despeje (con −0.05 el 1X2 del mercado se reproduce mejor que con 0)
TOTAL_PRIOR_WEIGHT = 0.1  # sin Over/Under, el total se acerca a la media de la liga (el empate solo es ruidoso)
SOLVER_ITERATIONS = 20
MIN_TEAM_VOTES = 3
MIN_VOTE_SHARE = 0.5  # votos mínimos de la pareja, en proporción a los partidos del equipo en el periodo común
MAX_DATE_GAP = pd.Timedelta(days=1)
ESPN_ODDS_DAYS = 400  # ESPN guarda cuotas de partidos pasados desde finales de 2025: no se piden más antiguas

_GOALS = np.arange(11)
_I, _J = np.indices((11, 11))


def _model_probs(lam_h: np.ndarray, lam_a: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, ...]:
    """P(local), P(visitante) y P(más goles que `line`) de un Poisson-Dixon-Coles, vectorizado."""
    m = poisson.pmf(_GOALS[None, :], lam_h[:, None])[:, :, None] * poisson.pmf(_GOALS[None, :], lam_a[:, None])[:, None, :]
    rho = SOLVER_RHO
    m[:, 0, 0] *= 1 - lam_h * lam_a * rho
    m[:, 0, 1] *= 1 + lam_h * rho
    m[:, 1, 0] *= 1 + lam_a * rho
    m[:, 1, 1] *= 1 - rho
    m /= m.sum(axis=(1, 2), keepdims=True)
    over = ((_I + _J).ravel()[None, :] > line[:, None]) * m.reshape(len(m), -1)
    return m[:, _I > _J].sum(1), m[:, _I < _J].sum(1), over.sum(1)


def implied_goals(p_home: np.ndarray, p_away: np.ndarray, p_over: np.ndarray,
                  prior_total: np.ndarray | float, over_line: np.ndarray | float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """λ local / visitante que reproducen las probabilidades del mercado (sin margen).
    Mínimos cuadrados (Gauss-Newton en log λ, todos los partidos a la vez) sobre P(local),
    P(visitante) y P(Over `over_line`); donde no hay Over/Under (`p_over` NaN), el total se
    acerca suavemente a `prior_total`."""
    p_home, p_away, p_over = (np.asarray(v, float) for v in (p_home, p_away, p_over))
    n = len(p_home)
    prior = np.broadcast_to(np.asarray(prior_total, float), (n,))
    line = np.broadcast_to(np.asarray(over_line, float), (n,))
    has_over = np.isfinite(p_over)
    target_over = np.nan_to_num(p_over)
    x = np.column_stack([np.log(prior * 0.55), np.log(prior * 0.45)])

    def residuals(x: np.ndarray) -> np.ndarray:
        lh, la = np.exp(x[:, 0]), np.exp(x[:, 1])
        ph, pa, po = _model_probs(lh, la, line)
        third = np.where(has_over, po - target_over, TOTAL_PRIOR_WEIGHT * np.log((lh + la) / prior))
        return np.column_stack([ph - p_home, pa - p_away, third])

    eps = 1e-5
    for _ in range(SOLVER_ITERATIONS):
        r = residuals(x)
        j0 = (residuals(x + [eps, 0.0]) - r) / eps
        j1 = (residuals(x + [0.0, eps]) - r) / eps
        a, b, d = (j0 * j0).sum(1) + 1e-9, (j0 * j1).sum(1), (j1 * j1).sum(1) + 1e-9
        g0, g1 = (j0 * r).sum(1), (j1 * r).sum(1)
        det = a * d - b * b
        step = np.column_stack([-(d * g0 - b * g1) / det, -(a * g1 - b * g0) / det])
        x = x + np.clip(step, -0.5, 0.5)
    return np.exp(x[:, 0]), np.exp(x[:, 1])


def map_teams(ours: pd.DataFrame, theirs: pd.DataFrame) -> dict[str, str]:
    """{nombre en football-data: nombre propio} por coincidencia de resultados (fecha ±1 día y
    marcador). Ambos DataFrames con columnas date, home, away, hg, ag. Uno a uno, de más a
    menos votos; solo parejas con votos suficientes para descartar coincidencias al azar."""
    cand = theirs.merge(ours, on=["hg", "ag"], suffixes=("", "_ours"))
    cand = cand[(cand["date"] - cand["date_ours"]).abs() <= MAX_DATE_GAP]
    votes = Counter(zip(cand["home"], cand["home_ours"])) + Counter(zip(cand["away"], cand["away_ours"]))
    lo, hi = ours["date"].min() - MAX_DATE_GAP, ours["date"].max() + MAX_DATE_GAP
    period = theirs[(theirs["date"] >= lo) & (theirs["date"] <= hi)]
    games = Counter(period["home"]) + Counter(period["away"])
    mapping, used = {}, set()
    for (their, our), n in sorted(votes.items(), key=lambda kv: -kv[1]):
        if their in mapping or our in used:
            continue
        if n >= max(MIN_TEAM_VOTES, MIN_VOTE_SHARE * games.get(their, 0)):
            mapping[their] = our
            used.add(our)
    return mapping


def _competition_market(rows: pd.DataFrame, code: str) -> tuple[pd.DataFrame, list[SourceInfo]]:
    """λ de mercado de los partidos jugados de una competición: DataFrame indexado como
    `rows` con columnas h_mkt, a_mkt (solo los partidos cruzados con cuotas) y archivos usados."""
    ours = pd.DataFrame({"date": rows["datetime"].dt.normalize(), "home": rows["home"], "away": rows["away"],
                         "hg": rows["hg"], "ag": rows["ag"], "row": rows.index})
    theirs, sources = fd.load_with_sources(code, ours["date"].min(), ours["date"].max())
    theirs = theirs.dropna(subset=["odds_h", "odds_d", "odds_a"])
    if theirs.empty:
        return pd.DataFrame(columns=["h_mkt", "a_mkt"]), sources
    names = map_teams(ours, theirs)
    theirs = theirs.assign(home=theirs["home"].map(names), away=theirs["away"].map(names)).dropna(subset=["home", "away"])
    joined = theirs.merge(ours, on=["home", "away", "hg", "ag"], suffixes=("", "_ours"))
    joined = joined[(joined["date"] - joined["date_ours"]).abs() <= MAX_DATE_GAP]
    joined = joined.drop_duplicates("row").drop_duplicates(["date", "home", "away"])
    if joined.empty:
        return pd.DataFrame(columns=["h_mkt", "a_mkt"]), sources
    inv = 1 / joined[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    p = inv / inv.sum(axis=1, keepdims=True)
    inv_ou = 1 / joined[["odds_over25", "odds_under25"]].to_numpy(float)
    p_over = inv_ou[:, 0] / inv_ou.sum(axis=1)  # NaN si no hay Over/Under
    prior_total = float((rows["hg"] + rows["ag"]).mean())
    lam_h, lam_a = implied_goals(p[:, 0], p[:, 2], p_over, prior_total)
    return pd.DataFrame({"h_mkt": lam_h, "a_mkt": lam_a}, index=joined["row"].to_numpy()), sources


def espn_market(rows: pd.DataFrame, code: str) -> tuple[pd.DataFrame, SourceInfo | None]:
    """λ de mercado de partidos jugados de ESPN (ids "espn:...") de los últimos ESPN_ODDS_DAYS
    días, con las cuotas previas que ESPN guarda en la ficha de cada partido. Columnas h_mkt,
    a_mkt y las cuotas 1X2 (odds_h, odds_d, odds_a), indexado como `rows`; y la fuente."""
    columns = ["h_mkt", "a_mkt", "odds_h", "odds_d", "odds_a"]
    recent = rows[rows["id"].astype(str).str.startswith("espn:")
                  & (rows["datetime"] >= utc_now() - pd.Timedelta(days=ESPN_ODDS_DAYS))]
    if recent.empty:
        return pd.DataFrame(columns=columns), None
    odds, from_cache = espn_source.fetch_past_odds(code, [i.removeprefix("espn:") for i in recent["id"].astype(str)])
    found = [(idx, o) for idx, i in zip(recent.index, recent["id"].astype(str)) if (o := odds.get(i.removeprefix("espn:")))]
    source = SourceInfo(f"{espn_source.ESPN}/{code}/summary?event=<id>", "cuotas previas (DraftKings)", len(found),
                        from_cache)
    if not found:
        return pd.DataFrame(columns=columns), source
    index = [idx for idx, _ in found]
    o1x2 = np.array([[o["home"], o["draw"], o["away"]] for _, o in found], float)
    inv = 1 / o1x2
    p = inv / inv.sum(axis=1, keepdims=True)
    ou = np.array([[o.get("over") or np.nan, o.get("under") or np.nan, o.get("line") or np.nan] for _, o in found], float)
    p_over = np.where(np.isfinite(ou).all(axis=1), (1 / ou[:, 0]) / (1 / ou[:, 0] + 1 / ou[:, 1]), np.nan)
    prior_total = float((rows["hg"] + rows["ag"]).mean())
    lam_h, lam_a = implied_goals(p[:, 0], p[:, 2], p_over, prior_total, np.nan_to_num(ou[:, 2], nan=2.5))
    return pd.DataFrame({"h_mkt": lam_h, "a_mkt": lam_a, "odds_h": o1x2[:, 0], "odds_d": o1x2[:, 1],
                         "odds_a": o1x2[:, 2]}, index=index), source


def attach(matches: pd.DataFrame, progress: Callable[[str], None] | None = None,
           espn_codes: tuple[str, ...] = ()) -> tuple[pd.DataFrame, list[SourceInfo]]:
    """Añade h_mkt / a_mkt (λ implícitos en las cuotas de cierre; NaN sin cuotas) a los partidos
    jugados de las competiciones que football-data cubre y, en `espn_codes`, con las cuotas
    pasadas de ESPN (partidos recientes). Si una fuente falla, esos partidos quedan sin señal
    de mercado (el modelo sigue con xG y goles). Devuelve (partidos, fuentes de cuotas)."""
    matches = matches.assign(h_mkt=np.nan, a_mkt=np.nan)
    sources: list[SourceInfo] = []
    played = matches[matches["played"] & matches["hg"].notna() & matches["ag"].notna()]
    codes = [c for c in played["competition"].unique() if fd.has_odds(c)]
    if progress and codes:
        progress(f"Cruzando cuotas de cierre de football-data.co.uk ({len(codes)} competiciones)...")
    for code in codes:
        try:
            market, used = _competition_market(played[played["competition"] == code], code)
        except PredictionError:
            continue
        matches.loc[market.index, ["h_mkt", "a_mkt"]] = market[["h_mkt", "a_mkt"]].to_numpy()
        sources += used
    for code in espn_codes:
        rows = played[(played["competition"] == code) & matches.loc[played.index, "h_mkt"].isna()]
        if rows.empty:
            continue
        if progress:
            progress(f"Cuotas de partidos pasados de ESPN ({code})...")
        market, source = espn_market(rows, code)
        matches.loc[market.index, ["h_mkt", "a_mkt"]] = market[["h_mkt", "a_mkt"]].to_numpy()
        if source is not None:
            sources.append(source)
    return matches, sources
