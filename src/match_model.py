"""Modelo de predicción de partidos: Poisson (Dixon-Coles) + regresión logística.

Núcleo independiente de la fuente de datos, compartido por la página
"Partidos del día" y el script `modelo_prediccion.py`. Recibe un DataFrame de
partidos con columnas estándar (ver `STANDARD_COLUMNS`) que preparan
`src/understat_source.py` y `src/espn_source.py`, orquestados por
`src/competitions.py`.

La fuerza de cada equipo se estima con hasta tres señales y se mezclan:

- la **señal de calidad de ocasiones**: xG real (Understat) o, si no hay, un
  xG aproximado a partir de tiros a puerta y tiros fuera (ESPN);
- los **goles**;
- el **mercado**: goles esperados implícitos en las cuotas de cierre de los
  partidos anteriores (`src/market_signal.py`), donde las hay.

Para cada señal, el ataque/defensa de cada equipo se ajusta por la calidad de
sus rivales (ajuste proporcional iterativo sobre los últimos 12 meses, con un
suavizado bayesiano de 2 partidos "promedio") y se mezcla con su forma en los
últimos 10 partidos. El peso de la forma reciente y el de cada señal se eligen
con datos (máxima verosimilitud de los goles reales en el periodo de
entrenamiento).

Modelos:
- Poisson con corrección de Dixon-Coles (ρ por máxima verosimilitud): matriz
  de marcadores → 1X2, Over/Under 2.5, ambos anotan.
- Regresión logística multinomial: reajusta la señal de Poisson con la racha
  de puntos, la localía y el descanso.
- Ensemble: el peso de cada modelo minimiza el log loss en el 30% más reciente
  del histórico, que la regresión logística no vio al entrenar.
"""

from __future__ import annotations

import difflib
import unicodedata
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columnas que deben traer los partidos (horas en UTC sin zona; NaN donde no aplica).
STANDARD_COLUMNS = [
    "id", "competition", "season", "datetime", "home", "away", "played", "extra_time", "neutral",
    "hg", "ag", "h_sig", "a_sig", "odds_h", "odds_d", "odds_a", "odds_over25", "odds_under25",
]

N_RECENT = 10  # partidos recientes que definen la forma
N_FORM = 5  # partidos para la racha de puntos
WINDOW_DAYS = 365  # ventana de la fuerza "de fondo" (ajustada por rival)
MIN_WINDOW_MATCHES = 50  # partidos en la ventana necesarios para ajustar las fuerzas
MIN_MATCHES = 5  # mínimo de partidos de cada equipo para usar un partido al entrenar
PRIOR_MATCHES = 2.0  # suavizado: cada equipo arranca con 2 partidos "de media de la liga"
DECAY_HALF_LIFE_DAYS: float | None = 120.0  # peso de los partidos de la ventana: se reduce a la mitad cada 120 días
IPF_ITERATIONS = 25
MAX_GOALS = 10
LAMBDA_MIN, LAMBDA_MAX = 0.1, 5.0
STRENGTH_FLOOR = 0.05  # una racha sin goles no debe dar fuerza 0 (log -inf en la mezcla)
# Peso (en "partidos equivalentes") de la fuerza de 12 meses frente a los últimos 10.
SHRINK_GRID = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 1e9)
# Peso de la señal (xG / xG aproximado) frente a los goles en la fuerza.
SIGNAL_WEIGHT_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0)
# Peso de la señal de mercado (goles esperados implícitos en las cuotas de cierre de partidos
# anteriores) frente a la mezcla de señal y goles.
MARKET_WEIGHT_GRID = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# Uso de la señal de mercado según los partidos con cuotas: con cuotas en casi todo el histórico, su
# peso se aprende con datos; si solo hay cuotas recientes (ESPN las guarda desde finales de 2025), el
# entrenamiento no puede estimarlo y se usa el peso típico de las ligas con histórico completo.
MARKET_LEARN_COVERAGE = 0.6  # partidos con cuotas desde el inicio del entrenamiento
MARKET_RECENT_COVERAGE = 0.5  # partidos con cuotas en los últimos 12 meses
MARKET_WEIGHT_DEFAULT = 0.7
RHO_BOUNDS = (-0.2, 0.2)
# Peso del total de goles propio del partido frente a la media de la competición (1 = sin acercar).
TOTAL_SHRINK_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
VAL_FRACTION = 0.3
KAPPA_PRIOR_MATCHES = 20  # suavizado de la calibración de goles por competición (partidos de "media")
FOCUS_MIN_VAL = 150  # partidos de validación de la propia competición para evaluarla solo con ellos
# "Recién llegado": menos de estos partidos en la competición en los últimos 12 meses (ascendidos).
# Su fuerza sale de pocos partidos o de otra división, con un sesgo de nivel que se calibra con datos.
NEWCOMER_MATCHES = 10
MIN_TRAIN_ROWS, MIN_VAL_ROWS = 200, 60
ENSEMBLE_WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
CLASSES = ["H", "D", "A"]
# Un partido del calendario más lejano que esto (p. ej. la vuelta, meses después) no fija el corte:
# se predice con los datos de hoy en lugar de proyectar el descanso hasta esa fecha.
FIXTURE_HORIZON_DAYS = 14

NUMERIC_FEATURES = [
    "log_lambda_ratio",
    "log_lambda_total",
    "poisson_home_vs_draw",
    "poisson_away_vs_draw",
    "ppg5_home",
    "ppg5_away",
    "ppg5_home_at_home",
    "ppg5_away_away",
]
CATEGORICAL_FEATURES = ["rest_home", "rest_away"]

# Abreviaturas que la búsqueda por subcadena no resuelve sola.
TEAM_ALIASES = {
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "man city": "Manchester City",
    "spurs": "Tottenham",
    "wolves": "Wolverhampton Wanderers",
    "nottm forest": "Nottingham Forest",
    "nott'm forest": "Nottingham Forest",
    "forest": "Nottingham Forest",
    "barca": "Barcelona",
    "atleti": "Atletico Madrid",
    "psg": "Paris Saint Germain",
    "inter milan": "Inter",
    "gladbach": "Borussia M.Gladbach",
}


class PredictionError(RuntimeError):
    """Error mostrable al usuario (competición/equipo desconocido, fuente caída, datos insuficientes)."""


@dataclass
class SourceInfo:
    url: str
    label: str  # p. ej. "2025/26" o "2026"
    played: int
    from_cache: bool


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(text.lower().replace("-", " ").replace(".", " ").split())


# Palabras genéricas que no distinguen equipos al comparar nombres entre fuentes.
NAME_STOPWORDS = {"fc", "cf", "afc", "sc", "ac", "cd", "club", "de", "del", "la", "el", "the", "ud", "sd", "ca",
                  "fk", "sk", "if", "bk", "ss", "us", "as", "cp"}


def _name_tokens(name: str) -> list[str]:
    name = TEAM_ALIASES.get(normalize(name), name)
    return [t for t in normalize(name).replace("'", "").split() if t not in NAME_STOPWORDS]


def team_similarity(a: str, b: str) -> float:
    """Parecido (0-1) entre dos nombres del mismo equipo en fuentes distintas
    ("Club Leon" ~ "León", "Man United" ~ "Manchester United", "Leeds" ~ "Leeds United")."""
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return 0.0
    ja, jb = " ".join(ta), " ".join(tb)
    if ja == jb:
        return 1.0
    if ja in jb or jb in ja:
        return 0.95
    overlap = len(set(ta) & set(tb)) / min(len(set(ta)), len(set(tb)))
    return max(0.9 * overlap, difflib.SequenceMatcher(None, ja, jb).ratio())


def resolve_team(name: str, teams: list[str]) -> str:
    query = normalize(name)
    by_norm = {normalize(t): t for t in teams}
    if query in by_norm:
        return by_norm[query]
    alias = TEAM_ALIASES.get(query)
    if alias in teams:
        return alias
    partial = [t for n, t in by_norm.items() if query in n or n in query]
    if len(partial) == 1:
        return partial[0]
    close = difflib.get_close_matches(query, list(by_norm), n=1, cutoff=0.6)
    if close:
        return by_norm[close[0]]
    raise PredictionError(f"Equipo no encontrado: '{name}'. Equipos disponibles: {', '.join(sorted(teams))}.")


def team_long_frame(played: pd.DataFrame) -> pd.DataFrame:
    """Una fila por equipo y partido jugado (vista desde ese equipo)."""
    common = {"id": played["id"], "season": played["season"], "datetime": played["datetime"],
              "neutral": played["neutral"], "competition": played["competition"]}
    h_mkt = played["h_mkt"] if "h_mkt" in played else np.nan
    a_mkt = played["a_mkt"] if "a_mkt" in played else np.nan
    common["mkt_real"] = played["mkt_real"] if "mkt_real" in played else False  # λ de cuotas (no relleno)
    home = pd.DataFrame(
        {**common, "team": played["home"], "opp": played["away"], "venue": "h",
         "gf": played["hg"], "ga": played["ag"], "sf": played["h_sig"], "sa": played["a_sig"],
         "mf": h_mkt, "ma": a_mkt}
    )
    away = pd.DataFrame(
        {**common, "team": played["away"], "opp": played["home"], "venue": "a",
         "gf": played["ag"], "ga": played["hg"], "sf": played["a_sig"], "sa": played["h_sig"],
         "mf": a_mkt, "ma": h_mkt}
    )
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.select([long["gf"] > long["ga"], long["gf"] == long["ga"]], [3, 1], 0)
    return long.sort_values(["datetime", "id"]).reset_index(drop=True)


class MatchIndex:
    """Partidos jugados indexados por fecha y por equipo (con ids enteros), para
    cortar el historial anterior a cualquier instante sin filtrar DataFrames.
    Con `has_signal=False` la señal es la propia columna de goles; con `has_market=True`
    también se ajusta la fuerza según el mercado (columnas h_mkt / a_mkt, sin NaN)."""

    def __init__(self, played: pd.DataFrame, has_signal: bool = True, focus: str | None = None,
                 has_market: bool = False):
        self.played = played.sort_values(["datetime", "id"]).reset_index(drop=True)
        self.has_signal = has_signal
        self.has_market = has_market
        self.focus = focus  # competición que se predice (cuenta los partidos "en la competición")
        self.teams = sorted(set(self.played["home"]) | set(self.played["away"]))
        self.team_id = {t: i for i, t in enumerate(self.teams)}
        p = self.played
        self.times = p["datetime"].to_numpy()
        self.arrays = {
            "h": np.array([self.team_id[t] for t in p["home"]], dtype=int),
            "a": np.array([self.team_id[t] for t in p["away"]], dtype=int),
            "neutral": p["neutral"].to_numpy(bool),
            "sig_h": p["h_sig"].to_numpy(float), "sig_a": p["a_sig"].to_numpy(float),
            "gls_h": p["hg"].to_numpy(float), "gls_a": p["ag"].to_numpy(float),
        }
        if has_market:
            self.arrays["mkt_h"], self.arrays["mkt_a"] = p["h_mkt"].to_numpy(float), p["a_mkt"].to_numpy(float)
        long = team_long_frame(p)
        self.team_rows = {team: rows.reset_index(drop=True) for team, rows in long.groupby("team", sort=False)}
        self.team_arrays = {
            team: {
                "times": rows["datetime"].to_numpy(),
                "opp": np.array([self.team_id[o] for o in rows["opp"]], dtype=int),
                "home": (rows["venue"] == "h").to_numpy(),
                "neutral": rows["neutral"].to_numpy(bool),
                "sig_for": rows["sf"].to_numpy(float),
                "sig_against": rows["sa"].to_numpy(float),
                "gls_for": rows["gf"].to_numpy(float),
                "gls_against": rows["ga"].to_numpy(float),
                "mkt_for": rows["mf"].to_numpy(float),
                "mkt_against": rows["ma"].to_numpy(float),
                "mkt_real": rows["mkt_real"].to_numpy(bool),
                "pts": rows["pts"].to_numpy(float),
                "in_focus": (rows["competition"] == focus).to_numpy() if focus else np.ones(len(rows), bool),
            }
            for team, rows in self.team_rows.items()
        }

    def window_slice(self, cutoff: pd.Timestamp) -> slice:
        start = np.searchsorted(self.times, (cutoff - pd.Timedelta(days=WINDOW_DAYS)).to_datetime64(), "left")
        end = np.searchsorted(self.times, cutoff.to_datetime64(), "left")
        return slice(int(start), int(end))

    def window(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        return self.played.iloc[self.window_slice(cutoff)]


# --------------------------------------------------------------------------
# Fuerza de ataque/defensa
# --------------------------------------------------------------------------


@dataclass
class LeagueRatings:
    index: dict[str, int]
    mu_home: float  # señal media del local en la ventana (partidos no neutrales)
    mu_away: float  # señal media del visitante
    goals_home: float  # goles reales medios del local (nivel de los λ)
    goals_away: float  # goles reales medios del visitante
    attack: np.ndarray  # 1.0 = media; >1 genera más que la media
    defense: np.ndarray  # 1.0 = media; >1 concede más que la media

    def venue_means(self, home: np.ndarray, neutral: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Señal esperada a favor/en contra de una media de la liga según sede."""
        mid = (self.mu_home + self.mu_away) / 2
        mu_for = np.where(neutral, mid, np.where(home, self.mu_home, self.mu_away))
        mu_against = np.where(neutral, mid, np.where(home, self.mu_away, self.mu_home))
        return mu_for, mu_against


def fit_ratings_arrays(h: np.ndarray, a: np.ndarray, hs: np.ndarray, as_: np.ndarray, neutral: np.ndarray,
                       hg: np.ndarray, ag: np.ndarray, index: dict[str, int],
                       weights: np.ndarray | None = None) -> LeagueRatings:
    """Modelo multiplicativo señal = μ_sede · ataque · defensa_rival, ajustado por
    ajuste proporcional iterativo (máxima verosimilitud de Poisson sobre la
    señal), con un suavizado de PRIOR_MATCHES partidos de media de la liga que
    evita fuerzas extremas (o nulas) con pocos partidos. `h`/`a` son ids de
    `index`; los equipos sin partidos en la ventana quedan en 1.0."""
    n = len(index)
    regular = ~neutral if (~neutral).any() else np.ones_like(neutral)
    mu_h, mu_a = hs[regular].mean(), as_[regular].mean()
    mid = (mu_h + mu_a) / 2
    mh = np.where(neutral, mid, mu_h)  # media esperada del "local" de cada partido
    ma = np.where(neutral, mid, mu_a)

    wt = np.ones(len(h)) if weights is None else weights  # peso de cada partido (decaimiento temporal)
    prior = PRIOR_MATCHES * mid
    sig_for = np.bincount(h, wt * hs, n) + np.bincount(a, wt * as_, n) + prior
    sig_against = np.bincount(h, wt * as_, n) + np.bincount(a, wt * hs, n) + prior
    attack, defense = np.ones(n), np.ones(n)
    for _ in range(IPF_ITERATIONS):
        attack = sig_for / (np.bincount(h, wt * mh * defense[a], n) + np.bincount(a, wt * ma * defense[h], n) + prior)
        defense = sig_against / (np.bincount(h, wt * ma * attack[a], n) + np.bincount(a, wt * mh * attack[h], n) + prior)
    games = np.bincount(h, wt, n) + np.bincount(a, wt, n)
    scale = np.average(attack, weights=games)
    return LeagueRatings(index, float(mu_h), float(mu_a), float(hg[regular].mean()), float(ag[regular].mean()),
                         attack / scale, defense * scale)


def fit_ratings(window: pd.DataFrame, h_col: str, a_col: str) -> LeagueRatings:
    """`fit_ratings_arrays` para un DataFrame de partidos (columnas home, away, neutral, hg, ag)."""
    teams = sorted(set(window["home"]) | set(window["away"]))
    index = {t: i for i, t in enumerate(teams)}
    neutral = window["neutral"].to_numpy(bool) if "neutral" in window else np.zeros(len(window), bool)
    return fit_ratings_arrays(window["home"].map(index).to_numpy(), window["away"].map(index).to_numpy(),
                              window[h_col].to_numpy(float), window[a_col].to_numpy(float), neutral,
                              window["hg"].to_numpy(float), window["ag"].to_numpy(float), index)


def rest_category(days: float) -> str:
    if np.isnan(days) or days >= 8:
        return "largo"
    return "corto" if days <= 4 else "normal"


def _recent_strength(arrays: dict, recent: slice, ratings: LeagueRatings, kind: str) -> tuple[float, float]:
    """Ataque/defensa de los partidos dados: señal real / señal esperada según sede y rival."""
    opp = arrays["opp"][recent]
    mu_for, mu_against = ratings.venue_means(arrays["home"][recent], arrays["neutral"][recent])
    att = arrays[f"{kind}_for"][recent].sum() / (mu_for * ratings.defense[opp]).sum()
    dfn = arrays[f"{kind}_against"][recent].sum() / (mu_against * ratings.attack[opp]).sum()
    return float(att), float(dfn)


def team_snapshot(index: MatchIndex, team: str, venue: str, cutoff: pd.Timestamp,
                  ratings: dict[str, LeagueRatings], with_rows: bool = False) -> dict | None:
    """Estado de un equipo justo antes de `cutoff` (solo partidos anteriores)."""
    arrays = index.team_arrays.get(team)
    if arrays is None or team not in ratings["sig"].index:
        return None
    times = arrays["times"]
    end = int(np.searchsorted(times, cutoff.to_datetime64(), "left"))
    start = int(np.searchsorted(times, (cutoff - pd.Timedelta(days=WINDOW_DAYS)).to_datetime64(), "left"))
    if end <= start:
        return None

    recent = slice(max(start, end - N_RECENT), end)
    form = slice(max(start, end - N_FORM), end)
    same_venue = np.flatnonzero(arrays["home"][start:end] == (venue == "h"))[-N_FORM:] + start
    pts = arrays["pts"]
    snap = {
        "n_window": end - start,
        "n_focus": int(arrays["in_focus"][start:end].sum()),
        "n_recent": recent.stop - recent.start,
        "ppg5": float(pts[form].mean()),
        "ppg5_venue": float(pts[same_venue].mean()) if len(same_venue) else float(pts[form].mean()),
        "rest_days": float((cutoff.to_datetime64() - times[end - 1]) / np.timedelta64(1, "D")),
        "mkt_share": float(arrays["mkt_real"][recent].mean()),  # partidos recientes con cuotas
    }
    for kind, r in ratings.items():
        att, dfn = _recent_strength(arrays, recent, r, kind)
        snap[f"att_recent_{kind}"], snap[f"def_recent_{kind}"] = att, dfn
        snap[f"att_long_{kind}"] = float(r.attack[r.index[team]])
        snap[f"def_long_{kind}"] = float(r.defense[r.index[team]])
    if with_rows:
        rows = index.team_rows[team]
        snap["recent_rows"] = rows.iloc[recent]
        snap["form_rows"] = rows.iloc[form]
        snap["venue_rows"] = rows.iloc[same_venue]
    return snap


def window_ratings(index: MatchIndex, cutoff: pd.Timestamp, cache: dict) -> dict[str, LeagueRatings] | None:
    """Fuerzas de todos los equipos con los partidos de los 12 meses anteriores al
    día de `cutoff` (se recalculan una vez por día: los partidos del mismo día no entran)."""
    day = cutoff.normalize()
    if day not in cache:
        w = index.window_slice(day)
        if w.stop - w.start < MIN_WINDOW_MATCHES:
            cache[day] = None
        else:
            arr = {k: v[w] for k, v in index.arrays.items()}
            common = (arr["h"], arr["a"])
            weights = None
            if DECAY_HALF_LIFE_DAYS:
                age = (day.to_datetime64() - index.times[w]) / np.timedelta64(1, "D")
                weights = 0.5 ** (age / DECAY_HALF_LIFE_DAYS)
            rest = (arr["neutral"], arr["gls_h"], arr["gls_a"], index.team_id, weights)
            sig = fit_ratings_arrays(*common, arr["sig_h"], arr["sig_a"], *rest)
            gls = fit_ratings_arrays(*common, arr["gls_h"], arr["gls_a"], *rest) if index.has_signal else sig
            cache[day] = {"sig": sig, "gls": gls}
            if index.has_market:
                cache[day]["mkt"] = fit_ratings_arrays(*common, arr["mkt_h"], arr["mkt_a"], *rest)
    return cache[day]


def match_snapshot(index: MatchIndex, home: str, away: str, cutoff: pd.Timestamp, ratings_cache: dict,
                   with_rows: bool = False) -> tuple[dict, dict, dict[str, LeagueRatings]] | None:
    ratings = window_ratings(index, cutoff, ratings_cache)
    if ratings is None:
        return None
    h = team_snapshot(index, home, "h", cutoff, ratings, with_rows)
    a = team_snapshot(index, away, "a", cutoff, ratings, with_rows)
    if h is None or a is None:
        return None
    return h, a, ratings


def blended(snap: dict, shrink: float, kind: str, side: str) -> float:
    """Fuerza (att/def) de una señal: forma reciente mezclada con la de 12 meses."""
    w = snap["n_recent"] / (snap["n_recent"] + shrink)
    return w * snap[f"{side}_recent_{kind}"] + (1 - w) * snap[f"{side}_long_{kind}"]


def strength(snap: dict, shrink: float, signal_weight: float, side: str, market_weight: float = 0.0) -> float:
    """Fuerza final: media geométrica ponderada de la de la señal y la de goles y, con
    `market_weight` > 0, de esa mezcla con la del mercado (en proporción a sus partidos recientes
    con cuotas)."""
    sig = max(blended(snap, shrink, "sig", side), STRENGTH_FLOOR)
    gls = max(blended(snap, shrink, "gls", side), STRENGTH_FLOOR)
    log_s = signal_weight * np.log(sig) + (1 - signal_weight) * np.log(gls)
    if market_weight:
        mkt = max(blended(snap, shrink, "mkt", side), STRENGTH_FLOOR)
        mw = market_weight * snap["mkt_share"]  # sin cuotas recientes, la fuerza de mercado es solo relleno
        log_s = mw * np.log(mkt) + (1 - mw) * log_s
    return float(np.exp(log_s))


def base_goals(goals_home, goals_away, neutral):
    """Goles medios de la liga por sede (en campo neutral, la media de ambos)."""
    mid = (np.asarray(goals_home) + np.asarray(goals_away)) / 2
    return np.where(neutral, mid, goals_home), np.where(neutral, mid, goals_away)


def expected_goals(h: dict, a: dict, ratings: dict[str, LeagueRatings], shrink: float,
                   signal_weight: float, neutral: bool = False, market_weight: float = 0.0) -> tuple[float, float]:
    """λ = goles medios reales de la liga (por sede) × ataque propio × defensa rival.
    La señal y los goles fijan la fuerza relativa; el nivel sale de los goles
    reales (el xG de Understat, por ejemplo, va por encima de los goles marcados)."""
    gh, ga = base_goals(ratings["gls"].goals_home, ratings["gls"].goals_away, neutral)
    s = {(name, side): strength(snap, shrink, signal_weight, side, market_weight)
         for name, snap in (("h", h), ("a", a)) for side in ("att", "def")}
    lam_h = gh * s[("h", "att")] * s[("a", "def")]
    lam_a = ga * s[("a", "att")] * s[("h", "def")]
    return float(np.clip(lam_h, LAMBDA_MIN, LAMBDA_MAX)), float(np.clip(lam_a, LAMBDA_MIN, LAMBDA_MAX))


# --------------------------------------------------------------------------
# Poisson con corrección de Dixon-Coles y mercados
# --------------------------------------------------------------------------


def dixon_coles_tau(hg, ag, lam_h, lam_a, rho):
    """Factor de Dixon-Coles para los marcadores bajos (0-0, 1-0, 0-1, 1-1)."""
    hg, ag = np.asarray(hg), np.asarray(ag)
    tau = np.ones(np.broadcast(hg, ag, lam_h, lam_a).shape)
    tau = np.where((hg == 0) & (ag == 0), 1 - lam_h * lam_a * rho, tau)
    tau = np.where((hg == 0) & (ag == 1), 1 + lam_h * rho, tau)
    tau = np.where((hg == 1) & (ag == 0), 1 + lam_a * rho, tau)
    return np.where((hg == 1) & (ag == 1), 1 - rho, tau)


def fit_rho(hg: np.ndarray, ag: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray) -> float:
    def neg_ll(rho: float) -> float:
        return -float(np.sum(np.log(np.clip(dixon_coles_tau(hg, ag, lam_h, lam_a, rho), 1e-9, None))))

    return float(minimize_scalar(neg_ll, bounds=RHO_BOUNDS, method="bounded").x)


def score_matrix(lam_h: float, lam_a: float, rho: float = 0.0, max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
    if rho:
        matrix[:2, :2] *= dixon_coles_tau(goals[:2, None], goals[None, :2], lam_h, lam_a, rho)
    return matrix / matrix.sum()


def outcome_probs(matrix: np.ndarray) -> np.ndarray:
    """[local, empate, visitante]; filas = goles del local, columnas = visitante."""
    return np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])


def goal_markets(matrix: np.ndarray) -> tuple[float, float]:
    goals = np.arange(matrix.shape[0])
    total = goals[:, None] + goals[None, :]
    return float(matrix[total >= 3].sum()), float(matrix[1:, 1:].sum())


def reweight_matrix(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reescala cada región (victoria local / empate / visitante) para que la
    matriz reproduzca el 1X2 objetivo sin cambiar la forma dentro de cada región."""
    base = outcome_probs(matrix)
    i, j = np.indices(matrix.shape)
    factor = np.select([i > j, i == j], [target[0] / base[0], target[1] / base[1]], target[2] / base[2])
    return matrix * factor


def shift_home_away(p: np.ndarray, shift: float) -> np.ndarray:
    """1X2 con el log-odds local/visitante desplazado `shift` a favor del visitante (el empate se
    reparte solo al renormalizar)."""
    lp = np.log(np.clip(np.asarray(p, float), 1e-12, 1.0))
    lp[0] -= shift
    lp[2] += shift
    q = np.exp(lp)
    return q / q.sum()


def top_scores(matrix: np.ndarray, k: int = 3) -> list[tuple[int, int, float]]:
    flat = np.argsort(matrix, axis=None)[::-1][:k]
    return [(int(i), int(j), float(matrix[i, j])) for i, j in zip(*np.unravel_index(flat, matrix.shape))]


def market_probs(fixture: pd.Series | None) -> dict | None:
    """Probabilidades implícitas de las cuotas del partido, sin el margen de la casa."""
    if fixture is None:
        return None
    odds = np.array([fixture.get("odds_h"), fixture.get("odds_d"), fixture.get("odds_a")], dtype=float)
    if not np.isfinite(odds).all() or (odds <= 1).any():
        return None
    inv = 1 / odds
    market = {"p_1x2": inv / inv.sum(), "margin": float(inv.sum() - 1), "over25": None}
    ou = np.array([fixture.get("odds_over25"), fixture.get("odds_under25")], dtype=float)
    if np.isfinite(ou).all() and (ou > 1).all():
        market["over25"] = float((1 / ou[0]) / (1 / ou).sum())
    return market


# --------------------------------------------------------------------------
# Histórico, regresión logística y ensemble
# --------------------------------------------------------------------------


def snapshot_columns(h: dict, a: dict) -> dict:
    """Variables numéricas de las instantáneas local (h_) y visitante (a_)."""
    return {
        **{f"h_{k}": v for k, v in h.items() if not k.endswith("_rows")},
        **{f"a_{k}": v for k, v in a.items() if not k.endswith("_rows")},
    }


def build_history(index: MatchIndex, since: pd.Timestamp, competition: str | None = None) -> pd.DataFrame:
    """Instantánea pre-partido (solo con datos anteriores a cada partido) de los
    partidos jugados desde `since` (solo de `competition` si se indica; el resto
    de partidos sigue contando como historial de las fuerzas)."""
    rows, cache = [], {}
    played = index.played
    if competition is not None:
        played = played[played["competition"] == competition]
    for m in played[played["datetime"] >= since].itertuples(index=False):
        snap = match_snapshot(index, m.home, m.away, m.datetime, cache)
        if snap is None:
            continue
        h, a, ratings = snap
        rows.append(
            {
                "id": m.id,
                "competition": m.competition,
                "datetime": m.datetime,
                "home": m.home,
                "away": m.away,
                "neutral": bool(m.neutral),
                "hg": m.hg,
                "ag": m.ag,
                "result": "H" if m.hg > m.ag else ("D" if m.hg == m.ag else "A"),
                "goals_home": ratings["gls"].goals_home,
                "goals_away": ratings["gls"].goals_away,
                "low_data": min(h["n_window"], a["n_window"]) < MIN_MATCHES,  # no se entrena con ellas
                **snapshot_columns(h, a),
            }
        )
    return pd.DataFrame(rows)


def history_lambdas(hist: pd.DataFrame, shrink: float, signal_weight: float,
                    market_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    def col(name: str) -> np.ndarray:  # numpy: se llama cientos de veces al elegir los pesos
        return hist[name].to_numpy(float)

    def mix(side: str, part: str) -> np.ndarray:
        n_recent = col(f"{side}_n_recent")
        w = n_recent / (n_recent + shrink)

        def log_strength(kind: str) -> np.ndarray:
            return np.log(np.maximum(w * col(f"{side}_{part}_recent_{kind}") + (1 - w) * col(f"{side}_{part}_long_{kind}"),
                                     STRENGTH_FLOOR))

        log_s = signal_weight * log_strength("sig") + (1 - signal_weight) * log_strength("gls")
        if market_weight:
            mw = market_weight * col(f"{side}_mkt_share")
            log_s = mw * log_strength("mkt") + (1 - mw) * log_s
        return np.exp(log_s)

    gh, ga = base_goals(hist["goals_home"].to_numpy(), hist["goals_away"].to_numpy(), hist["neutral"].to_numpy())
    lam_h = np.clip(gh * mix("h", "att") * mix("a", "def"), LAMBDA_MIN, LAMBDA_MAX)
    lam_a = np.clip(ga * mix("a", "att") * mix("h", "def"), LAMBDA_MIN, LAMBDA_MAX)
    return lam_h, lam_a


def history_features(hist: pd.DataFrame, lam_h: np.ndarray, lam_a: np.ndarray, p_poisson: np.ndarray) -> pd.DataFrame:
    p = np.clip(p_poisson, 1e-6, 1)
    return pd.DataFrame(
        {
            "log_lambda_ratio": np.log(lam_h / lam_a),
            "log_lambda_total": np.log(lam_h + lam_a),
            "poisson_home_vs_draw": np.log(p[:, 0] / p[:, 1]),  # la forma no lineal del empate según Poisson
            "poisson_away_vs_draw": np.log(p[:, 2] / p[:, 1]),
            "ppg5_home": hist["h_ppg5"],
            "ppg5_away": hist["a_ppg5"],
            "ppg5_home_at_home": hist["h_ppg5_venue"],
            "ppg5_away_away": hist["a_ppg5_venue"],
            "rest_home": hist["h_rest_days"].map(rest_category),
            "rest_away": hist["a_rest_days"].map(rest_category),
        }
    )


def logistic_pipeline() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 9), cv=TimeSeriesSplit(n_splits=5), scoring="neg_log_loss", max_iter=5000
    )
    return Pipeline([("prep", prep), ("lr", model)])


def fit_logistic(features: pd.DataFrame, y: np.ndarray) -> Pipeline:
    with warnings.catch_warnings():  # avisos de deprecación de scikit-learn, sin efecto en el ajuste
        warnings.simplefilter("ignore", FutureWarning)
        return logistic_pipeline().fit(features, y)


def multiclass_log_loss(y: np.ndarray, proba_hda: np.ndarray) -> float:
    """Log loss con columnas en orden CLASSES (H, D, A)."""
    cols = np.array([CLASSES.index(v) for v in y])
    p = np.clip(proba_hda[np.arange(len(y)), cols], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def predict_hda(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(features)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return proba[:, order]


def competition_kappa(competition: np.ndarray, hg: np.ndarray, ag: np.ndarray, lam_h: np.ndarray,
                      lam_a: np.ndarray, exclude: frozenset[str] = frozenset()) -> dict[tuple[str, str], float]:
    """Calibración del nivel de goles por competición y sede: goles reales / goles esperados,
    con un suavizado de KAPPA_PRIOR_MATCHES partidos. Corrige que la base de λ sea la media
    de todas las competiciones mezcladas (divisiones inferiores, copas con sus ligas)."""
    comps, idx = np.unique(np.asarray(competition, dtype=str), return_inverse=True)
    n = np.bincount(idx, minlength=len(comps))
    sums = {k: np.bincount(idx, np.asarray(v, float), len(comps)) for k, v in
            (("hg", hg), ("ag", ag), ("lh", lam_h), ("la", lam_a))}
    kappa = {}
    for i, comp in enumerate(comps):
        if comp in exclude:
            continue
        for side, goals, lam in (("h", sums["hg"][i], sums["lh"][i]), ("a", sums["ag"][i], sums["la"][i])):
            prior = KAPPA_PRIOR_MATCHES * lam / n[i]
            kappa[(str(comp), side)] = float((goals + prior) / (lam + prior))
    return kappa


def apply_kappa(competition: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray,
                kappa: dict[tuple[str, str], float]) -> tuple[np.ndarray, np.ndarray]:
    kh = np.array([kappa.get((c, "h"), 1.0) for c in competition])
    ka = np.array([kappa.get((c, "a"), 1.0) for c in competition])
    return np.clip(lam_h * kh, LAMBDA_MIN, LAMBDA_MAX), np.clip(lam_a * ka, LAMBDA_MIN, LAMBDA_MAX)


def newcomer_factors(hg: np.ndarray, ag: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray, new_h: np.ndarray,
                     new_a: np.ndarray) -> tuple[float, float]:
    """Calibración de los recién llegados: (factor de sus goles esperados, factor de los del rival),
    goles reales / esperados con un suavizado de KAPPA_PRIOR_MATCHES partidos."""
    scored = np.concatenate([hg[new_h], ag[new_a]])
    own = np.concatenate([lam_h[new_h], lam_a[new_a]])
    conceded = np.concatenate([ag[new_h], hg[new_a]])
    rival = np.concatenate([lam_a[new_h], lam_h[new_a]])
    if len(own) == 0:
        return 1.0, 1.0
    prior_own, prior_rival = KAPPA_PRIOR_MATCHES * own.mean(), KAPPA_PRIOR_MATCHES * rival.mean()
    return (float((scored.sum() + prior_own) / (own.sum() + prior_own)),
            float((conceded.sum() + prior_rival) / (rival.sum() + prior_rival)))


def shrink_totals(competition, lam_h, lam_a, shrink: float, total_mean: dict[str, float]):
    """Acerca el total de goles esperado a la media de su competición (en log, con peso
    1 − shrink) manteniendo la proporción local/visitante: el producto ataque × defensa
    exagera las diferencias de total entre partidos."""
    if shrink >= 1:
        return lam_h, lam_a
    total = lam_h + lam_a
    mean = np.array([total_mean.get(c, np.nan) for c in np.atleast_1d(competition)], dtype=float)
    mean = np.where(np.isfinite(mean), mean, total)
    factor = mean ** (1 - shrink) * total ** shrink / total
    return np.clip(lam_h * factor, LAMBDA_MIN, LAMBDA_MAX), np.clip(lam_a * factor, LAMBDA_MIN, LAMBDA_MAX)


def apply_newcomer(lam_h, lam_a, new_h, new_a, own: float, rival: float):
    lam_h = lam_h * np.where(new_h, own, 1.0) * np.where(new_a, rival, 1.0)
    lam_a = lam_a * np.where(new_a, own, 1.0) * np.where(new_h, rival, 1.0)
    return np.clip(lam_h, LAMBDA_MIN, LAMBDA_MAX), np.clip(lam_a, LAMBDA_MIN, LAMBDA_MAX)


@dataclass
class TrainedModels:
    shrink: float  # peso de la fuerza de 12 meses frente a la forma reciente
    signal_weight: float  # peso de la señal (xG / xG aproximado) frente a los goles
    rho: float  # corrección de Dixon-Coles
    kappa: dict  # calibración de goles por (competición, sede)
    newcomer: tuple[float, float]  # recién llegados: (factor de sus goles, factor de los del rival)
    total_shrink: float  # peso del total de goles propio frente a la media de la competición
    total_mean: dict  # total de goles esperado medio por competición
    logistic: Pipeline
    w_poisson: float
    n_train: int
    n_val: int  # partidos de validación evaluados (los de la competición si hay suficientes)
    val_period: str
    ll_poisson: float
    ll_logistic: float
    ll_ensemble: float
    ll_baseline: float  # predecir siempre las frecuencias 1/X/2 del entrenamiento (referencia)
    val_predictions: pd.DataFrame = field(repr=False)  # predicciones fuera de muestra (para el backtest)
    evaluated_on: str = "all"  # "focus" (solo la competición) o "all" (todo el pool)
    n_val_focus: int = 0  # partidos de validación de la propia competición (copas con pool)
    ll_ensemble_focus: float | None = None
    market_weight: float = 0.0  # peso de la fuerza según el mercado (cuotas de cierre de partidos anteriores)


def train_models(hist: pd.DataFrame, has_signal: bool = True, focus: str | None = None,
                 kappa_exclude: frozenset[str] = frozenset(),
                 market_weights: tuple[float, ...] = (0.0,)) -> TrainedModels:
    """Entrena Poisson + logística sobre el histórico de instantáneas. `kappa_exclude`: competiciones
    sin calibración propia de goles (las copas: pocos partidos, y en validación empeoraba).
    `market_weights`: pesos posibles de la fuerza según el mercado (columnas *_mkt del histórico)."""
    hist = hist.sort_values("datetime").reset_index(drop=True)
    n_val = max(MIN_VAL_ROWS, int(len(hist) * VAL_FRACTION))
    if len(hist) - n_val < MIN_TRAIN_ROWS:
        raise PredictionError(f"Histórico insuficiente para entrenar y validar el modelo ({len(hist)} partidos).")
    in_train = np.arange(len(hist)) < len(hist) - n_val
    val_mask = ~in_train
    # Se ajusta solo con partidos con datos suficientes; se valida con todos (como se usa el modelo).
    low = hist["low_data"].to_numpy(bool) if "low_data" in hist else np.zeros(len(hist), bool)
    train_mask = in_train & ~low
    train = hist[train_mask]
    comp = hist["competition"].to_numpy()
    hg, ag = hist["hg"].to_numpy(float), hist["ag"].to_numpy(float)

    # 1) Forma reciente vs. 12 meses, señal vs. goles (y mercado) y calibración por competición:
    #    máxima verosimilitud de los goles reales en el periodo de entrenamiento.
    #    Con un peso de mercado fijo (cuotas solo recientes, ausentes en casi todo el entrenamiento),
    #    la forma y la señal se eligen sin mercado y luego se le aplica su peso.
    weights = SIGNAL_WEIGHT_GRID if has_signal else (1.0,)
    fixed_market = len(market_weights) == 1 and market_weights[0] > 0
    best, best_ll = None, -np.inf
    for shrink in SHRINK_GRID:
        for sw in weights:
            for mw in ((0.0,) if fixed_market else market_weights):
                lh, la = history_lambdas(train, shrink, sw, mw)
                kappa = competition_kappa(comp[train_mask], hg[train_mask], ag[train_mask], lh, la, kappa_exclude)
                lh, la = apply_kappa(comp[train_mask], lh, la, kappa)
                ll = poisson.logpmf(hg[train_mask], lh).sum() + poisson.logpmf(ag[train_mask], la).sum()
                if ll > best_ll:
                    best, best_ll = (shrink, sw, mw, kappa), ll
    shrink, signal_weight, market_weight, kappa = best
    if fixed_market:
        market_weight = market_weights[0]
        lh, la = history_lambdas(train, shrink, signal_weight, market_weight)
        kappa = competition_kappa(comp[train_mask], hg[train_mask], ag[train_mask], lh, la, kappa_exclude)

    lam_h, lam_a = apply_kappa(comp, *history_lambdas(hist, shrink, signal_weight, market_weight), kappa)
    # 1b) Recién llegados (ascendidos): calibración de su nivel de goles a favor y en contra.
    new_h = (hist["h_n_focus"] < NEWCOMER_MATCHES).to_numpy() if "h_n_focus" in hist else np.zeros(len(hist), bool)
    new_a = (hist["a_n_focus"] < NEWCOMER_MATCHES).to_numpy() if "a_n_focus" in hist else np.zeros(len(hist), bool)
    newcomer = newcomer_factors(hg[train_mask], ag[train_mask], lam_h[train_mask], lam_a[train_mask],
                                new_h[train_mask], new_a[train_mask])
    lam_h, lam_a = apply_newcomer(lam_h, lam_a, new_h, new_a, *newcomer)
    # 1c) Total de goles: cuánto acercarlo a la media de la competición (verosimilitud en entrenamiento).
    totals = pd.Series(lam_h[train_mask] + lam_a[train_mask]).groupby(comp[train_mask]).mean()
    total_mean = {str(k): float(v) for k, v in totals.items()}
    total_shrink, best_tll = 1.0, -np.inf
    for ts in TOTAL_SHRINK_GRID:
        th, ta = shrink_totals(comp[train_mask], lam_h[train_mask], lam_a[train_mask], ts, total_mean)
        tll = poisson.logpmf(hg[train_mask], th).sum() + poisson.logpmf(ag[train_mask], ta).sum()
        if tll > best_tll:
            total_shrink, best_tll = ts, tll
    lam_h, lam_a = shrink_totals(comp, lam_h, lam_a, total_shrink, total_mean)
    # 2) Dixon-Coles: ρ por máxima verosimilitud en entrenamiento.
    rho = fit_rho(hg[train_mask], ag[train_mask], lam_h[train_mask], lam_a[train_mask])
    p_poisson = np.array([outcome_probs(score_matrix(lh, la, rho)) for lh, la in zip(lam_h, lam_a)])
    features = history_features(hist, lam_h, lam_a, p_poisson)
    y = hist["result"].to_numpy()

    # 3) Regresión logística entrenada solo con el periodo anterior a la validación.
    logistic = fit_logistic(features[train_mask], y[train_mask])
    p_logistic_val = predict_hda(logistic, features[val_mask])

    # 4) Evaluación: la propia competición si tiene suficientes partidos de validación; si no, todo el pool.
    in_focus = (comp == focus) if focus is not None else np.ones(len(hist), bool)
    use_focus = focus is not None and (in_focus & val_mask).sum() >= FOCUS_MIN_VAL
    eval_val = in_focus[val_mask] if use_focus else np.ones(val_mask.sum(), bool)
    eval_train = in_focus[train_mask] if use_focus else np.ones(train_mask.sum(), bool)
    y_val, p_pois_val = y[val_mask][eval_val], p_poisson[val_mask][eval_val]
    p_log_val = p_logistic_val[eval_val]

    # 5) Peso del ensemble que minimiza el log loss de validación.
    losses = [multiclass_log_loss(y_val, w * p_pois_val + (1 - w) * p_log_val) for w in ENSEMBLE_WEIGHT_GRID]
    w_poisson = float(ENSEMBLE_WEIGHT_GRID[int(np.argmin(losses))])

    # Predicciones fuera de muestra de todo el periodo de validación (para el backtest contra el mercado).
    val = hist[val_mask].reset_index(drop=True)
    p_ens_all = w_poisson * p_poisson[val_mask] + (1 - w_poisson) * p_logistic_val
    over = [goal_markets(reweight_matrix(score_matrix(lh, la, rho), pe))[0]
            for lh, la, pe in zip(lam_h[val_mask], lam_a[val_mask], p_ens_all)]
    val_predictions = pd.DataFrame({
        "id": val["id"] if "id" in val else None, "datetime": val["datetime"], "competition": val["competition"],
        "home": val["home"], "away": val["away"],
        "hg": val["hg"], "ag": val["ag"], "result": val["result"],
        "lam_h": lam_h[val_mask], "lam_a": lam_a[val_mask], "over25": over,
        "newcomer": (new_h | new_a)[val_mask], "low_data": low[val_mask],
        **{f"p_{k}_{c}": p[:, i] for k, p in (("poisson", p_poisson[val_mask]), ("logistic", p_logistic_val),
                                              ("ensemble", p_ens_all)) for i, c in enumerate(CLASSES)},
    })

    focus_metrics = {}
    if focus is not None and not use_focus:
        n_focus = int((in_focus & val_mask).sum())
        if n_focus >= 20:
            sub = in_focus[val_mask]
            focus_metrics = {"n_val_focus": n_focus,
                             "ll_ensemble_focus": multiclass_log_loss(y[val_mask][sub], p_ens_all[sub])}

    val_dates = hist.loc[val_mask, "datetime"][eval_val]
    train_freq = [np.mean(y[train_mask][eval_train] == c) for c in CLASSES]
    return TrainedModels(
        shrink=shrink,
        signal_weight=signal_weight,
        rho=rho,
        kappa=kappa,
        newcomer=newcomer,
        total_shrink=total_shrink,
        total_mean=total_mean,
        logistic=fit_logistic(features[~low], y[~low]),  # modelo final: todo el histórico con datos suficientes
        w_poisson=w_poisson,
        n_train=int(train_mask.sum()),
        n_val=int(eval_val.sum()),
        val_period=f"{val_dates.min():%d/%m/%Y} – {val_dates.max():%d/%m/%Y}",
        ll_poisson=multiclass_log_loss(y_val, p_pois_val),
        ll_logistic=multiclass_log_loss(y_val, p_log_val),
        ll_ensemble=float(min(losses)),
        ll_baseline=multiclass_log_loss(y_val, np.tile(train_freq, (len(y_val), 1))),
        val_predictions=val_predictions,
        evaluated_on="focus" if use_focus else "all",
        **focus_metrics,
        market_weight=market_weight,
    )


# --------------------------------------------------------------------------
# API de alto nivel
# --------------------------------------------------------------------------


@dataclass
class LeagueData:
    competition: str  # código de la competición que se predice
    name: str
    matches: pd.DataFrame  # todos los partidos (jugados y por jugar; con pool en copas)
    index: MatchIndex  # partidos jugados usables por el modelo
    sources: list[SourceInfo]
    signal_name: str  # "xG", "xG aproximado (tiros)" o "goles"
    training_since: pd.Timestamp  # primer partido usado como fila de entrenamiento
    train_on_focus: bool = True  # filas de entrenamiento: solo la competición (ligas) o todo el pool (copas)
    kappa_exclude: frozenset = frozenset()  # competiciones sin calibración propia de goles (copas)
    market_mode: str | None = None  # señal de mercado: "learn" (peso aprendido), "fixed" (peso típico) o None
    market_coverage: float = 0.0  # partidos con cuotas en los últimos 12 meses (de la competición o su pool)

    @property
    def market_weights(self) -> tuple[float, ...]:
        """Pesos de la señal de mercado entre los que elige el entrenamiento."""
        return {"learn": MARKET_WEIGHT_GRID, "fixed": (MARKET_WEIGHT_DEFAULT,)}.get(self.market_mode, (0.0,))

    @property
    def focus_matches(self) -> pd.DataFrame:
        return self.matches[self.matches["competition"] == self.competition]

    @property
    def teams(self) -> list[str]:
        """Equipos de la competición en los últimos 400 días o con partidos por jugar."""
        m = self.focus_matches
        recent = m[(m["datetime"] >= utc_now() - pd.Timedelta(days=400)) | ~m["played"]]
        return sorted(set(recent["home"]) | set(recent["away"]))


def build_league_data(competition: str, name: str, matches: pd.DataFrame, sources: list[SourceInfo],
                      signal_name: str, training_since: pd.Timestamp, train_on_focus: bool = True,
                      kappa_exclude: frozenset = frozenset()) -> LeagueData:
    """Prepara los partidos usables: jugados en 90 minutos, con goles y señal.
    Donde falta la señal (p. ej. un partido sin estadísticas de tiros) se usan los goles.
    Si la competición (en copas, su pool) tiene cuotas de cierre (h_mkt / a_mkt) suficientes (ver
    MARKET_LEARN_COVERAGE), se usa la señal de mercado; donde falta, se rellena con la señal
    reescalada al nivel del mercado."""
    matches = matches.sort_values(["datetime", "id"]).reset_index(drop=True)
    usable = matches[matches["played"] & ~matches["extra_time"] & matches["hg"].notna()].copy()
    has_signal = signal_name != "goles"
    usable["h_sig"] = usable["h_sig"].fillna(usable["hg"]) if has_signal else usable["hg"]
    usable["a_sig"] = usable["a_sig"].fillna(usable["ag"]) if has_signal else usable["ag"]
    if usable.empty:
        raise PredictionError(f"Sin partidos jugados para {name}.")
    market_mode, coverage = None, 0.0
    if "h_mkt" in usable:
        with_market = usable["h_mkt"].notna() & usable["a_mkt"].notna()
        scope = (usable["competition"] == competition) if train_on_focus else pd.Series(True, index=usable.index)
        recent = usable["datetime"] >= usable["datetime"].max() - pd.Timedelta(days=WINDOW_DAYS)
        coverage = float(with_market[scope & recent].mean()) if (scope & recent).any() else 0.0
        if with_market[scope & (usable["datetime"] >= training_since)].mean() >= MARKET_LEARN_COVERAGE:
            market_mode = "learn"
        elif coverage >= MARKET_RECENT_COVERAGE:
            market_mode = "fixed"
        if market_mode:
            both = usable[with_market]
            scale = (both["h_mkt"] + both["a_mkt"]).sum() / (both["h_sig"] + both["a_sig"]).sum()
            # Partido "con mercado" si tiene cuotas o si su competición no tiene ninguna (p. ej. una división
            # inferior sin cuotas): mkt_share solo baja el peso del mercado por huecos de cuotas reales.
            no_odds_comp = ~with_market.groupby(usable["competition"]).transform("any")
            usable["mkt_real"] = with_market | no_odds_comp
            usable["h_mkt"] = usable["h_mkt"].where(with_market, usable["h_sig"] * scale)
            usable["a_mkt"] = usable["a_mkt"].where(with_market, usable["a_sig"] * scale)
    index = MatchIndex(usable, has_signal, focus=competition if train_on_focus else None,
                       has_market=market_mode is not None)
    return LeagueData(competition, name, matches, index, sources, signal_name, training_since, train_on_focus,
                      kappa_exclude, market_mode, coverage)


@dataclass
class LeagueModel:
    data: LeagueData
    trained: TrainedModels


def train_league(data: LeagueData) -> LeagueModel:
    hist = build_history(data.index, data.training_since, data.competition if data.train_on_focus else None)
    if hist.empty:
        raise PredictionError(f"Sin histórico suficiente para entrenar {data.name}.")
    pooled = (hist["competition"] != data.competition).any()
    return LeagueModel(data, train_models(hist, data.index.has_signal, data.competition if pooled else None,
                                          data.kappa_exclude, data.market_weights))


@dataclass
class MatchPrediction:
    competition: str
    home: str
    away: str
    cutoff: pd.Timestamp  # instante (UTC) hasta el que se usan datos
    match_id: str | None  # id del partido en su fuente si está en el calendario
    kickoff: pd.Timestamp | None  # hora de inicio en UTC
    neutral: bool
    final_score: tuple[int, int] | None  # resultado real si ya se jugó
    lam_home: float
    lam_away: float
    p_poisson: np.ndarray  # [local, empate, visitante] (Poisson con Dixon-Coles)
    p_logistic: np.ndarray
    p_final: np.ndarray
    over25: float
    btts: float
    over25_poisson: float
    btts_poisson: float
    top_scores: list[tuple[int, int, float]]  # de la matriz de Poisson
    market: dict | None  # probabilidades del mercado (sin margen) si hay cuotas
    home_snap: dict = field(repr=False)
    away_snap: dict = field(repr=False)
    ratings: dict = field(repr=False)
    trained: TrainedModels = field(repr=False)
    url: str | None = None
    signal_name: str = "xG"  # señal de calidad de ocasiones de la competición
    base_home: float = float("nan")  # goles medios esperados del local de la competición (con su calibración)
    base_away: float = float("nan")
    lineups: object | None = field(default=None, repr=False)  # src.lineups.LineupInfo si se aplicaron
    p_before_lineups: np.ndarray | None = None  # 1X2 final antes del ajuste por alineaciones

    @property
    def signal_short(self) -> str:
        return {"xG": "xG", "goles": "Goles"}.get(self.signal_name, "xG aprox.")

    def attack_defense(self, side: str) -> tuple[float, float]:
        snap = self.home_snap if side == "home" else self.away_snap
        t = self.trained
        return (strength(snap, t.shrink, t.signal_weight, "att", t.market_weight),
                strength(snap, t.shrink, t.signal_weight, "def", t.market_weight))

    @property
    def newcomers(self) -> list[str]:
        """Equipos con menos de NEWCOMER_MATCHES partidos en la competición en 12 meses (p. ej. ascendidos)."""
        return [name for name, snap in ((self.home, self.home_snap), (self.away, self.away_snap))
                if snap["n_focus"] < NEWCOMER_MATCHES]

    @property
    def low_data(self) -> bool:
        """Algún equipo con menos de MIN_MATCHES partidos en los últimos 12 meses."""
        return min(self.home_snap["n_window"], self.away_snap["n_window"]) < MIN_MATCHES


def find_fixture(data: LeagueData, home: str, away: str, now: pd.Timestamp | None = None,
                 horizon_days: int = FIXTURE_HORIZON_DAYS) -> pd.Series | None:
    """Próximo partido home-away de la competición aún sin jugar (o empezado hace
    menos de 3 h) en los próximos `horizon_days` días."""
    now = now if now is not None else utc_now()
    m = data.focus_matches
    upcoming = m[(~m["played"]) & (m["home"] == home) & (m["away"] == away)
                 & (m["datetime"] >= now - pd.Timedelta(hours=3))
                 & (m["datetime"] <= now + pd.Timedelta(days=horizon_days))]
    return upcoming.iloc[0] if len(upcoming) else None


def predict_match(model: LeagueModel, home: str, away: str, cutoff: pd.Timestamp | None = None,
                  fixture: pd.Series | None = None, url: str | None = None, lineups=None) -> MatchPrediction:
    """Predice home-away con los datos anteriores a `cutoff`. Si se pasa el
    `fixture` (fila de partidos estándar), el corte es su hora de inicio y se
    usan su campo neutral y sus cuotas. `lineups` (src.lineups.LineupInfo): ajusta el 1X2
    final por las rotaciones de los titulares."""
    data, t = model.data, model.trained
    teams = sorted(set(data.teams) | set(data.index.team_rows))
    home = resolve_team(home, teams)
    away = resolve_team(away, teams)
    if home == away:
        raise PredictionError("Local y visitante son el mismo equipo.")
    if fixture is None and cutoff is None:
        fixture = find_fixture(data, home, away)
    if fixture is not None:
        cutoff = fixture["datetime"]
    if cutoff is None:
        cutoff = utc_now()
    neutral = bool(fixture["neutral"]) if fixture is not None and pd.notna(fixture.get("neutral")) else False

    snap = match_snapshot(data.index, home, away, cutoff, {}, with_rows=True)
    if snap is None:
        raise PredictionError(f"Sin datos de los últimos {WINDOW_DAYS} días para {home} o {away}.")
    h, a, ratings = snap
    comp = str(fixture["competition"]) if fixture is not None and "competition" in fixture else data.competition
    kh, ka = t.kappa.get((comp, "h"), 1.0), t.kappa.get((comp, "a"), 1.0)
    lam_h, lam_a = expected_goals(h, a, ratings, t.shrink, t.signal_weight, neutral, t.market_weight)
    lam_h, lam_a = float(np.clip(lam_h * kh, LAMBDA_MIN, LAMBDA_MAX)), float(np.clip(lam_a * ka, LAMBDA_MIN, LAMBDA_MAX))
    new_h, new_a = h["n_focus"] < NEWCOMER_MATCHES, a["n_focus"] < NEWCOMER_MATCHES
    lam_h, lam_a = (float(x) for x in apply_newcomer(lam_h, lam_a, new_h, new_a, *t.newcomer))
    lam_h, lam_a = (float(np.atleast_1d(x)[0]) for x in shrink_totals(comp, lam_h, lam_a, t.total_shrink, t.total_mean))
    base_h, base_a = base_goals(ratings["gls"].goals_home, ratings["gls"].goals_away, neutral)
    matrix = score_matrix(lam_h, lam_a, t.rho)
    p_poisson = outcome_probs(matrix)
    features = history_features(pd.DataFrame([snapshot_columns(h, a)]), np.array([lam_h]), np.array([lam_a]),
                                p_poisson[None, :])
    p_logistic = predict_hda(t.logistic, features)[0]
    p_final = t.w_poisson * p_poisson + (1 - t.w_poisson) * p_logistic
    p_before_lineups = None
    if lineups is not None:
        p_before_lineups, p_final = p_final, shift_home_away(p_final, lineups.shift)
    over25, btts = goal_markets(reweight_matrix(matrix, p_final))
    over25_poisson, btts_poisson = goal_markets(matrix)

    final_score = None
    if fixture is not None and bool(fixture["played"]) and pd.notna(fixture["hg"]):
        final_score = (int(fixture["hg"]), int(fixture["ag"]))
    return MatchPrediction(
        competition=data.competition,
        home=home,
        away=away,
        cutoff=cutoff,
        match_id=str(fixture["id"]) if fixture is not None else None,
        kickoff=fixture["datetime"] if fixture is not None else None,
        neutral=neutral,
        final_score=final_score,
        lam_home=lam_h,
        lam_away=lam_a,
        p_poisson=p_poisson,
        p_logistic=p_logistic,
        p_final=p_final,
        over25=over25,
        btts=btts,
        over25_poisson=over25_poisson,
        btts_poisson=btts_poisson,
        top_scores=top_scores(matrix),
        market=market_probs(fixture),
        home_snap=h,
        away_snap=a,
        ratings=ratings,
        trained=t,
        url=url,
        signal_name=data.signal_name,
        base_home=float(base_h) * kh,
        base_away=float(base_a) * ka,
        lineups=lineups,
        p_before_lineups=p_before_lineups,
    )


def fixtures_between(data: LeagueData, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """Partidos de la competición (jugados o no) con inicio en [start_utc, end_utc)."""
    m = data.focus_matches
    return m[(m["datetime"] >= start_utc) & (m["datetime"] < end_utc)]


def form_string(rows: pd.DataFrame) -> str:
    """Racha en letras (G/E/P), del partido más antiguo al más reciente."""
    return " ".join({3: "G", 1: "E", 0: "P"}[int(p)] for p in rows["pts"])
