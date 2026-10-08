"""Modelo de predicción con datos reales de Understat (xG): Poisson + regresión logística.

Núcleo compartido por el script de terminal `modelo_prediccion.py` y la página
"Partidos del día" de la app. Descarga de understat.com el xG partido a
partido de la liga (temporada en curso y anteriores), estima la fuerza de
ataque/defensa de cada equipo ajustada por la calidad de sus rivales y
combina dos modelos:

- Poisson puro: goles esperados (λ) local/visitante a partir de la fuerza
  derivada del xG de los últimos 10 partidos, mezclada con la fuerza de los
  últimos 12 meses (el peso de la mezcla se elige con datos históricos).
- Regresión logística multinomial: reajusta la señal de Poisson con la racha
  de puntos (últimos 5), el rendimiento en casa/fuera (localía) y los días
  de descanso (categóricos), entrenada con temporadas anteriores.

El peso de cada modelo en el ensemble se elige minimizando el log loss sobre
partidos de validación que la regresión logística no vio al entrenar.

Flujo de uso:
    data = load_league("EPL")          # descarga (con caché en disco)
    model = train_league(data)         # ~segundos; reutilizable para toda la jornada
    pred = predict_match(model, "Arsenal", "Leeds")
"""

from __future__ import annotations

import difflib
import json
import time
import unicodedata
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

UNDERSTAT = "https://understat.com"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) football-predictor-macho",
    "X-Requested-With": "XMLHttpRequest",
}
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "understat_cache"
CURRENT_SEASON_CACHE_TTL_S = 3 * 3600  # temporadas pasadas no cambian: caché sin caducidad

# Código de liga en Understat -> (nombre visible, alias aceptados).
LEAGUES = {
    "EPL": ("Premier League", ["premier league", "premier", "epl", "inglaterra", "england"]),
    "La_liga": ("LaLiga", ["laliga", "la liga", "liga espanola", "espana", "spain"]),
    "Bundesliga": ("Bundesliga", ["bundesliga", "alemania", "germany"]),
    "Serie_A": ("Serie A", ["serie a", "italia", "italy"]),
    "Ligue_1": ("Ligue 1", ["ligue 1", "francia", "france"]),
    "RFPL": ("Liga rusa", ["rfpl", "liga premier de rusia", "rusia", "russia"]),
}

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

N_RECENT = 10  # partidos recientes que definen la forma en xG
N_FORM = 5  # partidos para la racha de puntos
WINDOW_DAYS = 365  # ventana de la fuerza "de fondo" (ajustada por rival)
MIN_WINDOW_MATCHES = 50  # partidos de liga necesarios en la ventana para ajustar las fuerzas
MIN_MATCHES = 5  # mínimo de partidos de cada equipo para usar un partido al entrenar
IPF_ITERATIONS = 25
MAX_GOALS = 10
LAMBDA_MIN, LAMBDA_MAX = 0.1, 5.0
# Peso (en "partidos equivalentes") de la fuerza de 12 meses frente a los últimos 10.
SHRINK_GRID = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 1e9)
ENSEMBLE_WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
CLASSES = ["H", "D", "A"]
DEFAULT_SEASONS_BACK = 4
# Un partido del calendario más lejano que esto (p. ej. la vuelta, meses después) no fija el corte:
# se predice con los datos de hoy en lugar de proyectar el descanso hasta esa fecha.
FIXTURE_HORIZON_DAYS = 14

NUMERIC_FEATURES = [
    "log_lambda_ratio",
    "log_lambda_total",
    "ppg5_home",
    "ppg5_away",
    "ppg5_home_at_home",
    "ppg5_away_away",
]
CATEGORICAL_FEATURES = ["rest_home", "rest_away"]


class PredictionError(RuntimeError):
    """Error mostrable al usuario (liga/equipo desconocido, Understat caído, datos insuficientes)."""


def season_label(season: int) -> str:
    return f"{season}/{(season + 1) % 100:02d}"


def current_season_for(now: pd.Timestamp) -> int:
    """Understat nombra cada temporada por su año de inicio (2026 = 2026/27)."""
    return now.year if now.month >= 7 else now.year - 1


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(text.lower().replace("-", " ").replace(".", " ").split())


def league_name(code: str) -> str:
    return LEAGUES[code][0]


def resolve_league(name: str) -> str:
    key = normalize(name)
    for code, (display, aliases) in LEAGUES.items():
        if key == normalize(code) or key == normalize(display) or key in aliases:
            return code
    valid = ", ".join(display for display, _ in LEAGUES.values())
    raise PredictionError(f"Liga no reconocida: '{name}'. Understat cubre: {valid}.")


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
    raise PredictionError(f"Equipo no encontrado: '{name}'. Equipos de la temporada: {', '.join(sorted(teams))}.")


# --------------------------------------------------------------------------
# Descarga (Understat)
# --------------------------------------------------------------------------


@dataclass
class SourceInfo:
    url: str
    season: int
    played: int
    from_cache: bool


def fetch_season(
    session: requests.Session, league: str, season: int, current_season: int, refresh: bool = False
) -> tuple[list[dict], SourceInfo]:
    url = f"{UNDERSTAT}/getLeagueData/{league}/{season}"
    cache_file = CACHE_DIR / f"{league}_{season}.json"
    if not refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if season < current_season or age < CURRENT_SEASON_CACHE_TTL_S:
            dates = json.loads(cache_file.read_text(encoding="utf-8"))
            return dates, SourceInfo(url, season, sum(bool(m.get("isResult")) for m in dates), True)

    headers = {**HTTP_HEADERS, "Referer": f"{UNDERSTAT}/league/{league}/{season}"}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = session.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            dates = resp.json()["dates"]
            break
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    else:
        raise PredictionError(f"No se pudo descargar {url}: {last_error}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(dates), encoding="utf-8")
    return dates, SourceInfo(url, season, sum(bool(m.get("isResult")) for m in dates), False)


def matches_frame(dates_by_season: dict[int, list[dict]]) -> pd.DataFrame:
    """Convierte el JSON `dates` de Understat en una fila por partido (horas en UTC)."""
    rows = []
    for season, dates in dates_by_season.items():
        for m in dates:
            played = bool(m.get("isResult"))
            rows.append(
                {
                    "id": int(m["id"]),
                    "season": season,
                    "datetime": pd.Timestamp(m["datetime"]),
                    "home": m["h"]["title"],
                    "away": m["a"]["title"],
                    "played": played,
                    "hg": float(m["goals"]["h"]) if played else np.nan,
                    "ag": float(m["goals"]["a"]) if played else np.nan,
                    "hxg": float(m["xG"]["h"]) if played and m["xG"]["h"] is not None else np.nan,
                    "axg": float(m["xG"]["a"]) if played and m["xG"]["a"] is not None else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["datetime", "id"]).reset_index(drop=True)


def team_long_frame(played: pd.DataFrame) -> pd.DataFrame:
    """Una fila por equipo y partido jugado (vista desde ese equipo)."""
    common = {"id": played["id"], "season": played["season"], "datetime": played["datetime"]}
    home = pd.DataFrame(
        {**common, "team": played["home"], "opp": played["away"], "venue": "h",
         "gf": played["hg"], "ga": played["ag"], "xgf": played["hxg"], "xga": played["axg"]}
    )
    away = pd.DataFrame(
        {**common, "team": played["away"], "opp": played["home"], "venue": "a",
         "gf": played["ag"], "ga": played["hg"], "xgf": played["axg"], "xga": played["hxg"]}
    )
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.select([long["gf"] > long["ga"], long["gf"] == long["ga"]], [3, 1], 0)
    return long.sort_values(["datetime", "id"]).reset_index(drop=True)


class MatchIndex:
    """Partidos jugados indexados por fecha y por equipo, para cortar el
    historial anterior a cualquier instante sin filtrar DataFrames enteros."""

    def __init__(self, played: pd.DataFrame):
        self.played = played.reset_index(drop=True)
        self.times = self.played["datetime"].to_numpy()
        long = team_long_frame(self.played)
        self.team_rows = {team: rows.reset_index(drop=True) for team, rows in long.groupby("team", sort=False)}
        self.team_arrays = {
            team: {
                "times": rows["datetime"].to_numpy(),
                "opp": rows["opp"].to_numpy(),
                "home": (rows["venue"] == "h").to_numpy(),
                "xgf": rows["xgf"].to_numpy(float),
                "xga": rows["xga"].to_numpy(float),
                "pts": rows["pts"].to_numpy(float),
            }
            for team, rows in self.team_rows.items()
        }

    def window(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        start = np.searchsorted(self.times, (cutoff - pd.Timedelta(days=WINDOW_DAYS)).to_datetime64(), "left")
        end = np.searchsorted(self.times, cutoff.to_datetime64(), "left")
        return self.played.iloc[start:end]


# --------------------------------------------------------------------------
# Fuerza de ataque/defensa y modelo de Poisson
# --------------------------------------------------------------------------


@dataclass
class LeagueRatings:
    index: dict[str, int]
    mu_home: float  # xG medio del local en la ventana
    mu_away: float  # xG medio del visitante en la ventana
    goals_home: float  # goles reales medios del local (nivel de los λ)
    goals_away: float  # goles reales medios del visitante
    attack: np.ndarray  # 1.0 = media de la liga; >1 genera más xG
    defense: np.ndarray  # 1.0 = media de la liga; >1 concede más xG


def fit_ratings(window: pd.DataFrame) -> LeagueRatings:
    """Modelo multiplicativo xG = μ_sede · ataque · defensa_rival, ajustado por
    ajuste proporcional iterativo (equivale al máximo de verosimilitud de
    Poisson sobre el xG), de modo que cada fuerza queda corregida por rival."""
    teams = sorted(set(window["home"]) | set(window["away"]))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h = window["home"].map(index).to_numpy()
    a = window["away"].map(index).to_numpy()
    hxg = window["hxg"].to_numpy(float)
    axg = window["axg"].to_numpy(float)
    mu_h, mu_a = hxg.mean(), axg.mean()

    xg_for = np.bincount(h, hxg, n) + np.bincount(a, axg, n)
    xg_against = np.bincount(h, axg, n) + np.bincount(a, hxg, n)
    attack, defense = np.ones(n), np.ones(n)
    for _ in range(IPF_ITERATIONS):
        attack = xg_for / (np.bincount(h, mu_h * defense[a], n) + np.bincount(a, mu_a * defense[h], n))
        defense = xg_against / (np.bincount(h, mu_a * attack[a], n) + np.bincount(a, mu_h * attack[h], n))
    games = np.bincount(h, minlength=n) + np.bincount(a, minlength=n)
    scale = np.average(attack, weights=games)
    return LeagueRatings(index, mu_h, mu_a, float(window["hg"].mean()), float(window["ag"].mean()),
                         attack / scale, defense * scale)


def rest_category(days: float) -> str:
    if np.isnan(days) or days >= 8:
        return "largo"
    return "corto" if days <= 4 else "normal"


def team_snapshot(index: MatchIndex, team: str, venue: str, cutoff: pd.Timestamp,
                  ratings: LeagueRatings, with_rows: bool = False) -> dict | None:
    """Estado de un equipo justo antes de `cutoff` (solo partidos anteriores)."""
    arrays = index.team_arrays.get(team)
    if arrays is None or team not in ratings.index:
        return None
    times = arrays["times"]
    end = int(np.searchsorted(times, cutoff.to_datetime64(), "left"))
    start = int(np.searchsorted(times, (cutoff - pd.Timedelta(days=WINDOW_DAYS)).to_datetime64(), "left"))
    if end <= start:
        return None

    recent = slice(max(start, end - N_RECENT), end)
    opp = np.array([ratings.index[o] for o in arrays["opp"][recent]])
    home = arrays["home"][recent]
    mu_for = np.where(home, ratings.mu_home, ratings.mu_away)
    mu_against = np.where(home, ratings.mu_away, ratings.mu_home)
    form = slice(max(start, end - N_FORM), end)
    same_venue = np.flatnonzero(arrays["home"][start:end] == (venue == "h"))[-N_FORM:] + start
    pts = arrays["pts"]

    snap = {
        "n_window": end - start,
        "n_recent": recent.stop - recent.start,
        "att_recent": float(arrays["xgf"][recent].sum() / (mu_for * ratings.defense[opp]).sum()),
        "def_recent": float(arrays["xga"][recent].sum() / (mu_against * ratings.attack[opp]).sum()),
        "att_long": float(ratings.attack[ratings.index[team]]),
        "def_long": float(ratings.defense[ratings.index[team]]),
        "ppg5": float(pts[form].mean()),
        "ppg5_venue": float(pts[same_venue].mean()) if len(same_venue) else float(pts[form].mean()),
        "rest_days": float((cutoff.to_datetime64() - times[end - 1]) / np.timedelta64(1, "D")),
    }
    if with_rows:
        rows = index.team_rows[team]
        snap["recent_rows"] = rows.iloc[recent]
        snap["form_rows"] = rows.iloc[form]
        snap["venue_rows"] = rows.iloc[same_venue]
    return snap


def match_snapshot(index: MatchIndex, home: str, away: str, cutoff: pd.Timestamp,
                   ratings_cache: dict, with_rows: bool = False) -> tuple[dict, dict, LeagueRatings] | None:
    if cutoff not in ratings_cache:
        window = index.window(cutoff)
        ratings_cache[cutoff] = fit_ratings(window) if len(window) >= MIN_WINDOW_MATCHES else None
    ratings = ratings_cache[cutoff]
    if ratings is None:
        return None
    h = team_snapshot(index, home, "h", cutoff, ratings, with_rows)
    a = team_snapshot(index, away, "a", cutoff, ratings, with_rows)
    if h is None or a is None:
        return None
    return h, a, ratings


def blended(snap: dict, shrink: float, kind: str) -> float:
    w = snap["n_recent"] / (snap["n_recent"] + shrink)
    return w * snap[f"{kind}_recent"] + (1 - w) * snap[f"{kind}_long"]


def expected_goals(h: dict, a: dict, ratings: LeagueRatings, shrink: float) -> tuple[float, float]:
    """λ = goles medios reales de la liga (por sede) × ataque propio × defensa rival.
    El xG fija la fuerza relativa; el nivel sale de los goles reales porque el
    xG de Understat va por encima de los goles marcados en las últimas temporadas."""
    lam_h = ratings.goals_home * blended(h, shrink, "att") * blended(a, shrink, "def")
    lam_a = ratings.goals_away * blended(a, shrink, "att") * blended(h, shrink, "def")
    return float(np.clip(lam_h, LAMBDA_MIN, LAMBDA_MAX)), float(np.clip(lam_a, LAMBDA_MIN, LAMBDA_MAX))


def score_matrix(lam_h: float, lam_a: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
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


def top_scores(matrix: np.ndarray, k: int = 3) -> list[tuple[int, int, float]]:
    flat = np.argsort(matrix, axis=None)[::-1][:k]
    return [(int(i), int(j), float(matrix[i, j])) for i, j in zip(*np.unravel_index(flat, matrix.shape))]


# --------------------------------------------------------------------------
# Dataset histórico, regresión logística y ensemble
# --------------------------------------------------------------------------


def snapshot_columns(h: dict, a: dict) -> dict:
    """Variables numéricas de las instantáneas local (h_) y visitante (a_)."""
    return {
        **{f"h_{k}": v for k, v in h.items() if not k.endswith("_rows")},
        **{f"a_{k}": v for k, v in a.items() if not k.endswith("_rows")},
    }


def build_history(index: MatchIndex, from_season: int) -> pd.DataFrame:
    """Instantánea pre-partido (solo con datos anteriores a cada partido) de
    todos los partidos jugados desde `from_season`."""
    rows, cache = [], {}
    played = index.played
    for m in played[played["season"] >= from_season].itertuples(index=False):
        snap = match_snapshot(index, m.home, m.away, m.datetime, cache)
        if snap is None:
            continue
        h, a, ratings = snap
        if h["n_window"] < MIN_MATCHES or a["n_window"] < MIN_MATCHES:
            continue
        rows.append(
            {
                "season": m.season,
                "datetime": m.datetime,
                "hg": m.hg,
                "ag": m.ag,
                "result": "H" if m.hg > m.ag else ("D" if m.hg == m.ag else "A"),
                "goals_home": ratings.goals_home,
                "goals_away": ratings.goals_away,
                **snapshot_columns(h, a),
            }
        )
    return pd.DataFrame(rows)


def history_lambdas(hist: pd.DataFrame, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    def mix(side: str, kind: str) -> np.ndarray:
        w = hist[f"{side}_n_recent"] / (hist[f"{side}_n_recent"] + shrink)
        return (w * hist[f"{side}_{kind}_recent"] + (1 - w) * hist[f"{side}_{kind}_long"]).to_numpy()

    lam_h = np.clip(hist["goals_home"].to_numpy() * mix("h", "att") * mix("a", "def"), LAMBDA_MIN, LAMBDA_MAX)
    lam_a = np.clip(hist["goals_away"].to_numpy() * mix("a", "att") * mix("h", "def"), LAMBDA_MIN, LAMBDA_MAX)
    return lam_h, lam_a


def history_features(hist: pd.DataFrame, lam_h: np.ndarray, lam_a: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_lambda_ratio": np.log(lam_h / lam_a),
            "log_lambda_total": np.log(lam_h + lam_a),
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


@dataclass
class TrainedModels:
    shrink: float
    logistic: Pipeline
    w_poisson: float
    n_train: int
    n_val: int
    val_seasons: str
    ll_poisson: float
    ll_logistic: float
    ll_ensemble: float


def train_models(hist: pd.DataFrame, val_from_season: int) -> TrainedModels:
    train_mask = (hist["season"] < val_from_season).to_numpy()
    val_mask = ~train_mask
    if train_mask.sum() < 200 or val_mask.sum() < 50:
        raise PredictionError("Histórico insuficiente para entrenar y validar el modelo (amplía las temporadas).")

    # 1) Mezcla forma reciente vs. 12 meses: máxima verosimilitud de los goles reales (solo entrenamiento).
    train = hist[train_mask]
    best_shrink, best_ll = SHRINK_GRID[0], -np.inf
    for shrink in SHRINK_GRID:
        lam_h, lam_a = history_lambdas(train, shrink)
        ll = poisson.logpmf(train["hg"], lam_h).sum() + poisson.logpmf(train["ag"], lam_a).sum()
        if ll > best_ll:
            best_shrink, best_ll = shrink, ll

    lam_h, lam_a = history_lambdas(hist, best_shrink)
    p_poisson = np.array([outcome_probs(score_matrix(lh, la)) for lh, la in zip(lam_h, lam_a)])
    features = history_features(hist, lam_h, lam_a)
    y = hist["result"].to_numpy()

    # 2) Regresión logística entrenada solo con temporadas anteriores a la validación.
    logistic = fit_logistic(features[train_mask], y[train_mask])
    p_logistic = predict_hda(logistic, features[val_mask])

    # 3) Peso del ensemble que minimiza el log loss de validación.
    y_val, p_pois_val = y[val_mask], p_poisson[val_mask]
    losses = [multiclass_log_loss(y_val, w * p_pois_val + (1 - w) * p_logistic) for w in ENSEMBLE_WEIGHT_GRID]
    w_poisson = float(ENSEMBLE_WEIGHT_GRID[int(np.argmin(losses))])

    seasons_val = sorted(hist.loc[val_mask, "season"].unique())
    return TrainedModels(
        shrink=best_shrink,
        logistic=fit_logistic(features, y),  # modelo final: todo el histórico
        w_poisson=w_poisson,
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        val_seasons=" + ".join(season_label(s) for s in seasons_val),
        ll_poisson=multiclass_log_loss(y_val, p_pois_val),
        ll_logistic=multiclass_log_loss(y_val, p_logistic),
        ll_ensemble=float(min(losses)),
    )


# --------------------------------------------------------------------------
# API de alto nivel: cargar liga, entrenar, predecir
# --------------------------------------------------------------------------


@dataclass
class LeagueData:
    league: str
    current_season: int
    matches: pd.DataFrame  # todos los partidos (jugados y por jugar), horas en UTC
    index: MatchIndex  # partidos jugados con xG
    sources: list[SourceInfo]

    @property
    def teams(self) -> list[str]:
        season = self.matches[self.matches["season"] == self.current_season]
        return sorted(set(season["home"]) | set(season["away"]))


def load_league(league: str, seasons_back: int = DEFAULT_SEASONS_BACK, refresh: bool = False,
                now: pd.Timestamp | None = None, progress: Callable[[str], None] | None = None) -> LeagueData:
    """Descarga la temporada en curso y `seasons_back` + 1 anteriores (la más
    antigua solo sirve de historial previo para las primeras instantáneas)."""
    now = now if now is not None else utc_now()
    current = current_season_for(now)
    session = requests.Session()
    dates_by_season, sources = {}, []
    for season in range(current, current - seasons_back - 2, -1):
        if progress:
            progress(f"Descargando Understat {league_name(league)} {season_label(season)}...")
        dates, info = fetch_season(session, league, season, current, refresh)
        if dates:
            dates_by_season[season] = dates
            sources.append(info)
    if not dates_by_season:
        raise PredictionError(f"Understat no devolvió partidos de {league_name(league)}.")
    matches = matches_frame(dates_by_season)
    played = matches[matches["played"] & matches["hxg"].notna() & matches["axg"].notna()]
    return LeagueData(league, current, matches, MatchIndex(played), sources)


@dataclass
class LeagueModel:
    data: LeagueData
    trained: TrainedModels


def train_league(data: LeagueData, seasons_back: int = DEFAULT_SEASONS_BACK) -> LeagueModel:
    first_training_season = data.current_season - seasons_back
    hist = build_history(data.index, first_training_season)
    if hist.empty:
        raise PredictionError(f"Sin histórico suficiente para entrenar {league_name(data.league)}.")
    return LeagueModel(data, train_models(hist, val_from_season=data.current_season - 1))


@dataclass
class MatchPrediction:
    league: str
    home: str
    away: str
    cutoff: pd.Timestamp  # instante (UTC) hasta el que se usan datos
    match_id: int | None  # id de Understat si el partido está en el calendario
    kickoff: pd.Timestamp | None  # hora de inicio en UTC
    final_score: tuple[int, int] | None  # resultado real si ya se jugó
    lam_home: float
    lam_away: float
    p_poisson: np.ndarray  # [local, empate, visitante]
    p_logistic: np.ndarray
    p_final: np.ndarray
    over25: float
    btts: float
    over25_poisson: float
    btts_poisson: float
    top_scores: list[tuple[int, int, float]]  # de la matriz de Poisson puro
    home_snap: dict = field(repr=False)
    away_snap: dict = field(repr=False)
    ratings: LeagueRatings = field(repr=False)
    trained: TrainedModels = field(repr=False)

    @property
    def understat_url(self) -> str | None:
        return f"{UNDERSTAT}/match/{self.match_id}" if self.match_id is not None else None

    def attack_defense(self, side: str) -> tuple[float, float]:
        snap = self.home_snap if side == "home" else self.away_snap
        shrink = self.trained.shrink
        return blended(snap, shrink, "att"), blended(snap, shrink, "def")


def find_fixture(data: LeagueData, home: str, away: str, now: pd.Timestamp | None = None,
                 horizon_days: int = FIXTURE_HORIZON_DAYS) -> pd.Series | None:
    """Próximo partido home-away aún sin jugar (o empezado hace menos de 3 h) en los
    próximos `horizon_days` días."""
    now = now if now is not None else utc_now()
    m = data.matches
    upcoming = m[(~m["played"]) & (m["home"] == home) & (m["away"] == away)
                 & (m["datetime"] >= now - pd.Timedelta(hours=3))
                 & (m["datetime"] <= now + pd.Timedelta(days=horizon_days))]
    return upcoming.iloc[0] if len(upcoming) else None


def predict_match(model: LeagueModel, home: str, away: str, cutoff: pd.Timestamp | None = None,
                  fixture: pd.Series | None = None) -> MatchPrediction:
    """Predice home-away con los datos anteriores a `cutoff`. Si se pasa el
    `fixture` (fila de `data.matches`), el corte es su hora de inicio."""
    data, trained = model.data, model.trained
    home = resolve_team(home, data.teams)
    away = resolve_team(away, data.teams)
    if home == away:
        raise PredictionError("Local y visitante son el mismo equipo.")
    if fixture is None and cutoff is None:
        fixture = find_fixture(data, home, away)
    if fixture is not None:
        cutoff = fixture["datetime"]
    if cutoff is None:
        cutoff = utc_now()

    snap = match_snapshot(data.index, home, away, cutoff, {}, with_rows=True)
    if snap is None:
        raise PredictionError(f"Sin datos de xG en los últimos {WINDOW_DAYS} días para {home} o {away}.")
    h, a, ratings = snap
    lam_h, lam_a = expected_goals(h, a, ratings, trained.shrink)
    matrix = score_matrix(lam_h, lam_a)
    p_poisson = outcome_probs(matrix)
    features = history_features(pd.DataFrame([snapshot_columns(h, a)]), np.array([lam_h]), np.array([lam_a]))
    p_logistic = predict_hda(trained.logistic, features)[0]
    p_final = trained.w_poisson * p_poisson + (1 - trained.w_poisson) * p_logistic
    over25, btts = goal_markets(reweight_matrix(matrix, p_final))
    over25_poisson, btts_poisson = goal_markets(matrix)

    final_score = None
    if fixture is not None and bool(fixture["played"]):
        final_score = (int(fixture["hg"]), int(fixture["ag"]))
    return MatchPrediction(
        league=data.league,
        home=home,
        away=away,
        cutoff=cutoff,
        match_id=int(fixture["id"]) if fixture is not None else None,
        kickoff=fixture["datetime"] if fixture is not None else None,
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
        home_snap=h,
        away_snap=a,
        ratings=ratings,
        trained=trained,
    )


def fixtures_between(data: LeagueData, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """Partidos (jugados o no) con inicio en [start_utc, end_utc)."""
    m = data.matches
    return m[(m["datetime"] >= start_utc) & (m["datetime"] < end_utc)]


def form_string(rows: pd.DataFrame) -> str:
    """Racha en letras (G/E/P), del partido más antiguo al más reciente."""
    return " ".join({3: "G", 1: "E", 0: "P"}[int(p)] for p in rows["pts"])
