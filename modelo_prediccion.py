#!/usr/bin/env python3
"""Predicción de un partido con datos reales de Understat (xG): Poisson + regresión logística.

Descarga de understat.com el xG partido a partido de la liga (temporada en
curso y anteriores), estima la fuerza de ataque/defensa de cada equipo
ajustada por la calidad de sus rivales y combina dos modelos:

- Poisson puro: goles esperados (λ) local/visitante a partir de la fuerza
  derivada del xG de los últimos 10 partidos, mezclada con la fuerza de los
  últimos 12 meses (el peso de la mezcla se elige con datos históricos).
- Regresión logística multinomial: reajusta la señal de Poisson con la racha
  de puntos (últimos 5), el rendimiento en casa/fuera (localía) y los días
  de descanso (categóricos), entrenada con temporadas anteriores.

El peso de cada modelo en el ensemble se elige minimizando el log loss sobre
partidos de validación que la regresión logística no vio al entrenar.

Uso:
    python3 modelo_prediccion.py "Arsenal" "Leeds" --liga "Premier League"
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
import unicodedata
import warnings
from dataclasses import dataclass
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
CACHE_DIR = Path(__file__).resolve().parent / "data" / "understat_cache"
CURRENT_SEASON_CACHE_TTL_S = 3 * 3600  # temporadas pasadas no cambian: caché sin caducidad

# Código de liga en Understat -> (nombre visible, alias aceptados en --liga).
LEAGUES = {
    "EPL": ("Premier League", ["premier league", "premier", "epl", "inglaterra", "england"]),
    "La_liga": ("LaLiga", ["laliga", "la liga", "liga espanola", "espana", "spain"]),
    "Bundesliga": ("Bundesliga", ["bundesliga", "alemania", "germany"]),
    "Serie_A": ("Serie A", ["serie a", "italia", "italy"]),
    "Ligue_1": ("Ligue 1", ["ligue 1", "francia", "france"]),
    "RFPL": ("Liga Premier de Rusia", ["rfpl", "rusia", "russia"]),
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
MIN_MATCHES = 5  # mínimo de partidos en la ventana para usar un partido al entrenar
IPF_ITERATIONS = 25
MAX_GOALS = 10
LAMBDA_MIN, LAMBDA_MAX = 0.1, 5.0
# Peso (en "partidos equivalentes") de la fuerza de 12 meses frente a los últimos 10.
SHRINK_GRID = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 1e9)
ENSEMBLE_WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
CLASSES = ["H", "D", "A"]

NUMERIC_FEATURES = [
    "log_lambda_ratio",
    "log_lambda_total",
    "ppg5_home",
    "ppg5_away",
    "ppg5_home_at_home",
    "ppg5_away_away",
]
CATEGORICAL_FEATURES = ["rest_home", "rest_away"]

SEP = "=" * 50


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def season_label(season: int) -> str:
    return f"{season}/{(season + 1) % 100:02d}"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(text.lower().replace("-", " ").replace(".", " ").split())


def resolve_league(name: str) -> str:
    key = normalize(name)
    for code, (display, aliases) in LEAGUES.items():
        if key == normalize(code) or key == normalize(display) or key in aliases:
            return code
    valid = ", ".join(display for display, _ in LEAGUES.values())
    raise SystemExit(f"Liga no reconocida: '{name}'. Understat cubre: {valid}.")


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
    raise SystemExit(f"Equipo no encontrado: '{name}'. Equipos de la temporada: {', '.join(sorted(teams))}.")


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
    session: requests.Session, league: str, season: int, current_season: int, refresh: bool
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
        raise SystemExit(f"No se pudo descargar {url}: {last_error}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(dates), encoding="utf-8")
    return dates, SourceInfo(url, season, sum(bool(m.get("isResult")) for m in dates), False)


def matches_frame(dates_by_season: dict[int, list[dict]]) -> pd.DataFrame:
    """Convierte el JSON `dates` de Understat en una fila por partido."""
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


def recent_strength(rows: pd.DataFrame, ratings: LeagueRatings) -> tuple[float, float]:
    """Ataque/defensa de los partidos dados: xG real / xG esperado según sede y rival."""
    opp = rows["opp"].map(ratings.index).to_numpy()
    home = (rows["venue"] == "h").to_numpy()
    mu_for = np.where(home, ratings.mu_home, ratings.mu_away)
    mu_against = np.where(home, ratings.mu_away, ratings.mu_home)
    attack = rows["xgf"].sum() / (mu_for * ratings.defense[opp]).sum()
    defense = rows["xga"].sum() / (mu_against * ratings.attack[opp]).sum()
    return float(attack), float(defense)


def rest_category(days: float) -> str:
    if np.isnan(days) or days >= 8:
        return "largo"
    return "corto" if days <= 4 else "normal"


def team_snapshot(long: pd.DataFrame, team: str, venue: str, cutoff: pd.Timestamp,
                  ratings: LeagueRatings) -> dict | None:
    start = cutoff - pd.Timedelta(days=WINDOW_DAYS)
    rows = long[(long["team"] == team) & (long["datetime"] < cutoff) & (long["datetime"] >= start)]
    if rows.empty or team not in ratings.index:
        return None
    recent = rows.tail(N_RECENT)
    att_recent, def_recent = recent_strength(recent, ratings)
    form = rows.tail(N_FORM)
    at_venue = rows[rows["venue"] == venue].tail(N_FORM)
    rest_days = (cutoff - rows["datetime"].iloc[-1]).total_seconds() / 86400
    i = ratings.index[team]
    return {
        "n_window": len(rows),
        "n_recent": len(recent),
        "att_recent": att_recent,
        "def_recent": def_recent,
        "att_long": float(ratings.attack[i]),
        "def_long": float(ratings.defense[i]),
        "ppg5": float(form["pts"].mean()),
        "ppg5_venue": float(at_venue["pts"].mean()) if len(at_venue) else float(form["pts"].mean()),
        "rest_days": rest_days,
        "recent_rows": recent,
        "form_rows": form,
        "venue_rows": at_venue,
    }


def match_snapshot(played: pd.DataFrame, long: pd.DataFrame, home: str, away: str,
                   cutoff: pd.Timestamp, ratings_cache: dict) -> tuple[dict, dict, LeagueRatings] | None:
    if cutoff not in ratings_cache:
        window = played[(played["datetime"] < cutoff)
                         & (played["datetime"] >= cutoff - pd.Timedelta(days=WINDOW_DAYS))]
        ratings_cache[cutoff] = fit_ratings(window) if len(window) >= 50 else None
    ratings = ratings_cache[cutoff]
    if ratings is None:
        return None
    h = team_snapshot(long, home, "h", cutoff, ratings)
    a = team_snapshot(long, away, "a", cutoff, ratings)
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


def build_history(played: pd.DataFrame, long: pd.DataFrame, from_season: int) -> pd.DataFrame:
    """Instantánea pre-partido (solo con datos anteriores a cada partido) de
    todos los partidos jugados desde `from_season`."""
    rows, cache = [], {}
    for m in played[played["season"] >= from_season].itertuples(index=False):
        snap = match_snapshot(played, long, m.home, m.away, m.datetime, cache)
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
        raise SystemExit("Histórico insuficiente para entrenar/validar (amplía --temporadas).")

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
# Salida
# --------------------------------------------------------------------------


def pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def form_string(rows: pd.DataFrame) -> str:
    return " ".join({3: "G", 1: "E", 0: "P"}[int(p)] for p in rows["pts"])


def team_metrics_line(name: str, role: str, snap: dict) -> list[str]:
    recent = snap["recent_rows"]
    venue_txt = "en casa" if role == "local" else "fuera"
    first, last = recent["datetime"].iloc[0], recent["datetime"].iloc[-1]
    seasons = " + ".join(season_label(s) for s in sorted(recent["season"].unique()))
    return [
        f"    {name} ({role}) — últimos {len(recent)} partidos de liga "
        f"[{first:%Y-%m-%d} → {last:%Y-%m-%d}, temporadas {seasons}]",
        f"      xG a favor {recent['xgf'].mean():.2f} | xG en contra {recent['xga'].mean():.2f} | "
        f"Goles a favor {recent['gf'].mean():.2f} | Goles en contra {recent['ga'].mean():.2f} (por partido)",
        f"      Forma últ. {len(snap['form_rows'])}: {form_string(snap['form_rows'])} "
        f"= {int(snap['form_rows']['pts'].sum())} pts | Últ. {len(snap['venue_rows'])} {venue_txt}: "
        f"{form_string(snap['venue_rows'])} = {int(snap['venue_rows']['pts'].sum())} pts | "
        f"Descanso: {snap['rest_days']:.0f} días ({rest_category(snap['rest_days'])})",
    ]


def detail_table(name: str, snap: dict) -> list[str]:
    lines = [f"    {name}: fecha | sede | rival | resultado | xG"]
    for r in snap["recent_rows"].itertuples(index=False):
        lines.append(
            f"      {r.datetime:%Y-%m-%d} | {'L' if r.venue == 'h' else 'V'} | {r.opp:<24} | "
            f"{int(r.gf)}-{int(r.ga)} | {r.xgf:.2f}-{r.xga:.2f}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predicción Poisson + regresión logística con xG de Understat.")
    parser.add_argument("local", help="Equipo local (p. ej. 'Arsenal')")
    parser.add_argument("visitante", help="Equipo visitante (p. ej. 'Leeds')")
    parser.add_argument("--liga", default="Premier League", help="Liga (Premier League, LaLiga, Serie A, ...)")
    parser.add_argument("--temporadas", type=int, default=4,
                        help="Temporadas anteriores usadas para entrenar/validar (por defecto 4)")
    parser.add_argument("--refrescar", action="store_true", help="Ignora la caché y vuelve a descargar")
    parser.add_argument("--detalle", action="store_true", help="Muestra los últimos partidos usados de cada equipo")
    args = parser.parse_args(argv)

    league = resolve_league(args.liga)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    current_season = now.year if now.month >= 7 else now.year - 1
    first_season = current_season - args.temporadas - 1  # una temporada extra de calentamiento

    session = requests.Session()
    dates_by_season, sources = {}, []
    for season in range(current_season, first_season - 1, -1):
        log(f"Descargando Understat {LEAGUES[league][0]} {season_label(season)}...")
        dates, info = fetch_season(session, league, season, current_season, args.refrescar)
        if dates:
            dates_by_season[season] = dates
            sources.append(info)

    matches = matches_frame(dates_by_season)
    played = matches[matches["played"] & matches["hxg"].notna() & matches["axg"].notna()].reset_index(drop=True)
    long = team_long_frame(played)

    season_teams = sorted(set(matches.loc[matches["season"] == current_season, "home"]))
    home = resolve_team(args.local, season_teams or sorted(set(matches["home"])))
    away = resolve_team(args.visitante, season_teams or sorted(set(matches["home"])))
    if home == away:
        raise SystemExit("Local y visitante son el mismo equipo.")

    fixture = matches[(~matches["played"]) & (matches["home"] == home) & (matches["away"] == away)
                      & (matches["datetime"] >= now - pd.Timedelta(hours=3))].head(1)
    cutoff = fixture["datetime"].iloc[0] if len(fixture) else now

    log("Construyendo histórico pre-partido y entrenando modelos...")
    hist = build_history(played, long, first_season + 1)
    models = train_models(hist, val_from_season=current_season - 1)

    snap = match_snapshot(played, long, home, away, cutoff, {})
    if snap is None:
        raise SystemExit(f"Sin datos de xG en los últimos {WINDOW_DAYS} días para {home} o {away}.")
    h, a, ratings = snap
    lam_h, lam_a = expected_goals(h, a, ratings, models.shrink)
    matrix = score_matrix(lam_h, lam_a)
    p_poisson = outcome_probs(matrix)
    target_features = history_features(pd.DataFrame([snapshot_columns(h, a)]), np.array([lam_h]), np.array([lam_a]))
    p_logistic = predict_hda(models.logistic, target_features)[0]
    p_final = models.w_poisson * p_poisson + (1 - models.w_poisson) * p_logistic
    final_matrix = reweight_matrix(matrix, p_final)
    over25, btts = goal_markets(final_matrix)
    over25_poisson, btts_poisson = goal_markets(matrix)

    w_recent_h = h["n_recent"] / (h["n_recent"] + models.shrink)
    out = [SEP, "🌐 FUENTES DE DATOS LOCALIZADAS EN LA WEB", "- URLs analizadas para el scraping:"]
    for s in sources:
        cache_txt = " (caché local)" if s.from_cache else ""
        out.append(f"    {s.url}  [{season_label(s.season)}: {s.played} partidos con xG]{cache_txt}")
    if len(fixture):
        out.append(f"- Partido: {home} vs {away} | {LEAGUES[league][0]} | {cutoff:%Y-%m-%d %H:%M} UTC | "
                   f"{UNDERSTAT}/match/{int(fixture['id'].iloc[0])}")
    else:
        out.append(f"- Partido: {home} vs {away} | {LEAGUES[league][0]} | no figura en el calendario de "
                   f"Understat: se usan los datos disponibles a {cutoff:%Y-%m-%d}")
    out.append("- Métricas extraídas:")
    out += team_metrics_line(home, "local", h)
    out += team_metrics_line(away, "visitante", a)
    if args.detalle:
        out += detail_table(home, h) + detail_table(away, a)
    out += [
        f"- Fuerza ajustada por rival (1.00 = media liga; {w_recent_h:.0%} últimos {N_RECENT} / "
        f"{1 - w_recent_h:.0%} últimos 12 meses):",
        f"    {home}: ataque {blended(h, models.shrink, 'att'):.2f} | defensa {blended(h, models.shrink, 'def'):.2f}"
        f"   ·   {away}: ataque {blended(a, models.shrink, 'att'):.2f} | defensa {blended(a, models.shrink, 'def'):.2f}",
        f"- Media de la liga (12 meses): xG local {ratings.mu_home:.2f} / visitante {ratings.mu_away:.2f} | "
        f"goles local {ratings.goals_home:.2f} / visitante {ratings.goals_away:.2f}",
        f"- Goles esperados del partido: λ local {lam_h:.2f} | λ visitante {lam_a:.2f}",
        "- Bajas/lesiones: Understat no las publica → no incluidas en el modelo (no se infieren).",
        "- Descanso: calculado solo con partidos de liga (no ve copas ni competiciones europeas).",
        SEP,
        "🧮 RESULTADOS DE LOS MODELOS (POISSON & LOGÍSTICO)",
        f"- Poisson Puro -> 1: {pct(p_poisson[0])} | X: {pct(p_poisson[1])} | 2: {pct(p_poisson[2])}",
        f"- Regresión Logística -> 1: {pct(p_logistic[0])} | X: {pct(p_logistic[1])} | 2: {pct(p_logistic[2])}",
        "- Marcadores exactos más probables: "
        + " | ".join(f"{i}-{j} ({pct(p)})" for i, j, p in top_scores(matrix)),
        f"- Poisson Puro goles -> Over 2.5: {pct(over25_poisson)} | Ambos anotan: {pct(btts_poisson)}",
        f"- Validación ({models.val_seasons}, {models.n_val} partidos no vistos; entrenamiento "
        f"{models.n_train}) log loss 1X2 -> Poisson {models.ll_poisson:.4f} | Logística "
        f"{models.ll_logistic:.4f} | Ensemble {models.ll_ensemble:.4f}",
        SEP,
        "🎯 PREDICCIÓN FINAL COMBINADA (ENSEMBLE MODEL)",
        f"- Pesos: Poisson {models.w_poisson:.0%} | Logística {1 - models.w_poisson:.0%} "
        "(mínimo log loss de validación)",
        f"- Mercado 1X2: Local {pct(p_final[0])} | Empate {pct(p_final[1])} | Visitante {pct(p_final[2])}",
        f"- Línea de Goles: Over 2.5 {pct(over25)} | Under 2.5 {pct(1 - over25)}",
        f"- Ambos Anotan: SÍ {pct(btts)} | NO {pct(1 - btts)}",
        SEP,
    ]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
