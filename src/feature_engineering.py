"""Construcción de variables predictivas (features) para el modelo de
respaldo (regresión logística).

La regla central de este módulo es evitar la fuga de información: para
calcular las variables de un partido, solo se usan partidos con fecha
ESTRICTAMENTE anterior a ese partido. Esto aplica tanto al construir el
dataset de entrenamiento (una fila por partido histórico) como al construir
las variables para una predicción en vivo (donde el "corte" es la fecha del
último partido disponible en el archivo, es decir, se usa todo lo histórico).

Reutiliza las funciones de `statistics.py` sobre subconjuntos del dataframe
ya filtrados por fecha, en vez de reimplementar el cálculo de récords/forma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import statistics

# Cantidad mínima de partidos previos (en cualquier condición) que debe
# tener CADA equipo para que un partido histórico se use como fila de
# entrenamiento. Evita generar variables sobre una base casi vacía en las
# primeras jornadas de una temporada.
MIN_PRIOR_MATCHES = 3

FEATURE_COLUMNS = [
    "home_form_ppg",
    "away_form_ppg",
    "quality_diff_ppg",
    "home_home_win_pct",
    "away_away_win_pct",
    "home_overall_win_pct",
    "away_overall_win_pct",
    "home_avg_gf_home",
    "home_avg_ga_home",
    "away_avg_gf_away",
    "away_avg_ga_away",
    "home_clean_sheet_pct_home",
    "away_clean_sheet_pct_away",
    "home_btts_pct_overall",
    "away_btts_pct_overall",
    "home_matches_played_overall",
    "away_matches_played_overall",
    "rest_days_home",
    "rest_days_away",
]

FEATURE_LABELS = {
    "home_form_ppg": "Forma local (puntos/partido, reciente)",
    "away_form_ppg": "Forma visitante (puntos/partido, reciente)",
    "quality_diff_ppg": "Diferencia de forma (local - visitante)",
    "home_home_win_pct": "% Victorias del local jugando en casa",
    "away_away_win_pct": "% Victorias del visitante jugando fuera",
    "home_overall_win_pct": "% Victorias generales del local",
    "away_overall_win_pct": "% Victorias generales del visitante",
    "home_avg_gf_home": "Goles anotados/partido del local (en casa)",
    "home_avg_ga_home": "Goles recibidos/partido del local (en casa)",
    "away_avg_gf_away": "Goles anotados/partido del visitante (fuera)",
    "away_avg_ga_away": "Goles recibidos/partido del visitante (fuera)",
    "home_clean_sheet_pct_home": "% Porterías a cero del local (en casa)",
    "away_clean_sheet_pct_away": "% Porterías a cero del visitante (fuera)",
    "home_btts_pct_overall": "% Ambos anotan — partidos del local",
    "away_btts_pct_overall": "% Ambos anotan — partidos del visitante",
    "home_matches_played_overall": "Partidos previos disponibles del local",
    "away_matches_played_overall": "Partidos previos disponibles del visitante",
    "rest_days_home": "Días de descanso del local",
    "rest_days_away": "Días de descanso del visitante",
}


def _rest_days(prior_matches: pd.DataFrame, team: str, cutoff_date) -> float:
    """Días transcurridos desde el último partido del equipo antes de
    `cutoff_date`. Devuelve NaN si no hay partido previo o no hay fechas."""
    if "date" not in prior_matches.columns:
        return np.nan
    team_matches = statistics.team_matches(prior_matches, team, venue="all")
    dates = team_matches["date"].dropna()
    if dates.empty or cutoff_date is None or pd.isna(cutoff_date):
        return np.nan
    last_date = dates.max()
    return float((cutoff_date - last_date).days)


def build_feature_row(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    cutoff_date=None,
    recent_n: int = 5,
) -> dict[str, float]:
    """Calcula el vector de variables para un partido (histórico o en vivo).

    `cutoff_date=None` significa "usar todo el historial disponible" (caso
    de una predicción en vivo, hacia un partido futuro real). Cuando se usa
    para construir el dataset de entrenamiento, se pasa la fecha exacta del
    partido histórico y solo se consideran partidos anteriores a esa fecha.
    """
    if cutoff_date is not None:
        prior = df[df["date"] < cutoff_date]
    else:
        prior = df

    home_overall = statistics.compute_team_record(prior, home_team, venue="all")
    home_home = statistics.compute_team_record(prior, home_team, venue="home")
    away_overall = statistics.compute_team_record(prior, away_team, venue="all")
    away_away = statistics.compute_team_record(prior, away_team, venue="away")

    home_form = statistics.compute_recent_form(prior, home_team, n=recent_n)
    away_form = statistics.compute_recent_form(prior, away_team, n=recent_n)

    home_ppg = home_form.points_per_game if home_form.n_available > 0 else np.nan
    away_ppg = away_form.points_per_game if away_form.n_available > 0 else np.nan

    live_cutoff = cutoff_date if cutoff_date is not None else (
        df["date"].max() if "date" in df.columns and df["date"].notna().any() else None
    )

    return {
        "home_form_ppg": home_ppg if home_ppg is not None else np.nan,
        "away_form_ppg": away_ppg if away_ppg is not None else np.nan,
        "quality_diff_ppg": (home_ppg - away_ppg) if (home_ppg is not None and away_ppg is not None) else np.nan,
        "home_home_win_pct": _nan_if_none(home_home.win_pct),
        "away_away_win_pct": _nan_if_none(away_away.win_pct),
        "home_overall_win_pct": _nan_if_none(home_overall.win_pct),
        "away_overall_win_pct": _nan_if_none(away_overall.win_pct),
        "home_avg_gf_home": _nan_if_none(home_home.avg_goals_for),
        "home_avg_ga_home": _nan_if_none(home_home.avg_goals_against),
        "away_avg_gf_away": _nan_if_none(away_away.avg_goals_for),
        "away_avg_ga_away": _nan_if_none(away_away.avg_goals_against),
        "home_clean_sheet_pct_home": _nan_if_none(home_home.clean_sheet_pct),
        "away_clean_sheet_pct_away": _nan_if_none(away_away.clean_sheet_pct),
        "home_btts_pct_overall": _nan_if_none(home_overall.both_teams_scored_pct),
        "away_btts_pct_overall": _nan_if_none(away_overall.both_teams_scored_pct),
        "home_matches_played_overall": float(home_overall.played),
        "away_matches_played_overall": float(away_overall.played),
        "rest_days_home": _rest_days(prior, home_team, live_cutoff),
        "rest_days_away": _rest_days(prior, away_team, live_cutoff),
    }


def _nan_if_none(value):
    return np.nan if value is None else value


def build_training_dataset(
    df: pd.DataFrame, recent_n: int = 5, min_prior_matches: int = MIN_PRIOR_MATCHES
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, int]:
    """Construye el dataset de entrenamiento: una fila por partido histórico
    con datos suficientes de contexto previo (`min_prior_matches` partidos
    previos de cada equipo, en cualquier condición).

    Requiere que `df` venga ordenado cronológicamente ascendente (lo
    garantiza `data_cleaner.clean_data`) y con columna `date` válida.

    Devuelve (X, y, meta, partidos_descartados). `y` contiene 'H'/'D'/'A'
    calculado desde los goles reales del partido (no desde la columna
    'result', para ser robustos si esa columna no existe).
    """
    if "date" not in df.columns or df["date"].isna().all():
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=object), pd.DataFrame(), len(df)

    rows: list[dict] = []
    labels: list[str] = []
    meta_rows: list[dict] = []
    skipped = 0

    for _, match in df.iterrows():
        cutoff = match["date"]
        if pd.isna(cutoff):
            skipped += 1
            continue

        prior = df[df["date"] < cutoff]
        home_prior_count = len(statistics.team_matches(prior, match["home_team"], venue="all"))
        away_prior_count = len(statistics.team_matches(prior, match["away_team"], venue="all"))

        if home_prior_count < min_prior_matches or away_prior_count < min_prior_matches:
            skipped += 1
            continue

        features = build_feature_row(df, match["home_team"], match["away_team"], cutoff_date=cutoff, recent_n=recent_n)
        rows.append(features)

        if match["home_goals"] > match["away_goals"]:
            label = "H"
        elif match["home_goals"] < match["away_goals"]:
            label = "A"
        else:
            label = "D"
        labels.append(label)

        meta_row = {
            "date": cutoff,
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "actual_outcome": label,
        }
        # Las cuotas de mercado (si existen) se pasan tal cual para poder
        # comparar el modelo contra el mercado en `market_odds.py`. Nunca se
        # usan como variable de entrenamiento (eso sería circular: estaríamos
        # prediciendo el resultado a partir de la propia opinión del mercado).
        for odds_col in ("market_home_odds", "market_draw_odds", "market_away_odds"):
            if odds_col in df.columns:
                meta_row[odds_col] = match[odds_col]
        meta_rows.append(meta_row)

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = pd.Series(labels, name="result")
    meta = pd.DataFrame(meta_rows)
    return X, y, meta, skipped
