"""Modelo predictivo principal basado en distribución de Poisson.

Estima los goles esperados de cada equipo a partir de su fortaleza
ofensiva/defensiva relativa a los promedios de la liga contenida en el
archivo cargado, y a partir de ahí deriva la matriz de marcadores y todos
los mercados (1X2, over/under, ambos anotan, etc.).

No usa ninguna información externa al archivo cargado por el usuario.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.statistics import team_matches
from src.utils import CLASS_LABELS, get_logger

logger = get_logger(__name__)

# Umbrales de partidos (en el rol relevante: local jugando de local, o
# visitante jugando de visitante) usados para calificar la confianza del
# modelo de Poisson. La confianza combinada con más factores (regresión
# logística, consistencia entre modelos) se implementa en la Fase 2.
CONFIDENCE_LOW_THRESHOLD = 5
CONFIDENCE_HIGH_THRESHOLD = 15

MAX_GOALS_INTERNAL = 12  # rango usado internamente para calcular mercados con precisión
MAX_GOALS_DISPLAY = 6  # tamaño de la matriz de marcadores mostrada (0-6 x 0-6)

# Los goles esperados se acotan a un rango plausible para evitar
# predicciones degeneradas cuando la muestra es muy pequeña (p. ej. un
# equipo con un solo partido de local). Esto no oculta la incertidumbre:
# el nivel de confianza y las advertencias siguen reflejando la muestra.
EXPECTED_GOALS_MIN = 0.15
EXPECTED_GOALS_MAX = 5.5


@dataclass
class TeamStrength:
    attack: float
    defense: float
    matches_used: int
    used_fallback: bool


@dataclass
class PoissonPrediction:
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    prob_over: dict[float, float]
    prob_under: dict[float, float]
    prob_btts_yes: float
    prob_btts_no: float
    prob_clean_sheet_home: float
    prob_clean_sheet_away: float
    score_matrix: pd.DataFrame  # filas = goles local (0..6), columnas = goles visitante (0..6)
    top_scores: list[tuple[str, float]]
    home_matches_used: int
    away_matches_used: int
    confidence: str
    warnings: list[str] = field(default_factory=list)


def _league_averages(df: pd.DataFrame) -> tuple[float, float]:
    avg_home = df["home_goals"].mean()
    avg_away = df["away_goals"].mean()
    return float(avg_home), float(avg_away)


def _team_home_attack_defense(
    df: pd.DataFrame, team: str, league_avg_home: float, league_avg_away: float
) -> tuple[TeamStrength, TeamStrength]:
    """Fortaleza de ataque y defensa del equipo jugando de LOCAL."""
    matches = team_matches(df, team, venue="home")
    n = len(matches)
    if n == 0 or league_avg_home == 0 or league_avg_away == 0:
        return (
            TeamStrength(attack=1.0, defense=1.0, matches_used=n, used_fallback=True),
            TeamStrength(attack=1.0, defense=1.0, matches_used=n, used_fallback=True),
        )
    avg_scored = matches["home_goals"].mean()
    avg_conceded = matches["away_goals"].mean()
    attack = TeamStrength(avg_scored / league_avg_home, 0.0, n, False)
    defense = TeamStrength(0.0, avg_conceded / league_avg_away, n, False)
    return attack, defense


def _team_away_attack_defense(
    df: pd.DataFrame, team: str, league_avg_home: float, league_avg_away: float
) -> tuple[TeamStrength, TeamStrength]:
    """Fortaleza de ataque y defensa del equipo jugando de VISITANTE."""
    matches = team_matches(df, team, venue="away")
    n = len(matches)
    if n == 0 or league_avg_home == 0 or league_avg_away == 0:
        return (
            TeamStrength(attack=1.0, defense=1.0, matches_used=n, used_fallback=True),
            TeamStrength(attack=1.0, defense=1.0, matches_used=n, used_fallback=True),
        )
    avg_scored = matches["away_goals"].mean()
    avg_conceded = matches["home_goals"].mean()
    attack = TeamStrength(avg_scored / league_avg_away, 0.0, n, False)
    defense = TeamStrength(0.0, avg_conceded / league_avg_home, n, False)
    return attack, defense


def _classify_confidence(home_matches: int, away_matches: int) -> str:
    lowest = min(home_matches, away_matches)
    if lowest < CONFIDENCE_LOW_THRESHOLD:
        return "baja"
    if lowest < CONFIDENCE_HIGH_THRESHOLD:
        return "media"
    return "alta"


def _score_probability_matrix(expected_home: float, expected_away: float, max_goals: int) -> np.ndarray:
    home_probs = poisson.pmf(np.arange(max_goals + 1), expected_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), expected_away)
    return np.outer(home_probs, away_probs)


def predict_match(df: pd.DataFrame, home_team: str, away_team: str) -> PoissonPrediction:
    """Genera la predicción Poisson completa para un partido hipotético
    entre `home_team` (jugando de local) y `away_team` (jugando de
    visitante), usando únicamente los partidos históricos de `df`.
    """
    warnings: list[str] = []
    league_avg_home, league_avg_away = _league_averages(df)

    home_attack, home_defense = _team_home_attack_defense(df, home_team, league_avg_home, league_avg_away)
    away_attack, away_defense = _team_away_attack_defense(df, away_team, league_avg_home, league_avg_away)

    if home_attack.used_fallback:
        warnings.append(
            f"'{home_team}' no tiene partidos como local en los datos cargados; "
            "se usó el promedio general de la liga como referencia neutral."
        )
    if away_attack.used_fallback:
        warnings.append(
            f"'{away_team}' no tiene partidos como visitante en los datos cargados; "
            "se usó el promedio general de la liga como referencia neutral."
        )

    expected_home_goals = home_attack.attack * away_defense.defense * league_avg_home
    expected_away_goals = away_attack.attack * home_defense.defense * league_avg_away

    expected_home_goals = float(np.clip(expected_home_goals, EXPECTED_GOALS_MIN, EXPECTED_GOALS_MAX))
    expected_away_goals = float(np.clip(expected_away_goals, EXPECTED_GOALS_MIN, EXPECTED_GOALS_MAX))

    full_matrix = _score_probability_matrix(expected_home_goals, expected_away_goals, MAX_GOALS_INTERNAL)

    idx = np.arange(MAX_GOALS_INTERNAL + 1)
    home_idx, away_idx = np.meshgrid(idx, idx, indexing="ij")

    prob_home_win = float(full_matrix[home_idx > away_idx].sum())
    prob_draw = float(full_matrix[home_idx == away_idx].sum())
    prob_away_win = float(full_matrix[home_idx < away_idx].sum())

    total_goals_grid = home_idx + away_idx
    prob_over = {}
    prob_under = {}
    for threshold in (1.5, 2.5, 3.5):
        over_mask = total_goals_grid > threshold
        prob_over[threshold] = float(full_matrix[over_mask].sum())
        prob_under[threshold] = float(1 - prob_over[threshold])

    prob_home_scoreless = float(full_matrix[home_idx == 0].sum())
    prob_away_scoreless = float(full_matrix[away_idx == 0].sum())
    prob_both_scoreless = float(full_matrix[(home_idx == 0) & (away_idx == 0)].sum())
    prob_btts_yes = float(1 - prob_home_scoreless - prob_away_scoreless + prob_both_scoreless)
    prob_btts_no = float(1 - prob_btts_yes)

    prob_clean_sheet_home = prob_away_scoreless  # el local no recibe goles
    prob_clean_sheet_away = prob_home_scoreless  # el visitante no recibe goles

    display_size = MAX_GOALS_DISPLAY + 1
    display_matrix = pd.DataFrame(
        full_matrix[:display_size, :display_size] * 100,
        index=[f"{i}" for i in range(display_size)],
        columns=[f"{i}" for i in range(display_size)],
    )

    flat_scores = [
        (f"{h}-{a}", float(full_matrix[h, a]))
        for h in range(MAX_GOALS_INTERNAL + 1)
        for a in range(MAX_GOALS_INTERNAL + 1)
    ]
    top_scores = sorted(flat_scores, key=lambda x: x[1], reverse=True)[:5]

    confidence = _classify_confidence(home_attack.matches_used, away_attack.matches_used)
    if confidence == "baja":
        warnings.append(
            "Confianza baja: hay pocos partidos disponibles para uno o ambos equipos en su "
            "condición de local/visitante. La predicción es orientativa, no concluyente."
        )

    return PoissonPrediction(
        home_team=home_team,
        away_team=away_team,
        expected_home_goals=round(expected_home_goals, 2),
        expected_away_goals=round(expected_away_goals, 2),
        prob_home_win=round(prob_home_win * 100, 1),
        prob_draw=round(prob_draw * 100, 1),
        prob_away_win=round(prob_away_win * 100, 1),
        prob_over={k: round(v * 100, 1) for k, v in prob_over.items()},
        prob_under={k: round(v * 100, 1) for k, v in prob_under.items()},
        prob_btts_yes=round(prob_btts_yes * 100, 1),
        prob_btts_no=round(prob_btts_no * 100, 1),
        prob_clean_sheet_home=round(prob_clean_sheet_home * 100, 1),
        prob_clean_sheet_away=round(prob_clean_sheet_away * 100, 1),
        score_matrix=display_matrix,
        top_scores=[(score, round(p * 100, 1)) for score, p in top_scores],
        home_matches_used=home_attack.matches_used,
        away_matches_used=away_attack.matches_used,
        confidence=confidence,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Backtest (evaluación en un split cronológico train/test)
# --------------------------------------------------------------------------
#
# Permite comparar el desempeño de Poisson contra la regresión logística
# (src/prediction_model.py) usando exactamente el mismo split cronológico:
# las fortalezas de ataque/defensa se calculan SOLO con `train_df`, y se
# evalúan las predicciones sobre los partidos de `test_df` (que el modelo
# nunca "vio"). Esto da una base objetiva y explicable para ponderar ambos
# modelos en la predicción combinada (Fase 2, sección 9 del encargo).


@dataclass
class BacktestMetrics:
    n_test: int
    accuracy: float | None
    log_loss: float | None
    brier_score: float | None
    confusion_matrix: list[list[int]] | None
    labels: list[str]




def predict_probs_for_rows(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict[str, float]]:
    """Probabilidades H/D/A (en %) de Poisson para cada partido de
    `test_df`, usando únicamente `train_df` para estimar las fortalezas de
    ataque/defensa (sin reentrenar partido a partido). Se usa tanto para
    `backtest()` como para comparar contra el mercado en `market_odds.py`.
    """
    return [
        {
            "H": pred.prob_home_win,
            "D": pred.prob_draw,
            "A": pred.prob_away_win,
        }
        for pred in (
            predict_match(train_df, match["home_team"], match["away_team"]) for _, match in test_df.iterrows()
        )
    ]


def backtest(train_df: pd.DataFrame, test_df: pd.DataFrame) -> BacktestMetrics:
    """Evalúa el modelo de Poisson en `test_df`, usando únicamente
    `train_df` para estimar las fortalezas de ataque/defensa (sin
    reentrenar partido a partido), igual que se evalúa la regresión
    logística. No usa ninguna información del propio `test_df` para
    calcular las probabilidades."""
    if train_df.empty or test_df.empty:
        return BacktestMetrics(0, None, None, None, None, CLASS_LABELS)

    eps = 1e-15
    correct = 0
    log_loss_sum = 0.0
    brier_sum = 0.0
    confusion = {a: {p: 0 for p in CLASS_LABELS} for a in CLASS_LABELS}

    predictions = predict_probs_for_rows(train_df, test_df)

    for (_, match), probs_pct in zip(test_df.iterrows(), predictions):
        probs = {k: v / 100 for k, v in probs_pct.items()}

        if match["home_goals"] > match["away_goals"]:
            actual = "H"
        elif match["home_goals"] < match["away_goals"]:
            actual = "A"
        else:
            actual = "D"

        predicted_label = max(probs, key=probs.get)
        if predicted_label == actual:
            correct += 1
        confusion[actual][predicted_label] += 1

        p_actual = max(probs[actual], eps)
        log_loss_sum += -np.log(p_actual)
        brier_sum += sum((probs[c] - (1.0 if c == actual else 0.0)) ** 2 for c in CLASS_LABELS)

    n = len(test_df)
    matrix = [[confusion[a][p] for p in CLASS_LABELS] for a in CLASS_LABELS]

    return BacktestMetrics(
        n_test=n,
        accuracy=round(correct / n, 4),
        log_loss=round(log_loss_sum / n, 4),
        brier_score=round(brier_sum / n, 4),
        confusion_matrix=matrix,
        labels=CLASS_LABELS,
    )


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide `df` (ya ordenado cronológicamente) en train/test respetando
    el orden temporal: los partidos más recientes quedan en el test set.
    Nunca se usa una división aleatoria cuando hay fechas disponibles."""
    n_test = max(1, int(round(len(df) * test_fraction)))
    n_test = min(n_test, len(df) - 1) if len(df) > 1 else 0
    split_idx = len(df) - n_test
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)
