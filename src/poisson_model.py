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


def _league_averages(
    df: pd.DataFrame, home_col: str = "home_goals", away_col: str = "away_goals"
) -> tuple[float, float]:
    avg_home = df[home_col].mean()
    avg_away = df[away_col].mean()
    return float(avg_home), float(avg_away)


def _team_home_attack_defense(
    df: pd.DataFrame,
    team: str,
    league_avg_home: float,
    league_avg_away: float,
    home_col: str = "home_goals",
    away_col: str = "away_goals",
) -> tuple[TeamStrength, TeamStrength]:
    """Fortaleza de ataque y defensa del equipo jugando de LOCAL, para la
    estadística de conteo dada por `home_col`/`away_col` (goles por
    defecto; también se usa con tiros, córners y tarjetas)."""
    matches = team_matches(df, team, venue="home")
    n = len(matches)
    if n == 0 or league_avg_home == 0 or league_avg_away == 0:
        return (
            TeamStrength(attack=1.0, defense=1.0, matches_used=n, used_fallback=True),
            TeamStrength(attack=1.0, defense=1.0, matches_used=n, used_fallback=True),
        )
    avg_scored = matches[home_col].mean()
    avg_conceded = matches[away_col].mean()
    attack = TeamStrength(avg_scored / league_avg_home, 0.0, n, False)
    defense = TeamStrength(0.0, avg_conceded / league_avg_away, n, False)
    return attack, defense


def _team_away_attack_defense(
    df: pd.DataFrame,
    team: str,
    league_avg_home: float,
    league_avg_away: float,
    home_col: str = "home_goals",
    away_col: str = "away_goals",
) -> tuple[TeamStrength, TeamStrength]:
    """Fortaleza de ataque y defensa del equipo jugando de VISITANTE, para
    la estadística de conteo dada por `home_col`/`away_col`."""
    matches = team_matches(df, team, venue="away")
    n = len(matches)
    if n == 0 or league_avg_home == 0 or league_avg_away == 0:
        return (
            TeamStrength(attack=1.0, defense=1.0, matches_used=n, used_fallback=True),
            TeamStrength(attack=1.0, defense=1.0, matches_used=n, used_fallback=True),
        )
    avg_scored = matches[away_col].mean()
    avg_conceded = matches[home_col].mean()
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
# Mercados de tiros, tiros a puerta, córners y tarjetas
# --------------------------------------------------------------------------
#
# Reutiliza el mismo motor de Poisson (fuerza de ataque/defensa relativa a
# la liga) que se usa para goles: tiros, tiros a puerta, córners y tarjetas
# son igual de "conteos de eventos por partido", así que el mismo modelo
# aplica sin duplicar lógica. Solo se calcula el total combinado
# (local + visitante) para mercados de over/under, aprovechando que la
# suma de dos variables Poisson independientes es a su vez Poisson con la
# tasa combinada — no hace falta una matriz conjunta como con los goles.

COUNT_STAT_CONFIG: dict[str, dict[str, str]] = {
    "shots": {"home_col": "home_shots", "away_col": "away_shots", "label": "Tiros"},
    "shots_on_target": {"home_col": "home_shots_target", "away_col": "away_shots_target", "label": "Tiros a puerta"},
    "corners": {"home_col": "home_corners", "away_col": "away_corners", "label": "Córners"},
    "yellow_cards": {"home_col": "home_yellow", "away_col": "away_yellow", "label": "Tarjetas amarillas"},
}

# Los conteos esperados se acotan a un rango amplio pero plausible, igual
# que con los goles, para evitar estimaciones degeneradas con muestras muy
# chicas.
EXPECTED_COUNT_MIN = 0.2
EXPECTED_COUNT_MAX = 30.0


@dataclass
class CountStatPrediction:
    stat_key: str
    label: str
    home_team: str
    away_team: str
    expected_home: float
    expected_away: float
    expected_total: float
    thresholds: list[float]
    prob_over: dict[float, float]
    prob_under: dict[float, float]
    most_likely_total: int
    most_likely_total_prob: float
    home_matches_used: int
    away_matches_used: int
    confidence: str
    warnings: list[str] = field(default_factory=list)


def _derive_thresholds(expected_total: float, n: int = 3) -> list[float]:
    """Líneas de over/under centradas en el promedio real del archivo (no
    fijas), para que se adapten solas a la liga cargada (una segunda
    división con menos córners por partido que una primera, por ejemplo)."""
    center = round(expected_total)
    start = max(0.5, center - 1.5)
    return [round(start + i, 1) for i in range(n)]


def predict_count_stat(df: pd.DataFrame, home_team: str, away_team: str, stat_key: str) -> CountStatPrediction | None:
    """Predicción de over/under para una estadística de conteo (tiros,
    tiros a puerta, córners o tarjetas amarillas). Devuelve `None` cuando
    el archivo no trae esas columnas o no tienen datos numéricos
    utilizables — nunca se inventa un valor."""
    config = COUNT_STAT_CONFIG[stat_key]
    home_col, away_col = config["home_col"], config["away_col"]

    if home_col not in df.columns or away_col not in df.columns:
        return None
    if df[home_col].notna().sum() == 0 or df[away_col].notna().sum() == 0:
        return None

    league_avg_home, league_avg_away = _league_averages(df, home_col, away_col)
    if pd.isna(league_avg_home) or pd.isna(league_avg_away) or league_avg_home == 0 or league_avg_away == 0:
        return None

    warnings: list[str] = []
    home_attack, home_defense = _team_home_attack_defense(
        df, home_team, league_avg_home, league_avg_away, home_col, away_col
    )
    away_attack, away_defense = _team_away_attack_defense(
        df, away_team, league_avg_home, league_avg_away, home_col, away_col
    )

    if home_attack.used_fallback:
        warnings.append(
            f"'{home_team}' no tiene partidos con datos de {config['label'].lower()} como local; "
            "se usó el promedio general de la liga."
        )
    if away_attack.used_fallback:
        warnings.append(
            f"'{away_team}' no tiene partidos con datos de {config['label'].lower()} como visitante; "
            "se usó el promedio general de la liga."
        )

    expected_home = home_attack.attack * away_defense.defense * league_avg_home
    expected_away = away_attack.attack * home_defense.defense * league_avg_away
    expected_home = float(np.clip(expected_home, EXPECTED_COUNT_MIN, EXPECTED_COUNT_MAX))
    expected_away = float(np.clip(expected_away, EXPECTED_COUNT_MIN, EXPECTED_COUNT_MAX))
    expected_total = expected_home + expected_away

    # Suma de dos Poisson independientes = Poisson(lambda_home + lambda_away).
    max_count = int(expected_total + 6 * (expected_total ** 0.5) + 10)
    counts = np.arange(0, max_count + 1)
    total_probs = poisson.pmf(counts, expected_total)

    thresholds = _derive_thresholds(expected_total)
    prob_over: dict[float, float] = {}
    prob_under: dict[float, float] = {}
    for t in thresholds:
        p_over = float(total_probs[counts > t].sum())
        prob_over[t] = round(p_over * 100, 1)
        prob_under[t] = round((1 - p_over) * 100, 1)

    most_likely_idx = int(np.argmax(total_probs))

    confidence = _classify_confidence(home_attack.matches_used, away_attack.matches_used)

    return CountStatPrediction(
        stat_key=stat_key,
        label=config["label"],
        home_team=home_team,
        away_team=away_team,
        expected_home=round(expected_home, 2),
        expected_away=round(expected_away, 2),
        expected_total=round(expected_total, 2),
        thresholds=thresholds,
        prob_over=prob_over,
        prob_under=prob_under,
        most_likely_total=int(counts[most_likely_idx]),
        most_likely_total_prob=round(float(total_probs[most_likely_idx]) * 100, 1),
        home_matches_used=home_attack.matches_used,
        away_matches_used=away_attack.matches_used,
        confidence=confidence,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Medio tiempo: primer tiempo, segunda mitad y mercado HT/FT
# --------------------------------------------------------------------------
#
# Usa las columnas de marcador al descanso (HTHG/HTAG en football-data.co.uk,
# mapeadas a home_goals_ht/away_goals_ht) con el mismo motor de Poisson.
#
# Para la segunda mitad y el mercado HT/FT se modelan el primer y el segundo
# tiempo como dos procesos de Poisson INDEPENDIENTES (goles de la 2ª mitad =
# goles totales − goles del descanso). Es el supuesto estándar: en un proceso
# de Poisson, los eventos en intervalos disjuntos son independientes entre sí.

HT_MAX_GOALS_INTERNAL = 8
HT_MAX_GOALS_DISPLAY = 4  # matriz de marcadores al descanso (0-4 x 0-4)
HT_THRESHOLDS = (0.5, 1.5, 2.5)  # las líneas típicas del primer tiempo
EXPECTED_HT_GOALS_MIN = 0.05
EXPECTED_HT_GOALS_MAX = 4.0
EXPECTED_SECOND_HALF_MIN = 0.05

# Rango por equipo y por mitad usado para construir la matriz HT/FT.
HT_FT_RANGE = 7


@dataclass
class HalfTimePrediction:
    home_team: str
    away_team: str
    n_matches_with_ht: int
    expected_home_ht: float
    expected_away_ht: float
    expected_total_ht: float
    expected_home_2h: float
    expected_away_2h: float
    expected_total_2h: float
    prob_home_lead_ht: float
    prob_draw_ht: float
    prob_away_lead_ht: float
    prob_over: dict[float, float]
    prob_under: dict[float, float]
    prob_btts_ht_yes: float
    prob_btts_ht_no: float
    prob_scoreless_ht: float
    prob_more_goals_first_half: float
    prob_more_goals_second_half: float
    prob_equal_halves: float
    score_matrix_ht: pd.DataFrame
    top_scores_ht: list[tuple[str, float]]
    ht_ft_matrix: pd.DataFrame  # filas = resultado al descanso, columnas = resultado final (%)
    home_matches_used: int
    away_matches_used: int
    confidence: str
    warnings: list[str] = field(default_factory=list)


def _ht_ft_probability_matrix(
    expected_home_ht: float,
    expected_away_ht: float,
    expected_home_2h: float,
    expected_away_2h: float,
) -> np.ndarray:
    """Matriz 3x3 de probabilidades combinadas (resultado al descanso ×
    resultado final), tratando cada mitad como un Poisson independiente."""
    rng = np.arange(HT_FT_RANGE)
    p_home_1h = poisson.pmf(rng, expected_home_ht)
    p_away_1h = poisson.pmf(rng, expected_away_ht)
    p_home_2h = poisson.pmf(rng, expected_home_2h)
    p_away_2h = poisson.pmf(rng, expected_away_2h)

    matrix = np.zeros((3, 3))
    for h1 in rng:
        for a1 in rng:
            p_first = p_home_1h[h1] * p_away_1h[a1]
            if p_first < 1e-12:
                continue
            ht_idx = 0 if h1 > a1 else (1 if h1 == a1 else 2)
            for h2 in rng:
                for a2 in rng:
                    p_second = p_home_2h[h2] * p_away_2h[a2]
                    if p_second < 1e-12:
                        continue
                    full_home, full_away = h1 + h2, a1 + a2
                    ft_idx = 0 if full_home > full_away else (1 if full_home == full_away else 2)
                    matrix[ht_idx, ft_idx] += p_first * p_second

    total = matrix.sum()
    return matrix / total if total > 0 else matrix


def predict_half_time(df: pd.DataFrame, home_team: str, away_team: str) -> HalfTimePrediction | None:
    """Predicción del primer tiempo, la segunda mitad y el mercado HT/FT.

    Devuelve `None` si el archivo no trae marcador de medio tiempo o no hay
    ninguna fila con ese dato completo — nunca se inventa un valor.

    Todo se calcula sobre el subconjunto de partidos que SÍ tienen marcador
    al descanso, para que los goles de la segunda mitad (total − descanso)
    sean coherentes entre sí.
    """
    home_col, away_col = "home_goals_ht", "away_goals_ht"
    if home_col not in df.columns or away_col not in df.columns:
        return None
    if "home_goals" not in df.columns or "away_goals" not in df.columns:
        return None

    valid = df[[home_col, away_col, "home_goals", "away_goals"]].notna().all(axis=1)
    ht_df = df[valid]
    if ht_df.empty:
        return None

    warnings: list[str] = []
    if len(ht_df) < len(df):
        warnings.append(
            f"Se usaron {len(ht_df)} de {len(df)} partidos: el resto no trae marcador de medio tiempo."
        )

    league_avg_home_ht, league_avg_away_ht = _league_averages(ht_df, home_col, away_col)
    league_avg_home_ft, league_avg_away_ft = _league_averages(ht_df, "home_goals", "away_goals")
    if (
        pd.isna(league_avg_home_ht)
        or pd.isna(league_avg_away_ht)
        or league_avg_home_ht == 0
        or league_avg_away_ht == 0
    ):
        return None

    ht_home_attack, ht_home_defense = _team_home_attack_defense(
        ht_df, home_team, league_avg_home_ht, league_avg_away_ht, home_col, away_col
    )
    ht_away_attack, ht_away_defense = _team_away_attack_defense(
        ht_df, away_team, league_avg_home_ht, league_avg_away_ht, home_col, away_col
    )
    ft_home_attack, ft_home_defense = _team_home_attack_defense(
        ht_df, home_team, league_avg_home_ft, league_avg_away_ft
    )
    ft_away_attack, ft_away_defense = _team_away_attack_defense(
        ht_df, away_team, league_avg_home_ft, league_avg_away_ft
    )

    if ht_home_attack.used_fallback:
        warnings.append(
            f"'{home_team}' no tiene partidos como local con marcador de medio tiempo; "
            "se usó el promedio general de la liga."
        )
    if ht_away_attack.used_fallback:
        warnings.append(
            f"'{away_team}' no tiene partidos como visitante con marcador de medio tiempo; "
            "se usó el promedio general de la liga."
        )

    expected_home_ht = float(
        np.clip(
            ht_home_attack.attack * ht_away_defense.defense * league_avg_home_ht,
            EXPECTED_HT_GOALS_MIN,
            EXPECTED_HT_GOALS_MAX,
        )
    )
    expected_away_ht = float(
        np.clip(
            ht_away_attack.attack * ht_home_defense.defense * league_avg_away_ht,
            EXPECTED_HT_GOALS_MIN,
            EXPECTED_HT_GOALS_MAX,
        )
    )
    expected_home_ft = float(
        np.clip(
            ft_home_attack.attack * ft_away_defense.defense * league_avg_home_ft,
            EXPECTED_GOALS_MIN,
            EXPECTED_GOALS_MAX,
        )
    )
    expected_away_ft = float(
        np.clip(
            ft_away_attack.attack * ft_home_defense.defense * league_avg_away_ft,
            EXPECTED_GOALS_MIN,
            EXPECTED_GOALS_MAX,
        )
    )

    expected_home_2h = expected_home_ft - expected_home_ht
    expected_away_2h = expected_away_ft - expected_away_ht
    if expected_home_2h < EXPECTED_SECOND_HALF_MIN or expected_away_2h < EXPECTED_SECOND_HALF_MIN:
        warnings.append(
            "La estimación del primer tiempo quedó tan alta como la del partido completo para "
            "algún equipo (muestra chica o datos irregulares): la segunda mitad se acotó a un "
            "mínimo y debe tomarse con cautela."
        )
    expected_home_2h = max(EXPECTED_SECOND_HALF_MIN, expected_home_2h)
    expected_away_2h = max(EXPECTED_SECOND_HALF_MIN, expected_away_2h)

    # --- Mercados del primer tiempo, sobre la matriz de marcadores al descanso ---
    ht_matrix = _score_probability_matrix(expected_home_ht, expected_away_ht, HT_MAX_GOALS_INTERNAL)
    idx = np.arange(HT_MAX_GOALS_INTERNAL + 1)
    home_idx, away_idx = np.meshgrid(idx, idx, indexing="ij")

    prob_home_lead = float(ht_matrix[home_idx > away_idx].sum())
    prob_draw_ht = float(ht_matrix[home_idx == away_idx].sum())
    prob_away_lead = float(ht_matrix[home_idx < away_idx].sum())

    totals = home_idx + away_idx
    prob_over: dict[float, float] = {}
    prob_under: dict[float, float] = {}
    for threshold in HT_THRESHOLDS:
        p_over = float(ht_matrix[totals > threshold].sum())
        prob_over[threshold] = round(p_over * 100, 1)
        prob_under[threshold] = round((1 - p_over) * 100, 1)

    prob_home_scoreless = float(ht_matrix[home_idx == 0].sum())
    prob_away_scoreless = float(ht_matrix[away_idx == 0].sum())
    prob_scoreless_ht = float(ht_matrix[0, 0])
    prob_btts_ht_yes = float(1 - prob_home_scoreless - prob_away_scoreless + prob_scoreless_ht)

    display_size = HT_MAX_GOALS_DISPLAY + 1
    score_matrix_ht = pd.DataFrame(
        ht_matrix[:display_size, :display_size] * 100,
        index=[str(i) for i in range(display_size)],
        columns=[str(i) for i in range(display_size)],
    )
    flat_scores = [
        (f"{h}-{a}", float(ht_matrix[h, a]))
        for h in range(HT_MAX_GOALS_INTERNAL + 1)
        for a in range(HT_MAX_GOALS_INTERNAL + 1)
    ]
    top_scores_ht = [
        (score, round(p * 100, 1)) for score, p in sorted(flat_scores, key=lambda x: x[1], reverse=True)[:5]
    ]

    # --- ¿Qué mitad tendrá más goles? (suma de dos Poisson por mitad) ---
    lambda_first = expected_home_ht + expected_away_ht
    lambda_second = expected_home_2h + expected_away_2h
    half_counts = np.arange(0, 16)
    p_first_half = poisson.pmf(half_counts, lambda_first)
    p_second_half = poisson.pmf(half_counts, lambda_second)
    halves_joint = np.outer(p_first_half, p_second_half)
    first_idx, second_idx = np.meshgrid(half_counts, half_counts, indexing="ij")
    prob_more_first = float(halves_joint[first_idx > second_idx].sum())
    prob_more_second = float(halves_joint[first_idx < second_idx].sum())
    prob_equal_halves = float(halves_joint[first_idx == second_idx].sum())

    # --- Mercado combinado HT/FT ---
    ht_ft = _ht_ft_probability_matrix(expected_home_ht, expected_away_ht, expected_home_2h, expected_away_2h)
    ht_ft_matrix = pd.DataFrame(
        ht_ft * 100,
        index=[f"HT: {home_team}", "HT: Empate", f"HT: {away_team}"],
        columns=[f"FT: {home_team}", "FT: Empate", f"FT: {away_team}"],
    )

    confidence = _classify_confidence(ht_home_attack.matches_used, ht_away_attack.matches_used)
    if confidence == "baja":
        warnings.append(
            "Confianza baja para el medio tiempo: pocos partidos con marcador al descanso para "
            "uno o ambos equipos."
        )

    return HalfTimePrediction(
        home_team=home_team,
        away_team=away_team,
        n_matches_with_ht=len(ht_df),
        expected_home_ht=round(expected_home_ht, 2),
        expected_away_ht=round(expected_away_ht, 2),
        expected_total_ht=round(expected_home_ht + expected_away_ht, 2),
        expected_home_2h=round(expected_home_2h, 2),
        expected_away_2h=round(expected_away_2h, 2),
        expected_total_2h=round(expected_home_2h + expected_away_2h, 2),
        prob_home_lead_ht=round(prob_home_lead * 100, 1),
        prob_draw_ht=round(prob_draw_ht * 100, 1),
        prob_away_lead_ht=round(prob_away_lead * 100, 1),
        prob_over=prob_over,
        prob_under=prob_under,
        prob_btts_ht_yes=round(prob_btts_ht_yes * 100, 1),
        prob_btts_ht_no=round((1 - prob_btts_ht_yes) * 100, 1),
        prob_scoreless_ht=round(prob_scoreless_ht * 100, 1),
        prob_more_goals_first_half=round(prob_more_first * 100, 1),
        prob_more_goals_second_half=round(prob_more_second * 100, 1),
        prob_equal_halves=round(prob_equal_halves * 100, 1),
        score_matrix_ht=score_matrix_ht,
        top_scores_ht=top_scores_ht,
        ht_ft_matrix=ht_ft_matrix,
        home_matches_used=ht_home_attack.matches_used,
        away_matches_used=ht_away_attack.matches_used,
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
