"""Análisis estadístico de equipos a partir de los partidos históricos.

Todas las funciones reciben el dataframe histórico ya limpio y mapeado a
nombres canónicos (ver `data_cleaner.py` / `column_mapper.py`) y devuelven
`None` (o la etiqueta "Datos insuficientes" al formatear) cuando no hay
datos suficientes, en vez de inventar valores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.utils import MIN_MATCHES_RELIABLE, safe_divide, safe_pct

Venue = Literal["all", "home", "away"]

GOAL_THRESHOLDS = (1.5, 2.5, 3.5)


# --------------------------------------------------------------------------
# Utilidades de selección / validación de equipos
# --------------------------------------------------------------------------


def get_team_list(df: pd.DataFrame) -> list[str]:
    """Lista ordenada de todos los equipos detectados (local + visitante)."""
    if df.empty or "home_team" not in df.columns or "away_team" not in df.columns:
        return []
    teams = set(df["home_team"].dropna().unique()) | set(df["away_team"].dropna().unique())
    return sorted(teams)


def validate_team_selection(df: pd.DataFrame, home_team: str, away_team: str) -> list[str]:
    """Valida que ambos equipos existan en el dataset y sean distintos.
    Devuelve una lista de mensajes de error (vacía si todo es válido).
    """
    errors = []
    teams = set(get_team_list(df))
    if home_team == away_team:
        errors.append("El equipo local y el visitante deben ser diferentes.")
    if home_team not in teams:
        errors.append(f"El equipo '{home_team}' no existe en los datos cargados.")
    if away_team not in teams:
        errors.append(f"El equipo '{away_team}' no existe en los datos cargados.")
    return errors


def team_matches(df: pd.DataFrame, team: str, venue: Venue = "all") -> pd.DataFrame:
    """Filtra los partidos en los que participó `team`, según el rol."""
    if venue == "home":
        return df[df["home_team"] == team]
    if venue == "away":
        return df[df["away_team"] == team]
    return df[(df["home_team"] == team) | (df["away_team"] == team)]


# --------------------------------------------------------------------------
# Récord (W/D/L) y goles
# --------------------------------------------------------------------------


def _match_outcome(row: pd.Series, team: str) -> str:
    """Devuelve 'W', 'D' o 'L' desde la perspectiva de `team`, calculado a
    partir de los goles (no del texto de la columna 'result', para ser
    robustos aunque esa columna no exista o tenga formato distinto)."""
    is_home = row["home_team"] == team
    gf = row["home_goals"] if is_home else row["away_goals"]
    ga = row["away_goals"] if is_home else row["home_goals"]
    if gf > ga:
        return "W"
    if gf < ga:
        return "L"
    return "D"


@dataclass
class TeamRecord:
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    win_pct: float | None = None
    draw_pct: float | None = None
    loss_pct: float | None = None
    goals_for: int = 0
    goals_against: int = 0
    avg_goals_for: float | None = None
    avg_goals_against: float | None = None
    goal_difference: int = 0
    clean_sheets: int = 0
    clean_sheet_pct: float | None = None
    failed_to_score: int = 0
    failed_to_score_pct: float | None = None
    both_teams_scored_pct: float | None = None
    over_pct: dict[float, float | None] = field(default_factory=dict)


def compute_team_record(df: pd.DataFrame, team: str, venue: Venue = "all") -> TeamRecord:
    """Calcula el récord y las estadísticas de goles de un equipo."""
    matches = team_matches(df, team, venue)
    record = TeamRecord()
    record.played = len(matches)

    if record.played == 0:
        return record

    gf_list = []
    ga_list = []
    wins = draws = losses = 0
    clean_sheets = failed_to_score = both_scored = 0
    over_counts = {t: 0 for t in GOAL_THRESHOLDS}

    for _, row in matches.iterrows():
        is_home = row["home_team"] == team
        gf = row["home_goals"] if is_home else row["away_goals"]
        ga = row["away_goals"] if is_home else row["home_goals"]
        gf_list.append(gf)
        ga_list.append(ga)

        outcome = _match_outcome(row, team)
        if outcome == "W":
            wins += 1
        elif outcome == "D":
            draws += 1
        else:
            losses += 1

        if ga == 0:
            clean_sheets += 1
        if gf == 0:
            failed_to_score += 1
        if gf > 0 and ga > 0:
            both_scored += 1

        total_goals = gf + ga
        for threshold in GOAL_THRESHOLDS:
            if total_goals > threshold:
                over_counts[threshold] += 1

    record.wins, record.draws, record.losses = wins, draws, losses
    record.win_pct = safe_pct(wins, record.played)
    record.draw_pct = safe_pct(draws, record.played)
    record.loss_pct = safe_pct(losses, record.played)

    record.goals_for = int(sum(gf_list))
    record.goals_against = int(sum(ga_list))
    record.avg_goals_for = safe_divide(record.goals_for, record.played)
    record.avg_goals_against = safe_divide(record.goals_against, record.played)
    record.goal_difference = record.goals_for - record.goals_against

    record.clean_sheets = clean_sheets
    record.clean_sheet_pct = safe_pct(clean_sheets, record.played)
    record.failed_to_score = failed_to_score
    record.failed_to_score_pct = safe_pct(failed_to_score, record.played)
    record.both_teams_scored_pct = safe_pct(both_scored, record.played)
    record.over_pct = {t: safe_pct(over_counts[t], record.played) for t in GOAL_THRESHOLDS}

    if record.avg_goals_for is not None:
        record.avg_goals_for = round(record.avg_goals_for, 2)
    if record.avg_goals_against is not None:
        record.avg_goals_against = round(record.avg_goals_against, 2)

    return record


# --------------------------------------------------------------------------
# Estadísticas complementarias (tiros, córners, tarjetas, posesión)
# --------------------------------------------------------------------------

# Cada entrada: campo_home, campo_away -> se promedia el valor correspondiente
# al rol que jugó el equipo en cada partido.
_EXTRA_STAT_FIELDS = {
    "shots": ("home_shots", "away_shots"),
    "shots_on_target": ("home_shots_target", "away_shots_target"),
    "corners": ("home_corners", "away_corners"),
    "fouls": ("home_fouls", "away_fouls"),
    "yellow_cards": ("home_yellow", "away_yellow"),
    "red_cards": ("home_red", "away_red"),
    "possession": ("home_possession", "away_possession"),
    "xg": ("home_xg", "away_xg"),
}


def compute_extra_averages(df: pd.DataFrame, team: str, venue: Venue = "all") -> dict[str, float | None]:
    """Promedios de tiros/córners/tarjetas/posesión/xG para un equipo, o
    None por métrica cuando la columna no existe en el archivo cargado.
    """
    matches = team_matches(df, team, venue)
    result: dict[str, float | None] = {}

    for stat_name, (home_col, away_col) in _EXTRA_STAT_FIELDS.items():
        if home_col not in df.columns or away_col not in df.columns or matches.empty:
            result[stat_name] = None
            continue
        values = matches.apply(
            lambda row: row[home_col] if row["home_team"] == team else row[away_col], axis=1
        )
        values = values.dropna()
        result[stat_name] = round(values.mean(), 2) if not values.empty else None

    return result


def stat_columns_available(df: pd.DataFrame) -> dict[str, bool]:
    """Indica, por estadística extra, si la columna existe en el archivo
    (independientemente de que tenga valores). Útil para mostrar
    'xG no disponible' en vez de ocultar la fila silenciosamente."""
    return {
        stat_name: (home_col in df.columns and away_col in df.columns)
        for stat_name, (home_col, away_col) in _EXTRA_STAT_FIELDS.items()
    }


# --------------------------------------------------------------------------
# Forma reciente
# --------------------------------------------------------------------------


@dataclass
class RecentForm:
    n_requested: int
    n_available: int
    results: list[str] = field(default_factory=list)  # más reciente primero, ej. ["W","D","L"]
    points: int = 0
    points_per_game: float | None = None
    avg_goals_for: float | None = None
    avg_goals_against: float | None = None
    dates: list = field(default_factory=list)


def compute_recent_form(df: pd.DataFrame, team: str, n: int = 5) -> RecentForm:
    """Calcula la forma en los últimos `n` partidos (cualquier condición),
    ordenados del más reciente al más antiguo. Requiere que `df` venga
    ordenado cronológicamente (lo garantiza `data_cleaner.clean_data`)."""
    matches = team_matches(df, team, venue="all")
    if "date" in matches.columns:
        matches = matches.sort_values("date")
    recent = matches.tail(n).iloc[::-1]  # más reciente primero

    form = RecentForm(n_requested=n, n_available=len(recent))
    if recent.empty:
        return form

    points = 0
    gf_total = ga_total = 0
    for _, row in recent.iterrows():
        outcome = _match_outcome(row, team)
        form.results.append(outcome)
        points += {"W": 3, "D": 1, "L": 0}[outcome]
        is_home = row["home_team"] == team
        gf_total += row["home_goals"] if is_home else row["away_goals"]
        ga_total += row["away_goals"] if is_home else row["home_goals"]
        form.dates.append(row.get("date"))

    form.points = points
    form.points_per_game = round(safe_divide(points, form.n_available, default=0.0), 2)
    form.avg_goals_for = round(safe_divide(gf_total, form.n_available, default=0.0), 2)
    form.avg_goals_against = round(safe_divide(ga_total, form.n_available, default=0.0), 2)
    return form


# --------------------------------------------------------------------------
# Enfrentamientos directos (head-to-head)
# --------------------------------------------------------------------------


def head_to_head(df: pd.DataFrame, team_a: str, team_b: str) -> pd.DataFrame:
    """Partidos previos entre dos equipos específicos, en cualquier
    condición, tal como existan en el archivo cargado. Si no hay
    enfrentamientos previos en el archivo, devuelve un dataframe vacío
    (no se buscan datos externos)."""
    mask = ((df["home_team"] == team_a) & (df["away_team"] == team_b)) | (
        (df["home_team"] == team_b) & (df["away_team"] == team_a)
    )
    cols = [c for c in ["date", "home_team", "away_team", "home_goals", "away_goals"] if c in df.columns]
    return df[mask][cols].copy()


# --------------------------------------------------------------------------
# Perfil completo de un equipo (para tarjetas de comparación)
# --------------------------------------------------------------------------


@dataclass
class TeamProfile:
    team: str
    overall: TeamRecord
    as_home: TeamRecord
    as_away: TeamRecord
    recent_form: RecentForm
    extra_overall: dict[str, float | None]


def build_team_profile(df: pd.DataFrame, team: str, recent_n: int = 5) -> TeamProfile:
    """Arma el perfil completo de un equipo combinando récord general,
    como local, como visitante, forma reciente y estadísticas extra."""
    return TeamProfile(
        team=team,
        overall=compute_team_record(df, team, "all"),
        as_home=compute_team_record(df, team, "home"),
        as_away=compute_team_record(df, team, "away"),
        recent_form=compute_recent_form(df, team, recent_n),
        extra_overall=compute_extra_averages(df, team, "all"),
    )


# --------------------------------------------------------------------------
# Resumen general del dataset
# --------------------------------------------------------------------------


@dataclass
class DatasetSummary:
    n_matches: int
    n_teams: int
    period_start: object
    period_end: object
    avg_goals_per_match: float | None
    home_win_pct: float | None
    draw_pct: float | None
    away_win_pct: float | None
    sufficient_data: bool


def compute_dataset_summary(df: pd.DataFrame) -> DatasetSummary:
    n_matches = len(df)
    teams = get_team_list(df)

    if n_matches == 0:
        return DatasetSummary(0, len(teams), None, None, None, None, None, None, False)

    total_goals = (df["home_goals"] + df["away_goals"]).sum()
    home_wins = (df["home_goals"] > df["away_goals"]).sum()
    draws = (df["home_goals"] == df["away_goals"]).sum()
    away_wins = (df["home_goals"] < df["away_goals"]).sum()

    period_start = df["date"].min() if "date" in df.columns and df["date"].notna().any() else None
    period_end = df["date"].max() if "date" in df.columns and df["date"].notna().any() else None

    return DatasetSummary(
        n_matches=n_matches,
        n_teams=len(teams),
        period_start=period_start,
        period_end=period_end,
        avg_goals_per_match=round(safe_divide(total_goals, n_matches, default=0.0), 2),
        home_win_pct=safe_pct(home_wins, n_matches),
        draw_pct=safe_pct(draws, n_matches),
        away_win_pct=safe_pct(away_wins, n_matches),
        sufficient_data=n_matches >= MIN_MATCHES_RELIABLE,
    )
