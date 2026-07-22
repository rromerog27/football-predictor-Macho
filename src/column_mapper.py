"""Reconocimiento y mapeo de columnas a nombres canónicos.

El caso principal soportado es el formato football-data.co.uk (el del
archivo de referencia del usuario: Div, Date, Time, HomeTeam, AwayTeam,
FTHG, FTAG, FTR, HTHG, HTAG, HTR, Referee, HS, AS, HST, AST, HF, AF, HC,
AC, HY, AY, HR, AR). Los nombres alternativos (español, snake_case, etc.)
son un respaldo para otros archivos que el usuario pueda cargar más
adelante.

Todo el resto del proyecto (limpieza, estadísticas, modelos) trabaja sobre
los nombres CANÓNICOS definidos en `CANONICAL_FIELDS`, nunca sobre los
nombres originales del archivo.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

REQUIRED_FIELDS = ["home_team", "away_team", "home_goals", "away_goals"]

# Campos opcionales: si no se encuentran, el resto del pipeline debe seguir
# funcionando y mostrar "xG no disponible" / "Datos insuficientes" según
# corresponda, nunca bloquear el análisis.
OPTIONAL_FIELDS = [
    "competition",
    "season",
    "date",
    "time",
    "result",
    "home_goals_ht",
    "away_goals_ht",
    "result_ht",
    "referee",
    "home_shots",
    "away_shots",
    "home_shots_target",
    "away_shots_target",
    "home_fouls",
    "away_fouls",
    "home_corners",
    "away_corners",
    "home_yellow",
    "away_yellow",
    "home_red",
    "away_red",
    "home_possession",
    "away_possession",
    "home_xg",
    "away_xg",
    "market_home_odds",
    "market_draw_odds",
    "market_away_odds",
]

ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

# Nombre legible en español para cada campo canónico (para menús desplegables
# de selección manual en el dashboard).
FIELD_LABELS = {
    "competition": "Competición / Liga",
    "season": "Temporada",
    "date": "Fecha",
    "time": "Hora",
    "home_team": "Equipo local",
    "away_team": "Equipo visitante",
    "home_goals": "Goles del local",
    "away_goals": "Goles del visitante",
    "result": "Resultado final (H/D/A)",
    "home_goals_ht": "Goles local (medio tiempo)",
    "away_goals_ht": "Goles visitante (medio tiempo)",
    "result_ht": "Resultado medio tiempo",
    "referee": "Árbitro",
    "home_shots": "Tiros local",
    "away_shots": "Tiros visitante",
    "home_shots_target": "Tiros a puerta local",
    "away_shots_target": "Tiros a puerta visitante",
    "home_fouls": "Faltas local",
    "away_fouls": "Faltas visitante",
    "home_corners": "Córners local",
    "away_corners": "Córners visitante",
    "home_yellow": "Tarjetas amarillas local",
    "away_yellow": "Tarjetas amarillas visitante",
    "home_red": "Tarjetas rojas local",
    "away_red": "Tarjetas rojas visitante",
    "home_possession": "Posesión local",
    "away_possession": "Posesión visitante",
    "home_xg": "xG local",
    "away_xg": "xG visitante",
    "market_home_odds": "Cuota de mercado — Victoria local",
    "market_draw_odds": "Cuota de mercado — Empate",
    "market_away_odds": "Cuota de mercado — Victoria visitante",
}

# Candidatos por campo. El primer grupo (football-data.co.uk) es el caso
# principal; el resto son alternativas de respaldo.
CANDIDATE_NAMES: dict[str, list[str]] = {
    "competition": ["Div", "League", "Liga", "Competition", "Competición", "Comp"],
    "season": ["Season", "Temporada", "Season_ID"],
    "date": ["Date", "Fecha", "MatchDate"],
    "time": ["Time", "Hora", "MatchTime"],
    "home_team": ["HomeTeam", "Local", "Equipo Local", "EquipoLocal", "home_team"],
    "away_team": ["AwayTeam", "Visitante", "Equipo Visitante", "EquipoVisitante", "away_team"],
    "home_goals": ["FTHG", "HomeGoals", "Goles Local", "GolesLocal", "home_goals"],
    "away_goals": ["FTAG", "AwayGoals", "Goles Visitante", "GolesVisitante", "away_goals"],
    "result": ["FTR", "Result", "Resultado"],
    "home_goals_ht": ["HTHG", "HomeGoalsHT", "GolesLocalMedioTiempo"],
    "away_goals_ht": ["HTAG", "AwayGoalsHT", "GolesVisitanteMedioTiempo"],
    "result_ht": ["HTR", "ResultHT", "ResultadoMedioTiempo"],
    "referee": ["Referee", "Árbitro", "Arbitro"],
    "home_shots": ["HS", "HomeShots", "TirosLocal"],
    "away_shots": ["AS", "AwayShots", "TirosVisitante"],
    "home_shots_target": ["HST", "HomeShotsOnTarget", "TirosPuertaLocal"],
    "away_shots_target": ["AST", "AwayShotsOnTarget", "TirosPuertaVisitante"],
    "home_fouls": ["HF", "HomeFouls", "FaltasLocal"],
    "away_fouls": ["AF", "AwayFouls", "FaltasVisitante"],
    "home_corners": ["HC", "HomeCorners", "CornersLocal", "CornerLocal"],
    "away_corners": ["AC", "AwayCorners", "CornersVisitante", "CornerVisitante"],
    "home_yellow": ["HY", "HomeYellow", "TarjetasAmarillasLocal"],
    "away_yellow": ["AY", "AwayYellow", "TarjetasAmarillasVisitante"],
    "home_red": ["HR", "HomeRed", "TarjetasRojasLocal"],
    "away_red": ["AR", "AwayRed", "TarjetasRojasVisitante"],
    "home_possession": ["HomePossession", "PosesionLocal", "Possession_Home", "PossH"],
    "away_possession": ["AwayPossession", "PosesionVisitante", "Possession_Away", "PossA"],
    "home_xg": ["HomeXG", "xG_Home", "HxG", "xGHome", "Home_xG"],
    "away_xg": ["AwayXG", "xG_Away", "AxG", "xGAway", "Away_xG"],
    # Prioridad: cuota de cierre promedio (más representativa del mercado)
    # > cuota de apertura promedio > Bet365 de cierre/apertura > Pinnacle
    # de cierre/apertura, para archivos que no traen columnas "Avg".
    "market_home_odds": ["AvgCH", "AvgH", "B365CH", "B365H", "PSCH", "PSH"],
    "market_draw_odds": ["AvgCD", "AvgD", "B365CD", "B365D", "PSCD", "PSD"],
    "market_away_odds": ["AvgCA", "AvgA", "B365CA", "B365A", "PSCA", "PSA"],
}

# Columnas football-data.co.uk que se conservan en el dataframe aunque no
# se usen todavía en ninguna fase (HTHG/HTAG/HTR/Referee), tal como pide
# el usuario, mapeándolas a sus campos canónicos igual que cualquier otra.


def _normalize(name: str) -> str:
    """Normaliza un nombre de columna para comparación difusa: sin acentos,
    minúsculas, sin espacios ni guiones bajos.
    """
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[\s_\-]+", "", ascii_only).lower()


def auto_map_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Intenta mapear automáticamente cada campo canónico a una columna
    real del dataframe. Devuelve {campo_canonico: nombre_columna_real | None}.

    Estrategia: coincidencia exacta primero, luego coincidencia normalizada
    (sin acentos/espacios/mayúsculas) para tolerar variantes de escritura.
    """
    normalized_columns = {_normalize(col): col for col in df.columns}
    mapping: dict[str, str | None] = {}

    for field, candidates in CANDIDATE_NAMES.items():
        found = None
        # 1) coincidencia exacta
        for candidate in candidates:
            if candidate in df.columns:
                found = candidate
                break
        # 2) coincidencia normalizada
        if found is None:
            for candidate in candidates:
                norm_candidate = _normalize(candidate)
                if norm_candidate in normalized_columns:
                    found = normalized_columns[norm_candidate]
                    break
        mapping[field] = found

    return mapping


def get_missing_required(mapping: dict[str, str | None]) -> list[str]:
    """Devuelve los campos obligatorios que no pudieron mapearse."""
    return [f for f in REQUIRED_FIELDS if not mapping.get(f)]


def get_unmapped_optional(mapping: dict[str, str | None]) -> list[str]:
    """Devuelve los campos opcionales que no se encontraron en el archivo."""
    return [f for f in OPTIONAL_FIELDS if not mapping.get(f)]


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Renombra las columnas mapeadas a sus nombres canónicos.

    Las columnas del archivo original que no fueron mapeadas a ningún campo
    canónico se conservan tal cual (por ejemplo, cualquier columna extra que
    el usuario quiera inspeccionar), no se eliminan aquí.
    """
    rename_dict = {col: field for field, col in mapping.items() if col is not None}
    return df.rename(columns=rename_dict)


def detect_single_value_competition(df: pd.DataFrame, competition_col: str = "competition") -> str | None:
    """Si la columna de competición tiene un único valor (caso típico de un
    archivo football-data.co.uk de una sola liga, ej. 'E0'), lo devuelve
    para asignarlo automáticamente sin pedir selección manual.
    """
    if competition_col not in df.columns:
        return None
    unique_values = df[competition_col].dropna().unique()
    if len(unique_values) == 1:
        return str(unique_values[0])
    return None


def infer_season_label(dates: pd.Series) -> str:
    """Infiere la etiqueta de temporada (ej. '2025-2026') a partir del rango
    de fechas del archivo, asumiendo el calendario europeo estándar
    (la temporada arranca en julio/agosto y termina en mayo/junio).

    football-data.co.uk entrega un archivo por temporada y no incluye una
    columna de temporada explícita, así que esto reemplaza a pedirla a mano.
    """
    valid_dates = dates.dropna()
    if valid_dates.empty:
        return "Temporada desconocida"

    min_date = valid_dates.min()
    max_date = valid_dates.max()

    start_year = min_date.year if min_date.month >= 7 else min_date.year - 1

    # Si la fecha más tardía ya cae en el año calendario siguiente al de
    # inicio, la temporada cruza el año nuevo (caso normal en Europa).
    end_year = start_year + 1 if max_date.year > start_year else start_year

    if end_year == start_year:
        return str(start_year)
    return f"{start_year}-{end_year}"


def has_xg_data(mapping: dict[str, str | None]) -> bool:
    """Indica si el archivo trae columnas de xG. Si no las trae (caso
    normal para archivos football-data.co.uk como el de referencia), el
    resto del pipeline debe seguir funcionando solo con goles reales.
    """
    return bool(mapping.get("home_xg")) and bool(mapping.get("away_xg"))


def has_market_odds(mapping: dict[str, str | None]) -> bool:
    """Indica si el archivo trae cuotas de mercado utilizables para comparar
    contra los modelos propios (ver `market_odds.py`)."""
    return bool(
        mapping.get("market_home_odds") and mapping.get("market_draw_odds") and mapping.get("market_away_odds")
    )
