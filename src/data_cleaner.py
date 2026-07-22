"""Limpieza y validación de los datos ya mapeados a nombres canónicos.

Este módulo asume que `column_mapper.apply_mapping()` ya se ejecutó, así
que trabaja sobre nombres canónicos (home_team, away_team, home_goals...),
nunca sobre los nombres originales del archivo.

No se inventan valores para completar datos faltantes: cuando no hay
suficiente información, se reporta explícitamente en `CleaningReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils import MIN_MATCHES_RELIABLE, get_logger

logger = get_logger(__name__)

STRING_FIELDS = ["home_team", "away_team", "referee", "competition", "season", "result", "result_ht"]

NUMERIC_FIELDS = [
    "home_goals",
    "away_goals",
    "home_goals_ht",
    "away_goals_ht",
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

# Rangos razonables para detectar valores imposibles/atípicos por columna.
# No se eliminan automáticamente: solo se reportan como advertencia, ya que
# podrían ser datos reales inusuales (goleadas, etc.) y no queremos borrar
# información real del usuario sin que lo sepa.
PLAUSIBLE_RANGES = {
    "home_goals": (0, 15),
    "away_goals": (0, 15),
    "home_goals_ht": (0, 10),
    "away_goals_ht": (0, 10),
    "home_shots": (0, 60),
    "away_shots": (0, 60),
    "home_shots_target": (0, 40),
    "away_shots_target": (0, 40),
    "home_corners": (0, 25),
    "away_corners": (0, 25),
    "home_yellow": (0, 11),
    "away_yellow": (0, 11),
    "home_red": (0, 5),
    "away_red": (0, 5),
    "home_possession": (0, 100),
    "away_possession": (0, 100),
    # Cuotas decimales: siempre > 1.0; por encima de 100 son prácticamente
    # inexistentes en 1X2 y casi seguro un error de formato del archivo.
    "market_home_odds": (1.01, 100),
    "market_draw_odds": (1.01, 100),
    "market_away_odds": (1.01, 100),
}


@dataclass
class CleaningReport:
    """Registro de todo lo que se hizo durante la limpieza, para mostrar en
    el dashboard (sección de calidad de datos)."""

    initial_rows: int = 0
    final_historical_rows: int = 0
    pending_rows: int = 0
    duplicates_removed: int = 0
    rows_without_date: int = 0
    implausible_value_warnings: list[str] = field(default_factory=list)
    general_warnings: list[str] = field(default_factory=list)
    sufficient_data: bool = True

    def add_warning(self, message: str) -> None:
        self.general_warnings.append(message)
        logger.warning(message)


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in STRING_FIELDS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan
    return df


def _parse_dates(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    if "date" not in df.columns:
        report.add_warning(
            "No se detectó columna de fecha: no será posible calcular forma "
            "reciente, orden cronológico ni temporada."
        )
        return df

    df = df.copy()
    original_non_null = df["date"].notna().sum()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    failed = original_non_null - df["date"].notna().sum()
    if failed > 0:
        report.rows_without_date = int(failed)
        report.add_warning(
            f"{failed} filas tenían una fecha con formato irreconocible y se marcaron como vacías."
        )
    return df


def _convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _flag_implausible_values(df: pd.DataFrame, report: CleaningReport) -> None:
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < low) | (df[col] > high)
        n_bad = int(mask.sum())
        if n_bad > 0:
            report.implausible_value_warnings.append(
                f"{col}: {n_bad} valor(es) fuera del rango esperado [{low}, {high}]."
            )


def _remove_duplicates(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    subset = [c for c in ["date", "home_team", "away_team"] if c in df.columns]
    if len(subset) < 2:
        subset = [c for c in ["home_team", "away_team"] if c in df.columns]
    if not subset:
        return df

    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    removed = before - len(df)
    if removed > 0:
        report.duplicates_removed = removed
        report.add_warning(f"Se eliminaron {removed} fila(s) duplicada(s) (mismo partido repetido).")
    return df


def _split_historical_pending(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "home_goals" not in df.columns or "away_goals" not in df.columns:
        # Sin goles no hay forma de distinguir jugado/pendiente; se trata
        # todo como histórico incompleto.
        return df, df.iloc[0:0]

    has_result = df["home_goals"].notna() & df["away_goals"].notna()
    historical = df[has_result].copy()
    pending = df[~has_result].copy()
    return historical, pending


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Limpia el dataframe ya mapeado a nombres canónicos.

    Devuelve (partidos_historicos, partidos_pendientes, reporte).
    Los partidos históricos quedan ordenados cronológicamente ascendente
    para que cualquier cálculo de forma/racha respete el orden temporal y
    no incurra en fuga de información.
    """
    report = CleaningReport(initial_rows=len(df))

    if df.empty:
        report.add_warning("El archivo no contiene filas.")
        report.sufficient_data = False
        return df, df, report

    working = _strip_strings(df)
    working = _parse_dates(working, report)
    working = _convert_numeric(working)
    _flag_implausible_values(working, report)
    working = _remove_duplicates(working, report)

    historical, pending = _split_historical_pending(working)

    if "date" in historical.columns and historical["date"].notna().any():
        historical = historical.sort_values("date", kind="stable").reset_index(drop=True)
    else:
        historical = historical.reset_index(drop=True)
        report.add_warning(
            "No hay fechas válidas suficientes para garantizar el orden cronológico."
        )

    pending = pending.reset_index(drop=True)

    report.final_historical_rows = len(historical)
    report.pending_rows = len(pending)

    if len(historical) < MIN_MATCHES_RELIABLE:
        report.sufficient_data = False
        report.add_warning(
            f"Solo hay {len(historical)} partido(s) con resultado. Se necesitan al menos "
            f"{MIN_MATCHES_RELIABLE} para cualquier estadística mínimamente confiable."
        )

    return historical, pending, report
