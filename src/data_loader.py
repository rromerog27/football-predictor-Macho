"""Carga de archivos Excel (.xlsx/.xls) y CSV con datos de partidos de fútbol.

Este módulo se encarga únicamente de LEER los archivos y de descartar la
mayoría de las columnas de cuotas de apuestas cuando el archivo tiene el
formato típico de football-data.co.uk — salvo un pequeño subconjunto (cuota
promedio/Bet365/Pinnacle de cierre y apertura para 1X2) que se conserva
como referencia del mercado para comparar contra los modelos propios (ver
`market_odds.py`). No limpia ni transforma los datos (eso es trabajo de
`data_cleaner.py`) ni mapea nombres de columnas a campos canónicos (eso es
trabajo de `column_mapper.py`).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# Columnas que football-data.co.uk usa siempre para identificar un partido.
# Si un archivo las contiene, lo tratamos como "formato football-data.co.uk"
# y le aplicamos el filtro de columnas de cuotas.
FOOTBALL_DATA_SIGNATURE_COLUMNS = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}

# Patrones de columnas de cuotas de casas de apuestas (football-data.co.uk).
# Cubren: 1X2 (B365H/D/A, BFDH...), 1X2 de cierre (B365CH...), over/under 2.5
# (B365>2.5, PC>2.5...), y hándicap asiático (AHh, B365AHH, B365CAHH...).
_ODDS_COLUMN_REGEX = re.compile(
    r"^(B365|BF[DE]|BMGM|BV|BW|CL|LB|PS|Max|Avg)C?(H|D|A)$"
    r"|^(B365|P|Max|Avg|BFE)C?[<>]2\.5$"
    r"|^AHC?h$"
    r"|^(B365|P|Max|Avg|BFE)C?AH[HA]$"
)

# De todas las columnas de cuotas 1X2, estas son las que se conservan como
# referencia del mercado (el resto — hándicap asiático, over/under, cuotas
# de casas menos representativas — se sigue descartando). `column_mapper.py`
# elige la mejor disponible por partido con una lista de prioridad: cuota
# de cierre promedio primero, con B365/Pinnacle y la versión de apertura
# como respaldo si el archivo no trae columnas "Avg".
PRESERVED_MARKET_ODDS_COLUMNS = {
    "AvgCH", "AvgCD", "AvgCA", "AvgH", "AvgD", "AvgA",
    "B365CH", "B365CD", "B365CA", "B365H", "B365D", "B365A",
    "PSCH", "PSCD", "PSCA", "PSH", "PSD", "PSA",
}


@dataclass
class LoadedFile:
    """Resultado de cargar un archivo: su dataframe crudo y metadatos."""

    filename: str
    file_type: str  # "csv" | "excel"
    sheet_names: list[str] = field(default_factory=list)
    selected_sheet: str | None = None
    raw_df: pd.DataFrame | None = None
    dropped_odds_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_file_type(filename: str) -> str:
    """Determina si un archivo es CSV o Excel según su extensión."""
    lower = filename.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return "excel"
    raise ValueError(
        f"Formato de archivo no soportado: '{filename}'. "
        "Se aceptan .xlsx, .xls y .csv."
    )


def get_excel_sheet_names(file_obj) -> list[str]:
    """Devuelve los nombres de las hojas disponibles en un archivo Excel."""
    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        excel_file = pd.ExcelFile(file_obj, engine="openpyxl")
        return excel_file.sheet_names
    except Exception as exc:  # noqa: BLE001 - queremos capturar y reportar cualquier fallo de lectura
        raise ValueError(f"No se pudo leer el archivo Excel: {exc}") from exc


def _drop_odds_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Elimina la mayoría de las columnas de cuotas de apuestas si el
    archivo tiene formato football-data.co.uk, conservando únicamente las
    de `PRESERVED_MARKET_ODDS_COLUMNS` (referencia del mercado 1X2). En
    cualquier otro archivo no se toca nada, ya que los nombres de columna
    de cuotas podrían no seguir este patrón.
    """
    if not FOOTBALL_DATA_SIGNATURE_COLUMNS.issubset(set(df.columns)):
        return df, []

    odds_columns = [
        col
        for col in df.columns
        if _ODDS_COLUMN_REGEX.match(str(col)) and col not in PRESERVED_MARKET_ODDS_COLUMNS
    ]
    if not odds_columns:
        return df, []

    logger.info(
        "Descartando %d columnas de cuotas de apuestas (se conservan las de referencia del mercado, si existen).",
        len(odds_columns),
    )
    return df.drop(columns=odds_columns), odds_columns


def read_csv(file_obj, filename: str = "archivo.csv") -> LoadedFile:
    """Lee un archivo CSV en formato football-data.co.uk (o similar).

    Usa encoding utf-8-sig porque estos archivos suelen traer BOM al inicio,
    y no parsea fechas aquí (eso se hace en data_cleaner con dayfirst=True,
    después de que column_mapper identifique cuál es la columna de fecha).
    """
    warnings: list[str] = []
    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding="utf-8-sig")
    except UnicodeDecodeError:
        warnings.append(
            "El archivo no está en UTF-8 con BOM; se reintentó con encoding 'latin-1'."
        )
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding="latin-1")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"No se pudo leer el archivo CSV '{filename}': {exc}") from exc

    df, dropped = _drop_odds_columns(df)

    return LoadedFile(
        filename=filename,
        file_type="csv",
        sheet_names=[],
        selected_sheet=None,
        raw_df=df,
        dropped_odds_columns=dropped,
        warnings=warnings,
    )


def read_excel_sheet(file_obj, sheet_name: str, filename: str = "archivo.xlsx") -> LoadedFile:
    """Lee una hoja específica de un archivo Excel."""
    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        df = pd.read_excel(file_obj, sheet_name=sheet_name, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"No se pudo leer la hoja '{sheet_name}' del archivo '{filename}': {exc}"
        ) from exc

    df, dropped = _drop_odds_columns(df)

    sheet_names = get_excel_sheet_names(file_obj) if hasattr(file_obj, "seek") else [sheet_name]

    return LoadedFile(
        filename=filename,
        file_type="excel",
        sheet_names=sheet_names,
        selected_sheet=sheet_name,
        raw_df=df,
        dropped_odds_columns=dropped,
        warnings=[],
    )


def load_file(file_obj, filename: str, sheet_name: str | None = None) -> LoadedFile:
    """Punto de entrada único: detecta el tipo de archivo y lo carga.

    Para Excel, si no se especifica `sheet_name`, se usa la primera hoja
    disponible (la app debe primero llamar a `get_excel_sheet_names` para
    permitir que el usuario elija antes de invocar esto con su selección).
    """
    file_type = detect_file_type(filename)

    if file_type == "csv":
        return read_csv(file_obj, filename)

    sheet_names = get_excel_sheet_names(file_obj)
    if not sheet_names:
        raise ValueError(f"El archivo '{filename}' no contiene hojas legibles.")
    chosen_sheet = sheet_name or sheet_names[0]
    if chosen_sheet not in sheet_names:
        raise ValueError(
            f"La hoja '{chosen_sheet}' no existe en '{filename}'. "
            f"Hojas disponibles: {', '.join(sheet_names)}"
        )
    return read_excel_sheet(file_obj, chosen_sheet, filename)


def dataset_overview(df: pd.DataFrame) -> dict:
    """Genera el resumen técnico del archivo cargado: filas, columnas,
    columnas detectadas, datos faltantes, tipos de datos, duplicados.
    """
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(1) if len(df) > 0 else missing

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_by_column": {
            col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
            for col in df.columns
            if missing[col] > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }


def concat_multiple_files(loaded_files: list[LoadedFile]) -> pd.DataFrame:
    """Concatena varios archivos (temporadas distintas) en un único
    dataframe, respetando el orden cronológico.

    Requiere que cada dataframe ya tenga una columna 'Date' (u otra usada
    para ordenar) parseable; si no está parseada aún, se intenta convertir
    aquí solo para poder ordenar, sin modificar el resto de columnas.
    """
    frames = []
    for lf in loaded_files:
        if lf.raw_df is None or lf.raw_df.empty:
            continue
        frames.append(lf.raw_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)

    date_col = next((c for c in combined.columns if str(c).lower() == "date"), None)
    if date_col is not None:
        parsed_dates = pd.to_datetime(combined[date_col], dayfirst=True, errors="coerce")
        combined = combined.assign(_sort_date=parsed_dates).sort_values(
            "_sort_date", kind="stable"
        ).drop(columns="_sort_date").reset_index(drop=True)

    return combined
