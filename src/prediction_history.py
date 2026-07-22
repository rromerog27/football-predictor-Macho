"""Historial de predicciones con persistencia local.

Se guarda en un CSV simple dentro de `data/` (no se usa SQLite: para una
app que corre en la máquina del usuario, un CSV es más simple de inspeccionar
y mantener). Permite registrar cada predicción generada, completar después
el resultado real del partido, y calcular el rendimiento histórico del
sistema (aciertos/fallos), general y por modelo utilizado.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_PATH = DATA_DIR / "prediction_history.csv"

# Versión de la lógica de predicción que generó el registro (no del archivo
# de datos del usuario). Súbela manualmente si cambia el método de cálculo
# de las probabilidades, para poder distinguir predicciones antiguas.
MODEL_VERSION = "fase3-1.0"

HISTORY_COLUMNS = [
    "id",
    "timestamp",
    "file_name",
    "competition",
    "season",
    "home_team",
    "away_team",
    "model_used",
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
    "predicted_outcome",
    "confidence",
    "expected_home_goals",
    "expected_away_goals",
    "actual_home_goals",
    "actual_away_goals",
    "actual_outcome",
    "correct",
    "model_version",
]

_TRUE_STRINGS = {"true", "1", "1.0"}

# Columnas de texto: se fuerzan a dtype "object" al leer el CSV porque,
# cuando todavía no tienen ningún valor (p. ej. actual_outcome antes de
# resolver un partido), pandas las infiere como float64 (todo NaN), y
# después falla al intentar escribir un string ahí.
_STRING_COLUMNS = [
    "file_name",
    "competition",
    "season",
    "home_team",
    "away_team",
    "model_used",
    "predicted_outcome",
    "confidence",
    "actual_outcome",
    "correct",
    "model_version",
]


def _ensure_history_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")


def load_history() -> pd.DataFrame:
    """Carga el historial completo (crea el archivo vacío si no existe)."""
    _ensure_history_file()
    try:
        df = pd.read_csv(
            HISTORY_PATH,
            encoding="utf-8-sig",
            dtype={col: "object" for col in _STRING_COLUMNS},
        )
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[HISTORY_COLUMNS]


def append_prediction(record: dict) -> int:
    """Agrega una predicción al historial. Devuelve el id asignado."""
    df = load_history()
    new_id = int(df["id"].max()) + 1 if not df.empty and df["id"].notna().any() else 1

    row = {col: record.get(col, "") for col in HISTORY_COLUMNS}
    row["id"] = new_id
    row["model_version"] = MODEL_VERSION

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    logger.info("Predicción #%d guardada en el historial (%s vs %s).", new_id, record.get("home_team"), record.get("away_team"))
    return new_id


def pending_records(history_df: pd.DataFrame) -> pd.DataFrame:
    """Predicciones guardadas a las que aún no se les cargó el resultado real."""
    return history_df[history_df["actual_outcome"].isna() | (history_df["actual_outcome"].astype(str).str.strip() == "")]


def update_actual_result(record_id: int, actual_home_goals: int, actual_away_goals: int) -> None:
    """Completa el resultado real de una predicción guardada y calcula si
    el pronóstico acertó."""
    df = load_history()
    mask = df["id"] == record_id
    if not mask.any():
        raise ValueError(f"No existe una predicción con id {record_id} en el historial.")

    if actual_home_goals > actual_away_goals:
        actual_outcome = "H"
    elif actual_home_goals < actual_away_goals:
        actual_outcome = "A"
    else:
        actual_outcome = "D"

    df.loc[mask, "actual_home_goals"] = actual_home_goals
    df.loc[mask, "actual_away_goals"] = actual_away_goals
    df.loc[mask, "actual_outcome"] = actual_outcome
    df.loc[mask, "correct"] = df.loc[mask, "predicted_outcome"] == actual_outcome
    df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    logger.info("Resultado real cargado para la predicción #%d.", record_id)


def compute_performance(history_df: pd.DataFrame) -> dict:
    """Rendimiento histórico del sistema: aciertos sobre las predicciones
    que ya tienen un resultado real cargado, general y por modelo."""
    resolved = history_df[history_df["actual_outcome"].notna() & (history_df["actual_outcome"].astype(str).str.strip() != "")]
    total = len(history_df)
    n_resolved = len(resolved)

    if n_resolved == 0:
        return {"total_predictions": total, "resolved": 0, "accuracy": None, "by_model": {}}

    def _is_correct(value) -> bool:
        return str(value).strip().lower() in _TRUE_STRINGS

    overall_correct = resolved["correct"].apply(_is_correct)
    accuracy = round(overall_correct.mean() * 100, 1)

    by_model = {}
    for model_name, group in resolved.groupby("model_used"):
        group_correct = group["correct"].apply(_is_correct)
        by_model[str(model_name)] = {"n": len(group), "accuracy": round(group_correct.mean() * 100, 1)}

    return {"total_predictions": total, "resolved": n_resolved, "accuracy": accuracy, "by_model": by_model}
