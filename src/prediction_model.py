"""Modelo de respaldo: regresión logística multiclase calibrada.

No reemplaza al modelo de Poisson (`poisson_model.py`): sirve como segundo
punto de referencia, entrenado sobre variables pre-partido (ver
`feature_engineering.py`) con una división cronológica de entrenamiento y
prueba (nunca aleatoria, para no mezclar pasado y futuro).

También implementa la predicción combinada (Poisson + regresión logística),
ponderando cada modelo según su desempeño de validación (log loss) en el
mismo split cronológico — una ponderación objetiva y explicable, no
arbitraria.
"""

from __future__ import annotations

import warnings as warnings_module
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import feature_engineering, poisson_model
from src.utils import CLASS_LABELS, MIN_MATCHES_FOR_ML_MODEL, RANDOM_SEED, get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILENAME = "logistic_model_latest.joblib"

TEST_FRACTION = 0.2
MIN_TRAIN_FOR_CALIBRATION = 60  # por debajo de esto, calibrar con CV es poco fiable


@dataclass
class ValidationMetrics:
    n_train: int
    n_test: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    log_loss: float
    brier_score: float
    confusion_matrix: list[list[int]]
    labels: list[str] = field(default_factory=lambda: list(CLASS_LABELS))


@dataclass
class LogisticModelResult:
    trained: bool
    model: Pipeline | None = None
    metrics: ValidationMetrics | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    n_usable_matches: int = 0
    n_skipped_matches: int = 0
    calibrated: bool = False
    trained_at: str | None = None
    # Detalle del propio conjunto de prueba (no el de Poisson): fecha,
    # equipos, resultado real y — cuando el archivo las trae — las cuotas
    # de mercado de esos partidos. Se usa en market_odds.py para comparar
    # el modelo contra el mercado y simular apuestas de valor.
    test_meta: pd.DataFrame | None = None
    test_predictions: list[dict[str, float]] = field(default_factory=list)


def _brier_score_multiclass(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    one_hot = np.array([[1.0 if c == label else 0.0 for c in classes] for label in y_true])
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _feature_importance(pipeline: Pipeline) -> dict[str, float]:
    """Magnitud promedio (entre clases) de los coeficientes estandarizados,
    como proxy de importancia de variable. Se calcula sobre un pipeline SIN
    calibrar (la calibración envuelve varios clasificadores internos y no
    expone coeficientes de forma directa ni fiable)."""
    classifier: LogisticRegression = pipeline.named_steps["classifier"]
    coefs = np.abs(classifier.coef_).mean(axis=0)
    return {
        feat: round(float(val), 4)
        for feat, val in sorted(
            zip(feature_engineering.FEATURE_COLUMNS, coefs), key=lambda x: x[1], reverse=True
        )
    }


def save_model(result: LogisticModelResult) -> Path | None:
    """Guarda el modelo entrenado con joblib en `models/`."""
    if not result.trained or result.model is None:
        return None
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / MODEL_FILENAME
    joblib.dump(
        {
            "model": result.model,
            "feature_columns": feature_engineering.FEATURE_COLUMNS,
            "classes": CLASS_LABELS,
            "trained_at": result.trained_at,
            "metrics": result.metrics,
        },
        path,
    )
    logger.info("Modelo guardado en %s", path)
    return path


def load_model() -> dict | None:
    path = MODEL_DIR / MODEL_FILENAME
    if not path.exists():
        return None
    return joblib.load(path)


def train_logistic_model(historical_df: pd.DataFrame, recent_n: int = 5) -> LogisticModelResult:
    """Entrena la regresión logística de respaldo con división cronológica.

    Si no hay suficientes partidos utilizables (tras calcular variables
    pre-partido, ver `feature_engineering.MIN_PRIOR_MATCHES`), devuelve un
    resultado con `trained=False` y una advertencia clara: en ese caso el
    dashboard debe mostrar solo el modelo de Poisson.
    """
    X, y, meta, skipped = feature_engineering.build_training_dataset(historical_df, recent_n=recent_n)
    n_usable = len(X)

    if n_usable < MIN_MATCHES_FOR_ML_MODEL:
        return LogisticModelResult(
            trained=False,
            n_usable_matches=n_usable,
            n_skipped_matches=skipped,
            warnings=[
                f"Solo hay {n_usable} partidos utilizables para entrenar la regresión logística "
                f"(se requieren al menos {MIN_MATCHES_FOR_ML_MODEL} tras calcular variables "
                f"pre-partido). Se usará únicamente el modelo de Poisson."
            ],
        )

    n_test = max(1, int(round(n_usable * TEST_FRACTION)))
    n_train = n_usable - n_test
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    meta_test = meta.iloc[n_train:].reset_index(drop=True)

    model_warnings: list[str] = []
    base_pipeline = _build_pipeline()
    base_pipeline.fit(X_train, y_train)
    feature_importance = _feature_importance(base_pipeline)

    calibrated = False
    final_pipeline = base_pipeline
    if n_train >= MIN_TRAIN_FOR_CALIBRATION:
        try:
            cv_folds = 3 if n_train >= 90 else 2
            calibrated_model = CalibratedClassifierCV(_build_pipeline(), method="sigmoid", cv=cv_folds)
            with warnings_module.catch_warnings():
                warnings_module.simplefilter("ignore")
                calibrated_model.fit(X_train, y_train)
            final_pipeline = calibrated_model
            calibrated = True
        except ValueError as exc:
            model_warnings.append(
                f"No se pudo calibrar el modelo (datos insuficientes por clase): {exc}. "
                "Se usará el modelo sin calibrar."
            )
    else:
        model_warnings.append(
            f"El conjunto de entrenamiento ({n_train} partidos) es pequeño para calibrar "
            "probabilidades de forma confiable; se usa el modelo sin calibrar."
        )

    proba_test = final_pipeline.predict_proba(X_test)
    classes_order = list(final_pipeline.classes_) if hasattr(final_pipeline, "classes_") else list(base_pipeline.classes_)
    pred_test = np.array(classes_order)[np.argmax(proba_test, axis=1)]

    proba_df = pd.DataFrame(proba_test, columns=classes_order)[CLASS_LABELS].values
    test_predictions = [
        {label: round(float(p) * 100, 2) for label, p in zip(CLASS_LABELS, row)} for row in proba_df
    ]

    metrics = ValidationMetrics(
        n_train=n_train,
        n_test=n_test,
        accuracy=round(float((pred_test == y_test.values).mean()), 4),
        precision_macro=round(float(precision_score(y_test, pred_test, labels=CLASS_LABELS, average="macro", zero_division=0)), 4),
        recall_macro=round(float(recall_score(y_test, pred_test, labels=CLASS_LABELS, average="macro", zero_division=0)), 4),
        f1_macro=round(float(f1_score(y_test, pred_test, labels=CLASS_LABELS, average="macro", zero_division=0)), 4),
        log_loss=round(float(log_loss(y_test, proba_df, labels=CLASS_LABELS)), 4),
        brier_score=round(_brier_score_multiclass(y_test.values, proba_df, CLASS_LABELS), 4),
        confusion_matrix=sk_confusion_matrix(y_test, pred_test, labels=CLASS_LABELS).tolist(),
    )

    if n_usable < MIN_MATCHES_FOR_ML_MODEL * 1.5:
        model_warnings.append(
            "El volumen de datos es moderado: las métricas de validación pueden variar bastante "
            "al cargar más partidos o temporadas adicionales."
        )

    result = LogisticModelResult(
        trained=True,
        model=final_pipeline,
        metrics=metrics,
        feature_importance=feature_importance,
        warnings=model_warnings,
        n_usable_matches=n_usable,
        n_skipped_matches=skipped,
        calibrated=calibrated,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        test_meta=meta_test,
        test_predictions=test_predictions,
    )
    save_model(result)
    return result


def predict_logistic(
    model_result: LogisticModelResult, historical_df: pd.DataFrame, home_team: str, away_team: str, recent_n: int = 5
) -> dict[str, float] | None:
    """Probabilidades H/D/A del modelo de respaldo para un partido en vivo.
    Devuelve None si el modelo no está entrenado."""
    if not model_result.trained or model_result.model is None:
        return None

    features = feature_engineering.build_feature_row(historical_df, home_team, away_team, cutoff_date=None, recent_n=recent_n)
    X_live = pd.DataFrame([features], columns=feature_engineering.FEATURE_COLUMNS)

    proba = model_result.model.predict_proba(X_live)[0]
    classes_order = list(model_result.model.classes_)
    proba_by_class = dict(zip(classes_order, proba))

    return {
        "home_win": round(proba_by_class.get("H", 0.0) * 100, 1),
        "draw": round(proba_by_class.get("D", 0.0) * 100, 1),
        "away_win": round(proba_by_class.get("A", 0.0) * 100, 1),
    }


# --------------------------------------------------------------------------
# Predicción combinada (Poisson + regresión logística)
# --------------------------------------------------------------------------


@dataclass
class CombinedPrediction:
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    weight_poisson: float
    weight_logistic: float
    weighting_explanation: str
    confidence: str
    confidence_breakdown: dict[str, float]


def _weights_from_log_loss(poisson_log_loss: float | None, logistic_log_loss: float | None) -> tuple[float, float, str]:
    """Pondera cada modelo de forma inversamente proporcional a su log loss
    de validación (menor log loss = mejor calibración = más peso), medido
    ambos en el MISMO split cronológico de prueba. Si algún log loss no
    está disponible, se reparte 50/50 con una advertencia explicativa."""
    if poisson_log_loss is None or logistic_log_loss is None or poisson_log_loss <= 0 or logistic_log_loss <= 0:
        return 0.5, 0.5, (
            "No se pudo calcular el log loss de validación de ambos modelos con los datos "
            "disponibles; se combinaron con el mismo peso (50/50)."
        )

    inv_poisson = 1 / poisson_log_loss
    inv_logistic = 1 / logistic_log_loss
    total = inv_poisson + inv_logistic
    w_poisson = inv_poisson / total
    w_logistic = inv_logistic / total
    explanation = (
        f"Ponderación basada en el desempeño de validación (log loss) de cada modelo sobre los "
        f"mismos partidos de prueba: Poisson={poisson_log_loss:.3f}, Regresión logística="
        f"{logistic_log_loss:.3f}. El modelo con menor log loss (mejor calibrado) recibe más peso "
        f"({w_poisson * 100:.0f}% Poisson / {w_logistic * 100:.0f}% regresión logística)."
    )
    return w_poisson, w_logistic, explanation


def _confidence_from_breakdown(breakdown: dict[str, float], max_possible: float) -> str:
    score_pct = sum(breakdown.values()) / max_possible * 100 if max_possible > 0 else 0
    if score_pct >= 70:
        return "alta"
    if score_pct >= 40:
        return "media"
    return "baja"


def simple_confidence(margin: float, log_loss_value: float | None) -> str:
    """Nivel de confianza simplificado para el modo 'solo regresión
    logística', combinando el margen entre la probabilidad más alta y la
    segunda (50%) con el desempeño de validación del modelo (50%), medido
    como qué tanto mejora su log loss frente al azar (ln(3))."""
    random_baseline = float(np.log(3))
    if log_loss_value is not None and log_loss_value > 0:
        validation_score = max(0.0, min(1.0, (random_baseline - log_loss_value) / random_baseline)) * 50
    else:
        validation_score = 25.0
    margin_score = min(margin / 30, 1.0) * 50
    total = validation_score + margin_score
    if total >= 70:
        return "alta"
    if total >= 40:
        return "media"
    return "baja"


def combine_predictions(
    poisson_pred: poisson_model.PoissonPrediction,
    logistic_probs: dict[str, float] | None,
    poisson_backtest: poisson_model.BacktestMetrics | None,
    logistic_metrics: ValidationMetrics | None,
    stat_completeness_pct: float,
) -> CombinedPrediction:
    """Combina Poisson y regresión logística (si está disponible) en una
    única predicción 1X2, y calcula un nivel de confianza a partir de
    varios factores medibles (no una sola heurística aislada):

    - Cantidad de partidos disponibles para los equipos (Poisson).
    - Diferencia entre la probabilidad más alta y la segunda (margen).
    - Consistencia entre modelos (si ambos coinciden en el resultado más probable).
    - Desempeño de validación (log loss del mejor modelo disponible frente al azar).
    - Completitud de los datos (columnas estadísticas disponibles en el archivo).
    """
    if logistic_probs is None:
        combined = {
            "home_win": poisson_pred.prob_home_win,
            "draw": poisson_pred.prob_draw,
            "away_win": poisson_pred.prob_away_win,
        }
        w_poisson, w_logistic = 1.0, 0.0
        explanation = "Solo el modelo de Poisson está disponible (regresión logística no entrenada)."
        agreement_active = False
    else:
        poisson_ll = poisson_backtest.log_loss if poisson_backtest else None
        logistic_ll = logistic_metrics.log_loss if logistic_metrics else None
        w_poisson, w_logistic, explanation = _weights_from_log_loss(poisson_ll, logistic_ll)
        combined_raw = {
            "home_win": w_poisson * poisson_pred.prob_home_win + w_logistic * logistic_probs["home_win"],
            "draw": w_poisson * poisson_pred.prob_draw + w_logistic * logistic_probs["draw"],
            "away_win": w_poisson * poisson_pred.prob_away_win + w_logistic * logistic_probs["away_win"],
        }
        total = sum(combined_raw.values())
        combined = {k: round(v / total * 100, 1) for k, v in combined_raw.items()} if total > 0 else combined_raw
        agreement_active = True

    sorted_probs = sorted(combined.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

    data_volume = min(poisson_pred.home_matches_used, poisson_pred.away_matches_used)
    data_volume_score = min(data_volume / 20, 1.0) * 25

    margin_score = min(margin / 30, 1.0) * 25

    best_log_loss = None
    if logistic_probs is not None and logistic_metrics is not None:
        best_log_loss = logistic_metrics.log_loss
    elif poisson_backtest is not None and poisson_backtest.log_loss is not None:
        best_log_loss = poisson_backtest.log_loss
    random_baseline = float(np.log(3))
    if best_log_loss is not None:
        validation_score = max(0.0, min(1.0, (random_baseline - best_log_loss) / random_baseline)) * 20
    else:
        validation_score = 12.0  # neutral: sin backtest disponible

    completeness_score = min(max(stat_completeness_pct, 0.0), 100.0) / 100 * 10

    breakdown = {
        "Cantidad de datos disponibles": round(data_volume_score, 1),
        "Diferencia entre probabilidades": round(margin_score, 1),
        "Desempeño en validación": round(validation_score, 1),
        "Completitud de los datos": round(completeness_score, 1),
    }
    max_possible = 25 + 25 + 20 + 10

    if agreement_active:
        poisson_top = max(
            {"home_win": poisson_pred.prob_home_win, "draw": poisson_pred.prob_draw, "away_win": poisson_pred.prob_away_win},
            key=lambda k: {"home_win": poisson_pred.prob_home_win, "draw": poisson_pred.prob_draw, "away_win": poisson_pred.prob_away_win}[k],
        )
        logistic_top = max(logistic_probs, key=logistic_probs.get)
        agreement_score = 20.0 if poisson_top == logistic_top else 8.0
        breakdown["Consistencia entre modelos"] = agreement_score
        max_possible += 20

    confidence = _confidence_from_breakdown(breakdown, max_possible)

    return CombinedPrediction(
        prob_home_win=combined["home_win"],
        prob_draw=combined["draw"],
        prob_away_win=combined["away_win"],
        weight_poisson=round(w_poisson * 100, 1),
        weight_logistic=round(w_logistic * 100, 1),
        weighting_explanation=explanation,
        confidence=confidence,
        confidence_breakdown=breakdown,
    )
