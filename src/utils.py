"""Utilidades comunes compartidas por todos los módulos del proyecto."""

from __future__ import annotations

import logging
from typing import Any

# Semilla global para reproducibilidad (usada por scikit-learn en Fase 2).
RANDOM_SEED = 42

# Debajo de este número de partidos históricos, cualquier estadística o
# predicción debe mostrarse con una advertencia explícita de datos insuficientes.
MIN_MATCHES_RELIABLE = 5

# Umbral usado en Fase 2 para decidir si hay datos suficientes para entrenar
# un modelo de regresión logística en vez de depender solo de Poisson.
MIN_MATCHES_FOR_ML_MODEL = 100

INSUFFICIENT_DATA_LABEL = "Datos insuficientes"

# Orden fijo de las clases de resultado (Away/Draw/Home) usado de forma
# consistente por poisson_model.py y prediction_model.py, para que sus
# métricas de validación (log loss, matriz de confusión) sean comparables.
CLASS_LABELS = ["A", "D", "H"]


def get_logger(name: str) -> logging.Logger:
    """Crea (o recupera) un logger configurado de forma consistente."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def safe_divide(numerator: float, denominator: float, default: Any = None) -> Any:
    """División segura que evita ZeroDivisionError.

    Devuelve `default` (por defecto None) cuando el denominador es 0, NaN
    o None, en vez de lanzar una excepción o inventar un valor.
    """
    if denominator is None or numerator is None:
        return default
    try:
        if denominator == 0:
            return default
        result = numerator / denominator
        if result != result:  # NaN check sin depender de numpy/pandas aquí
            return default
        return result
    except (TypeError, ZeroDivisionError):
        return default


def safe_pct(numerator: float, denominator: float, decimals: int = 1) -> Any:
    """Calcula un porcentaje seguro (0-100) o None si no se puede calcular."""
    ratio = safe_divide(numerator, denominator, default=None)
    if ratio is None:
        return None
    return round(ratio * 100, decimals)


def format_pct(value: Any, decimals: int = 1) -> str:
    """Formatea un valor porcentual para mostrarlo en el dashboard."""
    if value is None:
        return INSUFFICIENT_DATA_LABEL
    return f"{value:.{decimals}f}%"


def format_metric(value: Any, decimals: int = 2, suffix: str = "") -> str:
    """Formatea una métrica numérica genérica, o el aviso de datos insuficientes."""
    if value is None:
        return INSUFFICIENT_DATA_LABEL
    return f"{value:.{decimals}f}{suffix}"


def is_sufficient(n_matches: int, threshold: int = MIN_MATCHES_RELIABLE) -> bool:
    """Indica si hay suficientes partidos para considerar una métrica confiable."""
    return n_matches is not None and n_matches >= threshold
