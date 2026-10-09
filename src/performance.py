"""Rendimiento del modelo en lenguaje llano y picks del día.

Funciones puras sobre las predicciones de validación (`TrainedModels.val_predictions`: partidos que el
modelo no usó para ajustarse, predichos con los datos anteriores a cada uno) y sobre esas mismas
predicciones cruzadas con cuotas de cierre (`BacktestResult.matched`). La página "Partidos del día" las
muestra como aciertos, calibración e historial en lugar de log loss, y aplica la regla de los picks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.match_model import CLASSES

CALIBRATION_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
CALIBRATION_MIN_N = 15  # predicciones mínimas para mostrar un tramo de la calibración
FAVORITE_MIN = 0.65  # "favorito claro" en los resúmenes y en los picks
AVOID_MARGIN = 0.08  # el mercado ve un favorito claro y el modelo le da al menos esto menos


def model_probs(df: pd.DataFrame) -> np.ndarray:
    return df[[f"p_model_{c}" for c in CLASSES]].to_numpy(float)


def market_probs(df: pd.DataFrame) -> np.ndarray:
    """Probabilidades 1X2 de las cuotas (columnas odds_h/d/a), sin el margen de la casa."""
    inv = 1 / df[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    return inv / inv.sum(axis=1, keepdims=True)


def outcomes(df: pd.DataFrame) -> np.ndarray:
    """Resultado de cada partido como índice de CLASSES (0 local, 1 empate, 2 visitante)."""
    return df["result"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(int)


@dataclass(frozen=True)
class Rate:
    hits: int
    n: int

    @property
    def share(self) -> float:
        return self.hits / self.n if self.n else float("nan")


def hit_rate(p: np.ndarray, y: np.ndarray) -> Rate:
    """Partidos en los que salió el resultado más probable."""
    return Rate(int((p.argmax(axis=1) == y).sum()), len(y))


def favorites(p: np.ndarray, y: np.ndarray, threshold: float = FAVORITE_MIN) -> tuple[Rate, float]:
    """Partidos con un favorito de `threshold` o más: cuántos ganó y qué probabilidad media tenía."""
    top = p.max(axis=1)
    sel = top >= threshold
    won = (p.argmax(axis=1) == y) & sel
    return Rate(int(won.sum()), int(sel.sum())), float(top[sel].mean()) if sel.any() else float("nan")


def calibration(p: np.ndarray, y: np.ndarray, edges: tuple[float, ...] = CALIBRATION_EDGES,
                min_n: int = CALIBRATION_MIN_N) -> pd.DataFrame:
    """Lo que decían las probabilidades frente a lo que pasó, por tramos (los tres resultados de cada
    partido juntos). Columnas: lo, hi, n, predicted (media del tramo), observed (frecuencia real)."""
    flat_p = p.ravel()
    flat_y = (np.arange(p.shape[1])[None, :] == y[:, None]).ravel()
    bins = np.clip(np.digitize(flat_p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = bins == b
        if sel.sum() >= min_n:
            rows.append({"lo": edges[b], "hi": edges[b + 1], "n": int(sel.sum()),
                         "predicted": float(flat_p[sel].mean()), "observed": float(flat_y[sel].mean())})
    return pd.DataFrame(rows, columns=["lo", "hi", "n", "predicted", "observed"])


def history(val: pd.DataFrame) -> pd.DataFrame:
    """Partidos de validación del más reciente al más viejo, con el pronóstico del modelo (resultado
    más probable y su probabilidad) y si acertó."""
    if val.empty:
        return val.assign(pick=[], p_pick=[], hit=[])
    p, y = model_probs(val), outcomes(val)
    out = val.assign(pick=p.argmax(axis=1), p_pick=p.max(axis=1), hit=p.argmax(axis=1) == y)
    return out.sort_values("datetime", ascending=False).reset_index(drop=True)


@dataclass(frozen=True)
class Pick:
    """Pick: el favorito del modelo, si es claro y el modelo no le da menos que el mercado."""
    outcome: int  # índice de CLASSES
    p_model: float
    p_market: float
    odds: float

    @property
    def value(self) -> float:
        """Ganancia esperada por unidad si el modelo tuviera razón (p · cuota − 1)."""
        return self.p_model * self.odds - 1


def find_pick(p_model: np.ndarray, p_market: np.ndarray, odds: np.ndarray,
              threshold: float = FAVORITE_MIN) -> Pick | None:
    k = int(np.argmax(p_model))
    if not np.isfinite(odds[k]) or odds[k] <= 1 or p_model[k] < threshold or p_model[k] < p_market[k]:
        return None
    return Pick(k, float(p_model[k]), float(p_market[k]), float(odds[k]))


def find_avoid(p_model: np.ndarray, p_market: np.ndarray, threshold: float = FAVORITE_MIN,
               margin: float = AVOID_MARGIN) -> int | None:
    """Favorito claro del mercado al que el modelo le da bastante menos (resultado a evitar)."""
    k = int(np.argmax(p_market))
    return k if p_market[k] >= threshold and p_model[k] <= p_market[k] - margin else None


def pick_record(matched: pd.DataFrame, threshold: float = FAVORITE_MIN) -> pd.DataFrame:
    """La regla de los picks aplicada a partidos ya jugados con cuotas de cierre: un pick por partido
    que la cumple, con si ganó y la ganancia de apostar 1 unidad (del más reciente al más viejo)."""
    columns = ["datetime", "home", "away", "outcome", "p_model", "p_market", "odds", "won", "profit", "hg", "ag"]
    if matched.empty:
        return pd.DataFrame(columns=columns)
    p, m, y = model_probs(matched), market_probs(matched), outcomes(matched)
    odds = matched[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    rows = []
    for i in range(len(matched)):
        pk = find_pick(p[i], m[i], odds[i], threshold)
        if pk is None:
            continue
        won = bool(y[i] == pk.outcome)
        r = matched.iloc[i]
        rows.append({"datetime": r["datetime"], "home": r["home"], "away": r["away"], "outcome": pk.outcome,
                     "p_model": pk.p_model, "p_market": pk.p_market, "odds": pk.odds, "won": won,
                     "profit": pk.odds - 1 if won else -1.0, "hg": r.get("hg"), "ag": r.get("ag")})
    return pd.DataFrame(rows, columns=columns).sort_values("datetime", ascending=False).reset_index(drop=True)
