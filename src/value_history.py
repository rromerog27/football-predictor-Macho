"""Cómo salió, en partidos ya jugados, apostar cada tipo de "valor" del modelo frente a la casa.

La app marca los resultados a los que el modelo da bastante más que el mercado ("+12 al 1") y los picks con
valor (probabilidad × cuota − 1). Que el modelo vea valor no quiere decir que apostarlo gane: depende de la liga
y del tipo de apuesta. En Argentina, por ejemplo, el modelo espera menos goles de los que salen, y apostar al
menos de 2.5 o al empate cuando veía valor perdía plata; en Chile, apostar al local con valor ganaba.

Aquí se mide eso con los partidos de validación (que el modelo no usó para ajustarse) cruzados con las cuotas
de cierre (`backtest.BacktestResult.matched`): cada vez que se cumplía la misma condición, 1 unidad a la cuota
de cierre. Con pocos partidos el resultado es casi todo suerte, así que se muestra con el número de apuestas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import performance as perf
from src.match_model import CLASSES

OUTCOMES = ("local", "empate", "visitante")
EDGE_BANDS = ((0.05, 0.10), (0.10, 1.0))  # puntos que el modelo da de más que el mercado (sin margen)
OU_EDGE = 0.05  # más/menos de 2.5: puntos de más que el mercado
MIN_BETS = 20  # menos apuestas: sin historial
BAD_ROI, GOOD_ROI = -0.10, 0.05  # por debajo: "en contra"; por encima: "a favor"


@dataclass(frozen=True)
class Record:
    """Apuestas de 1 unidad a la cuota de cierre que cumplían la condición, en partidos ya jugados."""
    what: str  # la condición, en una frase ("al local con 10+ pts más que el mercado")
    n: int
    wins: int
    roi: float  # ganancia media por unidad

    @property
    def verdict(self) -> str:
        """"en contra", "a favor" o "sin señal" (con MIN_BETS apuestas o más)."""
        if self.roi <= BAD_ROI:
            return "en contra"
        return "a favor" if self.roi >= GOOD_ROI else "sin señal"

    @property
    def short(self) -> str:
        pct = round(abs(self.roi) * 100)
        if pct == 0:
            return f"quedó en empate en {self.n} partidos"
        return f"{'ganó' if self.roi > 0 else 'perdió'} {pct}% en {self.n} partidos"

    @property
    def text(self) -> str:
        return (f"En esta liga, apostar {self.what} {self.short} ya jugados ({self.wins} acertadas, a la cuota "
                "de cierre)")


def _record(what: str, odds: np.ndarray, won: np.ndarray) -> Record | None:
    n = len(odds)
    if n < MIN_BETS:
        return None
    profit = np.where(won, odds - 1, -1.0)
    return Record(what, n, int(won.sum()), float(profit.mean()))


def _band_text(lo: float, hi: float) -> str:
    return f"{lo * 100:.0f}+ pts" if hi >= 1 else f"{lo * 100:.0f}–{hi * 100:.0f} pts"


@dataclass(frozen=True)
class ValueHistory:
    """Historial de una competición: 1X2 por resultado y tramo de puntos de más, y más/menos de 2.5."""
    one_x_two: dict  # (resultado 0/1/2, tramo) → Record
    any_band: dict  # resultado → Record con 5+ pts (si el tramo no tiene apuestas suficientes)
    over_under: dict  # "más" / "menos" → Record
    favorites: Record | None = None  # la regla de los picks de resultado (ver performance.find_pick)

    def for_outcome(self, k: int, edge: float) -> Record | None:
        """El historial de apostar el resultado `k` cuando el modelo le daba `edge` puntos de más."""
        for band in EDGE_BANDS:
            if band[0] <= edge < band[1]:
                return self.one_x_two.get((k, band)) or self.any_band.get(k)
        return None

    def for_over_under(self, side: str) -> Record | None:
        return self.over_under.get(side)


def build(matched: pd.DataFrame) -> ValueHistory:
    """`matched`: partidos de validación con las probabilidades del modelo (p_model_H/D/A, over25), el
    resultado y las cuotas de cierre (odds_h/d/a, odds_over25/odds_under25)."""
    odds = matched[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    ok = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
    m, odds = matched[ok], odds[ok]
    inv = 1 / odds
    p_market = inv / inv.sum(axis=1, keepdims=True)
    p_model = m[[f"p_model_{c}" for c in CLASSES]].to_numpy(float)
    y = m["result"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    edge = p_model - p_market
    one_x_two, any_band = {}, {}
    for k, name in enumerate(OUTCOMES):
        for lo, hi in EDGE_BANDS:
            sel = (edge[:, k] >= lo) & (edge[:, k] < hi)
            rec = _record(f"al {name} cuando el modelo le daba {_band_text(lo, hi)} más que la casa",
                          odds[sel, k], y[sel] == k)
            if rec:
                one_x_two[(k, (lo, hi))] = rec
        sel = edge[:, k] >= EDGE_BANDS[0][0]
        rec = _record(f"al {name} cuando el modelo le daba {EDGE_BANDS[0][0] * 100:.0f}+ pts más que la casa",
                      odds[sel, k], y[sel] == k)
        if rec:
            any_band[k] = rec
    over_under = {}
    if {"odds_over25", "odds_under25", "over25"} <= set(m.columns):
        ou = m[["odds_over25", "odds_under25"]].to_numpy(float)
        has = np.isfinite(ou).all(axis=1) & (ou > 1).all(axis=1)
        inv = 1 / ou[has]
        p_over_market = inv[:, 0] / inv.sum(axis=1)
        p_over = m["over25"].to_numpy(float)[has]
        over = ((m["hg"] + m["ag"]).to_numpy(float) > 2.5)[has]
        for side, model_p, market_p, col, won in (("más", p_over, p_over_market, 0, over),
                                                  ("menos", 1 - p_over, 1 - p_over_market, 1, ~over)):
            sel = model_p - market_p >= OU_EDGE
            rec = _record(f"al {side} de 2.5 goles cuando el modelo le daba {OU_EDGE * 100:.0f}+ pts más que la casa",
                          ou[has][sel, col], won[sel])
            if rec:
                over_under[side] = rec
    rec = perf.pick_record(m)
    favorites = (_record("al favorito claro del modelo (65% o más y no menos que la casa)",
                         rec["odds"].to_numpy(float), rec["won"].to_numpy(bool)) if len(rec) else None)
    return ValueHistory(one_x_two, any_band, over_under, favorites)
