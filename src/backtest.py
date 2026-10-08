"""Backtest contra el mercado: predicciones fuera de muestra vs. cuotas de cierre.

Toma las predicciones del periodo de validación de un modelo entrenado (que la
regresión logística no vio al entrenar), las cruza por fecha, marcador y
nombres con los partidos de football-data.co.uk y compara:

- log loss 1X2 del modelo frente al de las cuotas de cierre (sin margen);
- la mezcla modelo + mercado: si el modelo aporta información que el mercado
  no tiene, mezclarlo mejora al mercado solo (peso por validación cruzada
  temporal en dos mitades, para no medir sobre los mismos partidos);
- log loss de Over/Under 2.5 (si hay cuotas);
- ROI simulado apostando 1 unidad, a la cuota de cierre, al resultado en el
  que el modelo supera al mercado en al menos un umbral.

Las cuotas de cierre son la referencia más exigente: cualquier modelo público
suele quedar por detrás. El objetivo es medir la distancia y si se acorta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import football_data_source as fd
from src.match_model import CLASSES, LeagueModel, PredictionError, multiclass_log_loss, team_similarity

MIN_NAME_SIMILARITY = 0.4  # con fecha (±1 día) y marcador exactos, basta un parecido moderado
ROI_THRESHOLDS = (0.03, 0.05, 0.10)
BLEND_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


@dataclass
class BacktestResult:
    competition: str
    n_val: int  # partidos de validación de la competición
    n_matched: int  # cruzados con cuotas
    period: str
    odds_source: str
    ll_baseline: float
    ll_model: float
    ll_poisson: float
    ll_logistic: float
    ll_market: float
    alpha: float  # peso del modelo en la mezcla modelo + mercado (con todos los partidos)
    ll_blend_cv: float  # log loss de la mezcla con el peso elegido en la otra mitad
    n_ou: int
    ll_ou_model: float | None
    ll_ou_market: float | None
    roi: pd.DataFrame
    urls: list[str]
    matched: pd.DataFrame = field(repr=False)

    @property
    def gap(self) -> float:
        """Distancia al mercado (log loss modelo − mercado; menor es mejor, 0 = igual que el mercado)."""
        return self.ll_model - self.ll_market


def market_probs(odds: np.ndarray) -> np.ndarray:
    inv = 1 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def match_predictions(preds: pd.DataFrame, fd_rows: pd.DataFrame) -> pd.DataFrame:
    """Empareja predicciones con partidos de football-data (misma fecha ±1 día, mismo
    marcador y nombres parecidos). Cada partido de football-data se usa una vez."""
    preds = preds.assign(date=preds["datetime"].dt.normalize(), _i=np.arange(len(preds)))
    cand = preds.merge(fd_rows.reset_index(names="_j"), on=["hg", "ag"], suffixes=("", "_fd"))
    cand = cand[(cand["date"] - cand["date_fd"]).abs() <= pd.Timedelta(days=1)]
    if cand.empty:
        return cand
    cand["score"] = [min(team_similarity(h, hf), team_similarity(a, af))
                     for h, hf, a, af in zip(cand["home"], cand["home_fd"], cand["away"], cand["away_fd"])]
    cand = cand[cand["score"] >= MIN_NAME_SIMILARITY].sort_values("score", ascending=False)
    cand = cand.drop_duplicates("_i").drop_duplicates("_j")
    return cand.sort_values("datetime").reset_index(drop=True)


def _binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _blend_cv(y: np.ndarray, p_model: np.ndarray, p_market: np.ndarray) -> tuple[float, float]:
    """Peso del modelo en la mezcla (con todos) y log loss de la mezcla con validación
    cruzada temporal en dos mitades (el peso de una mitad se evalúa en la otra)."""
    def best_alpha(idx):
        losses = [multiclass_log_loss(y[idx], a * p_model[idx] + (1 - a) * p_market[idx]) for a in BLEND_GRID]
        return float(BLEND_GRID[int(np.argmin(losses))])

    n = len(y)
    halves = [np.arange(n // 2), np.arange(n // 2, n)]
    total = 0.0
    for fit, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        a = best_alpha(fit)
        total += multiclass_log_loss(y[test], a * p_model[test] + (1 - a) * p_market[test]) * len(test)
    return best_alpha(np.arange(n)), total / n


def _roi_table(matched: pd.DataFrame, p_model: np.ndarray, p_market: np.ndarray) -> pd.DataFrame:
    odds = matched[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    outcome = matched["result"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    edge = p_model - p_market
    pick = edge.argmax(axis=1)
    best_edge = edge[np.arange(len(edge)), pick]
    rows = []
    for th in ROI_THRESHOLDS:
        bets = best_edge >= th
        won = bets & (pick == outcome)
        profit = (odds[np.arange(len(odds)), pick] - 1) * won - (bets & ~won)
        n = int(bets.sum())
        rows.append({"umbral": f"≥{th * 100:.0f} pts", "apuestas": n, "aciertos": int(won.sum()),
                     "ROI": float(profit.sum() / n) if n else np.nan})
    return pd.DataFrame(rows)


def run(model: LeagueModel) -> BacktestResult:
    data, t = model.data, model.trained
    code = data.competition
    if not fd.has_odds(code):
        raise PredictionError(f"football-data.co.uk no publica cuotas de {data.name}: no hay backtest.")
    preds = t.val_predictions[t.val_predictions["competition"] == code]
    if preds.empty:
        raise PredictionError(f"Sin partidos de validación de {data.name}.")
    fd_rows, urls = fd.load(code, preds["datetime"].min(), preds["datetime"].max())
    matched = match_predictions(preds, fd_rows)
    matched = matched[matched[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)].reset_index(drop=True)
    if len(matched) < 50:
        raise PredictionError(f"Solo {len(matched)} partidos de {data.name} cruzados con cuotas: muy pocos.")

    y = matched["result"].to_numpy()
    cols = {k: [f"p_{k}_{c}" for c in CLASSES] for k in ("ensemble", "poisson", "logistic")}
    p_model = matched[cols["ensemble"]].to_numpy()
    p_market = market_probs(matched[["odds_h", "odds_d", "odds_a"]].to_numpy(float))
    freq = np.array([np.mean(y == c) for c in CLASSES])  # referencia: frecuencias del propio periodo
    alpha, ll_blend = _blend_cv(y, p_model, p_market)

    ou = matched.dropna(subset=["odds_over25", "odds_under25"])
    ll_ou_model = ll_ou_market = None
    if len(ou) >= 50:
        y_ou = ((ou["hg"] + ou["ag"]) >= 3).to_numpy(float)
        inv = 1 / ou[["odds_over25", "odds_under25"]].to_numpy(float)
        ll_ou_model = _binary_log_loss(y_ou, ou["over25"].to_numpy())
        ll_ou_market = _binary_log_loss(y_ou, inv[:, 0] / inv.sum(axis=1))

    sources = matched["odds_source"].value_counts()
    return BacktestResult(
        competition=code,
        n_val=len(preds),
        n_matched=len(matched),
        period=f"{matched['datetime'].min():%d/%m/%Y} – {matched['datetime'].max():%d/%m/%Y}",
        odds_source=sources.index[0] if len(sources) else "",
        ll_baseline=multiclass_log_loss(y, np.tile(freq, (len(y), 1))),
        ll_model=multiclass_log_loss(y, p_model),
        ll_poisson=multiclass_log_loss(y, matched[cols["poisson"]].to_numpy()),
        ll_logistic=multiclass_log_loss(y, matched[cols["logistic"]].to_numpy()),
        ll_market=multiclass_log_loss(y, p_market),
        alpha=alpha,
        ll_blend_cv=ll_blend,
        n_ou=len(ou),
        ll_ou_model=ll_ou_model,
        ll_ou_market=ll_ou_market,
        roi=_roi_table(matched, p_model, p_market),
        urls=urls,
        matched=matched,
    )
