"""Comparación entre las probabilidades de los modelos propios y las del
mercado de apuestas (cuotas 1X2 ya presentes en el archivo cargado).

No se consulta ninguna cuota externa ni en vivo: todo sale de las columnas
de cuotas que `data_loader.py` conserva y `column_mapper.py` mapea a
`market_home_odds` / `market_draw_odds` / `market_away_odds`. Cuando el
archivo no las trae, todas las funciones de este módulo devuelven `None`
y el resto de la aplicación sigue funcionando igual (ver `has_market_odds`).

Este módulo sirve para dos cosas, ambas sobre datos históricos (nunca sobre
un partido futuro real, porque no existen cuotas para eso en el archivo):

1. `market_backtest`: qué tan bien predice el propio mercado los partidos
   de prueba, para comparar de forma justa contra Poisson y la regresión
   logística (misma idea que `poisson_model.backtest`).
2. `simulate_value_bets`: una simulación retrospectiva simple de apuestas
   de valor (value betting) — nunca una recomendación de apuesta real ni
   una garantía de resultados futuros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import CLASS_LABELS, get_logger

logger = get_logger(__name__)

MARKET_ODDS_COLUMNS = ["market_home_odds", "market_draw_odds", "market_away_odds"]
_ODDS_COL_BY_OUTCOME = {"H": "market_home_odds", "D": "market_draw_odds", "A": "market_away_odds"}


def has_market_odds(df: pd.DataFrame) -> bool:
    """Indica si `df` trae cuotas de mercado utilizables en al menos una fila."""
    if not all(c in df.columns for c in MARKET_ODDS_COLUMNS):
        return False
    return bool(df[MARKET_ODDS_COLUMNS].notna().all(axis=1).any())


def _valid_odds(home_odds, draw_odds, away_odds) -> bool:
    return all(o is not None and o == o and o > 1.0 for o in (home_odds, draw_odds, away_odds))


def implied_probabilities(home_odds: float, draw_odds: float, away_odds: float) -> dict[str, float] | None:
    """Convierte cuotas decimales 1X2 a probabilidades implícitas (%),
    quitando el margen de la casa de apuestas (overround) mediante
    normalización proporcional (método más simple y estándar de "devig").
    Devuelve None si alguna cuota falta o es inválida (<= 1.0)."""
    if not _valid_odds(home_odds, draw_odds, away_odds):
        return None
    raw = {"H": 1 / home_odds, "D": 1 / draw_odds, "A": 1 / away_odds}
    overround = sum(raw.values())
    if overround <= 0:
        return None
    return {k: round(v / overround * 100, 2) for k, v in raw.items()}


def overround_pct(home_odds: float, draw_odds: float, away_odds: float) -> float | None:
    """Margen de la casa de apuestas (%) implícito en las tres cuotas.
    Un mercado "justo" (sin margen) daría 0%; en la práctica suele rondar
    el 2-8% en las cuotas promedio de casas de apuestas importantes."""
    if not _valid_odds(home_odds, draw_odds, away_odds):
        return None
    return round((1 / home_odds + 1 / draw_odds + 1 / away_odds - 1) * 100, 2)


def _actual_outcome(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


@dataclass
class MarketBacktestMetrics:
    n_test: int
    accuracy: float | None
    log_loss: float | None
    brier_score: float | None
    avg_overround_pct: float | None


def market_backtest(rows: pd.DataFrame) -> MarketBacktestMetrics | None:
    """Evalúa qué tan bien predicen las cuotas de mercado los partidos de
    `rows`. `rows` debe tener las columnas de `MARKET_ODDS_COLUMNS` y
    `actual_outcome` (H/D/A) — mismo conjunto de prueba usado para evaluar
    Poisson o la regresión logística, para que la comparación sea justa.
    """
    if rows is None or rows.empty or not all(c in rows.columns for c in [*MARKET_ODDS_COLUMNS, "actual_outcome"]):
        return None

    eps = 1e-15
    correct = 0
    log_loss_sum = 0.0
    brier_sum = 0.0
    overrounds: list[float] = []
    n = 0

    for _, row in rows.iterrows():
        probs = implied_probabilities(row["market_home_odds"], row["market_draw_odds"], row["market_away_odds"])
        if probs is None:
            continue
        n += 1

        overround = overround_pct(row["market_home_odds"], row["market_draw_odds"], row["market_away_odds"])
        if overround is not None:
            overrounds.append(overround)

        actual = row["actual_outcome"]
        probs_frac = {k: v / 100 for k, v in probs.items()}
        predicted_label = max(probs_frac, key=probs_frac.get)
        if predicted_label == actual:
            correct += 1

        p_actual = max(probs_frac.get(actual, 0.0), eps)
        log_loss_sum += -np.log(p_actual)
        brier_sum += sum((probs_frac.get(c, 0.0) - (1.0 if c == actual else 0.0)) ** 2 for c in CLASS_LABELS)

    if n == 0:
        return None

    return MarketBacktestMetrics(
        n_test=n,
        accuracy=round(correct / n, 4),
        log_loss=round(log_loss_sum / n, 4),
        brier_score=round(brier_sum / n, 4),
        avg_overround_pct=round(float(np.mean(overrounds)), 2) if overrounds else None,
    )


@dataclass
class ValueBetSummary:
    edge_threshold_pct: float
    n_bets: int
    n_wins: int
    win_rate_pct: float | None
    roi_pct: float | None
    total_staked: float
    total_profit: float


def simulate_value_bets(
    rows: pd.DataFrame,
    model_probs_by_row: list[dict[str, float]],
    edge_threshold_pct: float = 5.0,
) -> ValueBetSummary | None:
    """Simulación retrospectiva de una estrategia simple de "apuesta de
    valor": por cada partido de `rows` y cada resultado (H/D/A), si la
    probabilidad del modelo supera a la probabilidad implícita del mercado
    por al menos `edge_threshold_pct` puntos, se simula una apuesta de 1
    unidad a ese resultado, a la cuota de mercado registrada en el archivo.

    `rows` y `model_probs_by_row` deben estar alineados 1 a 1 (misma
    cantidad de filas, mismo orden). `rows` debe tener las columnas de
    `MARKET_ODDS_COLUMNS` y `actual_outcome`.

    Esto NO es una recomendación de apuesta ni garantiza resultados
    futuros: es solo una forma de medir, en retrospectiva y sobre los
    datos disponibles, si el modelo habría tenido algún valor frente al
    mercado.
    """
    if rows is None or rows.empty or len(rows) != len(model_probs_by_row):
        return None
    if not all(c in rows.columns for c in [*MARKET_ODDS_COLUMNS, "actual_outcome"]):
        return None

    n_bets = n_wins = 0
    total_staked = 0.0
    total_profit = 0.0

    for (_, row), model_probs in zip(rows.iterrows(), model_probs_by_row):
        market_probs = implied_probabilities(row["market_home_odds"], row["market_draw_odds"], row["market_away_odds"])
        if market_probs is None:
            continue
        actual = row["actual_outcome"]

        for outcome in CLASS_LABELS:
            edge = model_probs.get(outcome, 0.0) - market_probs.get(outcome, 0.0)
            if edge < edge_threshold_pct:
                continue
            odds = row[_ODDS_COL_BY_OUTCOME[outcome]]
            if odds is None or odds != odds or odds <= 1.0:
                continue

            n_bets += 1
            total_staked += 1.0
            if outcome == actual:
                n_wins += 1
                total_profit += odds - 1
            else:
                total_profit -= 1

    if n_bets == 0:
        return ValueBetSummary(edge_threshold_pct, 0, 0, None, None, 0.0, 0.0)

    return ValueBetSummary(
        edge_threshold_pct=edge_threshold_pct,
        n_bets=n_bets,
        n_wins=n_wins,
        win_rate_pct=round(n_wins / n_bets * 100, 1),
        roi_pct=round(total_profit / total_staked * 100, 1),
        total_staked=round(total_staked, 2),
        total_profit=round(total_profit, 2),
    )


def build_rows_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataframe mínimo que requieren `market_backtest` y
    `simulate_value_bets` (cuotas + resultado real) a partir de un
    dataframe histórico completo, como el `test_df` de
    `poisson_model.chronological_split`."""
    if not all(c in df.columns for c in MARKET_ODDS_COLUMNS):
        return pd.DataFrame(columns=[*MARKET_ODDS_COLUMNS, "actual_outcome"])
    result = df[MARKET_ODDS_COLUMNS].copy()
    result["actual_outcome"] = [
        _actual_outcome(hg, ag) for hg, ag in zip(df["home_goals"], df["away_goals"])
    ]
    return result.reset_index(drop=True)
