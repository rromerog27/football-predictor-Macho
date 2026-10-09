"""Pruebas de src/performance.py: aciertos, calibración, historial y regla de los picks."""

import numpy as np
import pandas as pd
import pytest

from src import performance as perf


def _frame(probs, results, odds=None, start="2026-09-01"):
    probs = np.asarray(probs, float)
    df = pd.DataFrame({
        "datetime": pd.date_range(start, periods=len(probs), freq="D"),
        "home": [f"L{i}" for i in range(len(probs))], "away": [f"V{i}" for i in range(len(probs))],
        "result": results, "hg": 1.0, "ag": 0.0,
        **{f"p_model_{c}": probs[:, i] for i, c in enumerate(perf.CLASSES)},
    })
    if odds is not None:
        odds = np.asarray(odds, float)
        df[["odds_h", "odds_d", "odds_a"]] = odds
    return df


def test_hit_rate_and_favorites():
    df = _frame([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.1, 0.2, 0.7], [0.66, 0.2, 0.14]], ["H", "A", "A", "D"])
    p, y = perf.model_probs(df), perf.outcomes(df)
    assert perf.hit_rate(p, y) == perf.Rate(2, 4)
    rate, mean = perf.favorites(p, y)  # favoritos ≥65%: partidos 0, 2 y 3; ganaron 0 y 2
    assert rate == perf.Rate(2, 3) and mean == pytest.approx((0.7 + 0.7 + 0.66) / 3)


def test_calibration_pools_outcomes_and_drops_small_bins():
    rng = np.random.default_rng(0)
    p = np.tile([0.6, 0.25, 0.15], (400, 1))
    y = rng.choice(3, size=400, p=[0.6, 0.25, 0.15])
    cal = perf.calibration(p, y)
    assert list(cal["lo"]) == [0.1, 0.2, 0.6]  # tres tramos con datos (0.15, 0.25, 0.6)
    assert cal["n"].tolist() == [400, 400, 400]
    assert cal.loc[cal["lo"] == 0.6, "observed"].item() == pytest.approx(0.6, abs=0.06)
    assert perf.calibration(p[:5], y[:5]).empty  # menos de CALIBRATION_MIN_N por tramo


def test_history_is_newest_first_with_pick_and_hit():
    df = _frame([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]], ["H", "H"])
    h = perf.history(df)
    assert h["home"].tolist() == ["L1", "L0"]
    assert h["pick"].tolist() == [2, 0] and h["hit"].tolist() == [False, True]
    assert perf.history(df.iloc[0:0]).empty


def test_pick_rule():
    odds = np.array([1.4, 4.5, 7.0])
    pk = perf.find_pick(np.array([0.75, 0.15, 0.10]), np.array([0.70, 0.18, 0.12]), odds)
    assert pk.outcome == 0 and pk.value == pytest.approx(0.75 * 1.4 - 1)
    assert perf.find_pick(np.array([0.60, 0.25, 0.15]), np.array([0.55, 0.25, 0.20]), odds) is None  # no es claro
    assert perf.find_pick(np.array([0.70, 0.18, 0.12]), np.array([0.75, 0.15, 0.10]), odds) is None  # mercado da más
    assert perf.find_avoid(np.array([0.60, 0.25, 0.15]), np.array([0.72, 0.18, 0.10])) == 0
    assert perf.find_avoid(np.array([0.68, 0.2, 0.12]), np.array([0.72, 0.18, 0.10])) is None


def test_pick_record_profit_at_closing_odds():
    probs = [[0.8, 0.12, 0.08], [0.7, 0.2, 0.1], [0.4, 0.3, 0.3]]
    odds = [[1.40, 4.5, 8.0], [1.50, 4.0, 6.0], [2.2, 3.2, 3.4]]
    rec = perf.pick_record(_frame(probs, ["H", "A", "H"], odds))
    assert len(rec) == 2  # el tercero no es favorito claro
    assert rec["won"].tolist() == [False, True]  # del más reciente al más viejo
    assert rec["profit"].sum() == pytest.approx(0.40 - 1)
    assert perf.pick_record(_frame(probs, ["H", "A", "H"], odds).iloc[0:0]).empty
