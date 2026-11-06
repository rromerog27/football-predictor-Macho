"""Pruebas sin red del historial del valor (src/value_history.py)."""

import numpy as np
import pandas as pd
import pytest

from src import value_history as vh


def _matched(n=40, edge_home=0.12, home_wins=0.5, odds_home=2.0, over_edge=None, over_rate=0.5):
    """Partidos de validación sintéticos: el modelo da `edge_home` puntos más que la casa al local."""
    rng = np.random.default_rng(0)
    odds = np.tile([odds_home, 3.5, 4.0], (n, 1))
    inv = 1 / odds
    pm = inv / inv.sum(axis=1, keepdims=True)
    p = pm.copy()
    p[:, 0] += edge_home
    p[:, 1] -= edge_home / 2
    p[:, 2] -= edge_home / 2
    wins = np.arange(n) < round(n * home_wins)
    goals = np.where(np.arange(n) < round(n * over_rate), 3, 1)
    df = pd.DataFrame({"p_model_H": p[:, 0], "p_model_D": p[:, 1], "p_model_A": p[:, 2],
                       "odds_h": odds[:, 0], "odds_d": odds[:, 1], "odds_a": odds[:, 2],
                       "result": np.where(wins, "H", "A"), "hg": goals, "ag": 0,
                       "datetime": pd.date_range("2026-01-01", periods=n), "home": "L", "away": "V"})
    if over_edge is not None:
        df["odds_over25"], df["odds_under25"] = 1.9, 1.9
        df["over25"] = 0.5 + over_edge
    return df.sample(frac=1, random_state=int(rng.integers(100))).reset_index(drop=True)


def test_record_for_the_home_value_band():
    h = vh.build(_matched(n=40, edge_home=0.12, home_wins=0.5, odds_home=2.0))
    rec = h.for_outcome(0, 0.15)
    assert rec is not None and rec.n == 40 and rec.wins == 20
    assert rec.roi == pytest.approx(0.0) and rec.verdict == "sin señal" and "quedó en empate en 40" in rec.text
    # sin apuestas en el tramo de 5–10 pts: cae al historial de 5+ pts del local
    assert h.for_outcome(0, 0.07) == h.any_band[0]
    assert h.for_outcome(1, 0.20) is None  # nunca hubo valor al empate


def test_verdicts_and_texts():
    bad = vh.build(_matched(n=30, home_wins=0.3, odds_home=2.0)).for_outcome(0, 0.12)
    good = vh.build(_matched(n=30, home_wins=0.7, odds_home=2.0)).for_outcome(0, 0.12)
    assert bad.verdict == "en contra" and bad.roi == pytest.approx(-0.4)
    assert good.verdict == "a favor" and "ganó 40% en 30 partidos" in good.text


def test_too_few_bets_means_no_history():
    assert vh.build(_matched(n=vh.MIN_BETS - 1)).for_outcome(0, 0.12) is None


def test_over_under_record():
    h = vh.build(_matched(n=40, over_edge=-0.10, over_rate=0.6))  # el modelo veía el menos y salía el más
    rec = h.for_over_under("menos")
    assert rec is not None and rec.n == 40 and rec.wins == 16 and rec.verdict == "en contra"
    assert h.for_over_under("más") is None


def test_favorites_rule_record():
    df = _matched(n=40, edge_home=0.12, home_wins=0.8, odds_home=1.4)  # el local es favorito claro del modelo
    rec = vh.build(df).favorites
    assert rec is not None and rec.n == 40 and rec.roi == pytest.approx(0.8 * 0.4 - 0.2)
