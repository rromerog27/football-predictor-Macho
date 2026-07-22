"""Pruebas básicas para src/market_odds.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src import market_odds


# --------------------------------------------------------------------------
# implied_probabilities / overround_pct
# --------------------------------------------------------------------------


def test_implied_probabilities_removes_overround_and_sums_100():
    # Cuotas típicas con margen de casa (overround > 100%).
    probs = market_odds.implied_probabilities(2.0, 3.5, 4.0)
    assert probs is not None
    total = sum(probs.values())
    assert total == pytest.approx(100.0, abs=0.05)  # redondeo a 2 decimales por resultado
    # El favorito (cuota más baja) debe tener la mayor probabilidad.
    assert probs["H"] > probs["D"] > probs["A"]


def test_implied_probabilities_invalid_odds_returns_none():
    assert market_odds.implied_probabilities(None, 3.5, 4.0) is None
    assert market_odds.implied_probabilities(1.0, 3.5, 4.0) is None
    assert market_odds.implied_probabilities(0.5, 3.5, 4.0) is None
    assert market_odds.implied_probabilities(float("nan"), 3.5, 4.0) is None


def test_overround_pct_positive_for_typical_bookmaker_odds():
    overround = market_odds.overround_pct(2.0, 3.5, 4.0)
    assert overround is not None
    assert overround > 0


def test_overround_pct_zero_for_fair_odds():
    # 1/2 + 1/4 + 1/4 = 1.0 exacto -> mercado "justo", sin margen.
    overround = market_odds.overround_pct(2.0, 4.0, 4.0)
    assert overround == pytest.approx(0.0, abs=0.01)


# --------------------------------------------------------------------------
# has_market_odds / build_rows_from_df
# --------------------------------------------------------------------------


def test_has_market_odds_true_when_present():
    df = pd.DataFrame(
        {
            "market_home_odds": [2.0, 1.5],
            "market_draw_odds": [3.5, 4.0],
            "market_away_odds": [4.0, 6.0],
        }
    )
    assert market_odds.has_market_odds(df)


def test_has_market_odds_false_when_missing_columns():
    df = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
    assert not market_odds.has_market_odds(df)


def test_build_rows_from_df_computes_actual_outcome():
    df = pd.DataFrame(
        {
            "home_goals": [2, 0, 1],
            "away_goals": [0, 0, 1],
            "market_home_odds": [1.8, 3.0, 2.5],
            "market_draw_odds": [3.5, 3.2, 3.1],
            "market_away_odds": [4.5, 2.5, 2.8],
        }
    )
    rows = market_odds.build_rows_from_df(df)
    assert list(rows["actual_outcome"]) == ["H", "D", "D"]


# --------------------------------------------------------------------------
# market_backtest
# --------------------------------------------------------------------------


def test_market_backtest_perfect_when_market_always_favors_actual_winner():
    rows = pd.DataFrame(
        {
            "market_home_odds": [1.2, 10.0, 10.0],
            "market_draw_odds": [8.0, 1.2, 10.0],
            "market_away_odds": [10.0, 8.0, 1.2],
            "actual_outcome": ["H", "D", "A"],
        }
    )
    metrics = market_odds.market_backtest(rows)
    assert metrics is not None
    assert metrics.n_test == 3
    assert metrics.accuracy == 1.0
    assert metrics.log_loss < 0.3  # muy confiado y siempre acertado -> log loss bajo
    assert metrics.avg_overround_pct is not None


def test_market_backtest_returns_none_without_required_columns():
    rows = pd.DataFrame({"home_team": ["A"]})
    assert market_odds.market_backtest(rows) is None


def test_market_backtest_returns_none_for_empty_rows():
    rows = pd.DataFrame(columns=[*market_odds.MARKET_ODDS_COLUMNS, "actual_outcome"])
    assert market_odds.market_backtest(rows) is None


# --------------------------------------------------------------------------
# simulate_value_bets
# --------------------------------------------------------------------------


def test_simulate_value_bets_flags_and_wins_a_clear_edge():
    # El mercado da a "H" ~28.6% implícito (cuota 3.5 sin overround exacto);
    # el modelo dice 60% -> edge amplio, se apuesta y el resultado es H (gana).
    rows = pd.DataFrame(
        {
            "market_home_odds": [3.5],
            "market_draw_odds": [3.5],
            "market_away_odds": [3.5],
            "actual_outcome": ["H"],
        }
    )
    model_probs = [{"H": 60.0, "D": 20.0, "A": 20.0}]
    summary = market_odds.simulate_value_bets(rows, model_probs, edge_threshold_pct=5.0)
    assert summary is not None
    assert summary.n_bets == 1
    assert summary.n_wins == 1
    assert summary.win_rate_pct == 100.0
    assert summary.total_profit == pytest.approx(2.5, abs=0.01)  # cuota 3.5 -> ganancia neta 2.5
    assert summary.roi_pct == pytest.approx(250.0, abs=0.1)


def test_simulate_value_bets_no_bets_when_edge_below_threshold():
    rows = pd.DataFrame(
        {
            "market_home_odds": [2.0],
            "market_draw_odds": [3.5],
            "market_away_odds": [4.0],
            "actual_outcome": ["H"],
        }
    )
    # Probabilidades del modelo casi idénticas al mercado -> sin edge relevante.
    market_probs = market_odds.implied_probabilities(2.0, 3.5, 4.0)
    summary = market_odds.simulate_value_bets(rows, [market_probs], edge_threshold_pct=5.0)
    assert summary is not None
    assert summary.n_bets == 0
    assert summary.win_rate_pct is None
    assert summary.roi_pct is None


def test_simulate_value_bets_losing_bet_reduces_profit():
    rows = pd.DataFrame(
        {
            "market_home_odds": [3.5],
            "market_draw_odds": [3.5],
            "market_away_odds": [3.5],
            "actual_outcome": ["A"],  # el modelo apuesta a H y pierde
        }
    )
    model_probs = [{"H": 60.0, "D": 20.0, "A": 20.0}]
    summary = market_odds.simulate_value_bets(rows, model_probs, edge_threshold_pct=5.0)
    assert summary.n_bets == 1
    assert summary.n_wins == 0
    assert summary.total_profit == pytest.approx(-1.0, abs=0.01)
    assert summary.roi_pct == pytest.approx(-100.0, abs=0.1)


def test_simulate_value_bets_returns_none_on_length_mismatch():
    rows = pd.DataFrame(
        {
            "market_home_odds": [2.0, 2.0],
            "market_draw_odds": [3.5, 3.5],
            "market_away_odds": [4.0, 4.0],
            "actual_outcome": ["H", "A"],
        }
    )
    assert market_odds.simulate_value_bets(rows, [{"H": 50.0, "D": 30.0, "A": 20.0}]) is None
