"""Pruebas básicas para src/feature_engineering.py, src/prediction_model.py
y src/poisson_model.py (Fase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import feature_engineering, poisson_model, prediction_model


def _synthetic_league(n_matches: int = 220, n_teams: int = 8, seed: int = 42) -> pd.DataFrame:
    """Genera una liga sintética con resultados influidos por una fuerza de
    equipo fija, para que el modelo tenga una señal real que aprender
    (y las pruebas no dependan de resultados puramente aleatorios)."""
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(n_teams)]
    strengths = {team: rng.uniform(0.5, 2.5) for team in teams}

    rows = []
    start_date = pd.Timestamp("2023-08-01")
    for i in range(n_matches):
        home, away = rng.choice(teams, size=2, replace=False)
        home_lambda = max(0.3, strengths[home] * 1.15)
        away_lambda = max(0.3, strengths[away] * 0.9)
        home_goals = rng.poisson(home_lambda)
        away_goals = rng.poisson(away_lambda)
        rows.append(
            {
                "date": start_date + pd.Timedelta(days=i * 2),
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@pytest.fixture(scope="module")
def big_df() -> pd.DataFrame:
    return _synthetic_league(n_matches=220)


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return _synthetic_league(n_matches=15, n_teams=4, seed=1)


# --------------------------------------------------------------------------
# feature_engineering.py
# --------------------------------------------------------------------------


def test_build_feature_row_returns_all_expected_columns(big_df):
    features = feature_engineering.build_feature_row(big_df, "Team0", "Team1", cutoff_date=None)
    assert set(features.keys()) == set(feature_engineering.FEATURE_COLUMNS)


def test_build_feature_row_respects_cutoff_no_future_leakage(big_df):
    cutoff = big_df["date"].iloc[50]
    features = feature_engineering.build_feature_row(big_df, "Team0", "Team1", cutoff_date=cutoff)
    # Debe ser idéntico a calcularlo manualmente sobre el subconjunto anterior al corte.
    prior = big_df[big_df["date"] < cutoff]
    features_manual = feature_engineering.build_feature_row(prior, "Team0", "Team1", cutoff_date=None)
    assert features["home_matches_played_overall"] == features_manual["home_matches_played_overall"]


def test_build_training_dataset_skips_early_matches_without_history(big_df):
    X, y, meta, skipped = feature_engineering.build_training_dataset(big_df, min_prior_matches=3)
    assert len(X) == len(y) == len(meta)
    assert skipped > 0
    assert len(X) + skipped == len(big_df)
    assert set(y.unique()).issubset({"H", "D", "A"})


def test_build_training_dataset_meta_is_chronological(big_df):
    _, _, meta, _ = feature_engineering.build_training_dataset(big_df)
    assert meta["date"].is_monotonic_increasing


# --------------------------------------------------------------------------
# prediction_model.py — entrenamiento
# --------------------------------------------------------------------------


def test_train_logistic_model_insufficient_data_returns_untrained(tiny_df):
    result = prediction_model.train_logistic_model(tiny_df)
    assert result.trained is False
    assert result.model is None
    assert any("insuficient" in w.lower() or "requieren" in w.lower() for w in result.warnings)


def test_train_logistic_model_sufficient_data_trains_successfully(big_df):
    result = prediction_model.train_logistic_model(big_df)
    assert result.trained is True
    assert result.model is not None
    assert result.metrics is not None
    assert 0.0 <= result.metrics.accuracy <= 1.0
    assert result.metrics.log_loss > 0
    assert len(result.metrics.confusion_matrix) == 3


def test_train_logistic_model_feature_importance_covers_all_features(big_df):
    result = prediction_model.train_logistic_model(big_df)
    assert set(result.feature_importance.keys()) == set(feature_engineering.FEATURE_COLUMNS)
    assert all(v >= 0 for v in result.feature_importance.values())


def test_predict_logistic_probabilities_sum_to_100(big_df):
    result = prediction_model.train_logistic_model(big_df)
    probs = prediction_model.predict_logistic(result, big_df, "Team0", "Team1")
    assert probs is not None
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert total == pytest.approx(100.0, abs=0.5)


def test_predict_logistic_returns_none_when_untrained(tiny_df):
    result = prediction_model.train_logistic_model(tiny_df)
    assert prediction_model.predict_logistic(result, tiny_df, "Team0", "Team1") is None


# --------------------------------------------------------------------------
# poisson_model.py — backtest y split cronológico
# --------------------------------------------------------------------------


def test_chronological_split_keeps_recent_matches_in_test(big_df):
    train, test = poisson_model.chronological_split(big_df, test_fraction=0.2)
    assert len(train) + len(test) == len(big_df)
    assert train["date"].max() <= test["date"].min()


def test_poisson_backtest_produces_valid_metrics(big_df):
    train, test = poisson_model.chronological_split(big_df, test_fraction=0.2)
    metrics = poisson_model.backtest(train, test)
    assert metrics.n_test == len(test)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.log_loss > 0


# --------------------------------------------------------------------------
# prediction_model.py — combinación Poisson + regresión logística
# --------------------------------------------------------------------------


def test_combine_predictions_falls_back_to_poisson_only_when_no_logistic(big_df):
    pred = poisson_model.predict_match(big_df, "Team0", "Team1")
    combined = prediction_model.combine_predictions(
        poisson_pred=pred,
        logistic_probs=None,
        poisson_backtest=None,
        logistic_metrics=None,
        stat_completeness_pct=50.0,
    )
    assert combined.weight_poisson == 100.0
    assert combined.weight_logistic == 0.0
    assert combined.prob_home_win == pred.prob_home_win


def test_combine_predictions_weights_better_model_more_heavily(big_df):
    pred = poisson_model.predict_match(big_df, "Team0", "Team1")
    train, test = poisson_model.chronological_split(big_df, test_fraction=0.2)
    poisson_bt = poisson_model.backtest(train, test)

    logistic_metrics = prediction_model.ValidationMetrics(
        n_train=100,
        n_test=20,
        accuracy=0.8,
        precision_macro=0.8,
        recall_macro=0.8,
        f1_macro=0.8,
        log_loss=poisson_bt.log_loss / 2,  # el doble de bueno que Poisson
        brier_score=0.1,
        confusion_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    logistic_probs = {"home_win": 60.0, "draw": 20.0, "away_win": 20.0}

    combined = prediction_model.combine_predictions(
        poisson_pred=pred,
        logistic_probs=logistic_probs,
        poisson_backtest=poisson_bt,
        logistic_metrics=logistic_metrics,
        stat_completeness_pct=50.0,
    )
    assert combined.weight_logistic > combined.weight_poisson
    assert combined.prob_home_win + combined.prob_draw + combined.prob_away_win == pytest.approx(100.0, abs=0.5)


def test_combine_predictions_confidence_is_valid_label(big_df):
    pred = poisson_model.predict_match(big_df, "Team0", "Team1")
    combined = prediction_model.combine_predictions(
        poisson_pred=pred,
        logistic_probs=None,
        poisson_backtest=None,
        logistic_metrics=None,
        stat_completeness_pct=100.0,
    )
    assert combined.confidence in {"alta", "media", "baja"}
