"""Pruebas básicas para src/statistics.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src import statistics


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Cinco partidos sintéticos, ya ordenados cronológicamente, con
    nombres de columna canónicos (como los deja data_cleaner.clean_data)."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"]
            ),
            "home_team": ["A", "B", "A", "C", "A"],
            "away_team": ["B", "A", "C", "A", "B"],
            "home_goals": [2, 1, 0, 3, 1],
            "away_goals": [0, 1, 1, 3, 0],
        }
    )


def test_get_team_list(sample_df):
    assert statistics.get_team_list(sample_df) == ["A", "B", "C"]


def test_validate_team_selection_same_team(sample_df):
    errors = statistics.validate_team_selection(sample_df, "A", "A")
    assert any("diferentes" in e for e in errors)


def test_validate_team_selection_unknown_team(sample_df):
    errors = statistics.validate_team_selection(sample_df, "A", "Z")
    assert any("Z" in e for e in errors)


def test_validate_team_selection_valid(sample_df):
    assert statistics.validate_team_selection(sample_df, "A", "B") == []


def test_compute_team_record_overall(sample_df):
    record = statistics.compute_team_record(sample_df, "A", venue="all")
    assert record.played == 5
    assert (record.wins, record.draws, record.losses) == (2, 2, 1)
    assert record.win_pct == 40.0
    assert record.draw_pct == 40.0
    assert record.avg_goals_for == 1.4
    assert record.avg_goals_against == 1.0
    assert record.clean_sheet_pct == 40.0
    assert record.failed_to_score_pct == 20.0
    assert record.both_teams_scored_pct == 40.0


def test_compute_team_record_home_only(sample_df):
    record = statistics.compute_team_record(sample_df, "A", venue="home")
    assert record.played == 3  # partidos 1, 3 y 5


def test_compute_team_record_no_matches_returns_zero(sample_df):
    record = statistics.compute_team_record(sample_df, "NonExistent", venue="all")
    assert record.played == 0
    assert record.win_pct is None


def test_compute_recent_form_most_recent_first(sample_df):
    form = statistics.compute_recent_form(sample_df, "A", n=3)
    assert form.n_available == 3
    assert form.results == ["W", "D", "L"]
    assert form.points == 4
    assert form.points_per_game == pytest.approx(1.33, abs=0.01)


def test_head_to_head_returns_only_matches_between_both_teams(sample_df):
    h2h = statistics.head_to_head(sample_df, "A", "B")
    assert len(h2h) == 3  # partidos 1 (A-B), 2 (B-A) y 5 (A-B)


def test_compute_dataset_summary(sample_df):
    summary = statistics.compute_dataset_summary(sample_df)
    assert summary.n_matches == 5
    assert summary.n_teams == 3
    assert summary.home_win_pct == 40.0
    assert summary.draw_pct == 40.0
    assert summary.away_win_pct == 20.0
    assert summary.avg_goals_per_match == 2.4


def test_compute_dataset_summary_empty_df():
    empty = pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    summary = statistics.compute_dataset_summary(empty)
    assert summary.n_matches == 0
    assert summary.sufficient_data is False
