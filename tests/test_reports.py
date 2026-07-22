"""Pruebas básicas para src/report_generator.py y src/prediction_history.py
(Fase 3)."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src import prediction_history, report_generator, statistics


@pytest.fixture
def sample_df() -> pd.DataFrame:
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


# --------------------------------------------------------------------------
# report_generator.py
# --------------------------------------------------------------------------


def test_stats_table_to_csv_has_one_row_per_team(sample_df):
    csv_bytes = report_generator.stats_table_to_csv(sample_df)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert set(df["Equipo"]) == {"A", "B", "C"}
    assert "% Victorias" in df.columns


def test_prediction_to_csv_roundtrip():
    record = {"home_team": "A", "away_team": "B", "prob_home_win": 55.5}
    csv_bytes = report_generator.prediction_to_csv(record)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert df.iloc[0]["home_team"] == "A"
    assert df.iloc[0]["prob_home_win"] == 55.5


def test_score_matrix_to_csv_contains_values():
    matrix = pd.DataFrame([[10.0, 5.0], [3.0, 1.0]], index=["0", "1"], columns=["0", "1"])
    csv_bytes = report_generator.score_matrix_to_csv(matrix)
    text = csv_bytes.decode("utf-8-sig")
    assert "10.0" in text and "Goles local" in text


def test_variables_summary_to_csv_lists_mapped_fields():
    mapping = {"home_team": "HomeTeam", "away_team": "AwayTeam", "home_xg": None}
    labels = {"home_team": "Equipo local", "away_team": "Equipo visitante", "home_xg": "xG local"}
    csv_bytes = report_generator.variables_summary_to_csv(mapping, labels)
    text = csv_bytes.decode("utf-8-sig")
    assert "HomeTeam" in text
    assert "No disponible" in text


def test_variables_summary_to_csv_includes_feature_importance():
    mapping = {"home_team": "HomeTeam"}
    labels = {"home_team": "Equipo local"}
    csv_bytes = report_generator.variables_summary_to_csv(mapping, labels, feature_importance={"home_form_ppg": 0.42})
    text = csv_bytes.decode("utf-8-sig")
    assert "home_form_ppg" in text
    assert "Importancia" in text


def test_comparison_to_excel_has_expected_sheets(sample_df):
    home_profile = statistics.build_team_profile(sample_df, "A", 5)
    away_profile = statistics.build_team_profile(sample_df, "B", 5)
    excel_bytes = report_generator.comparison_to_excel("A", "B", home_profile, away_profile)

    sheets = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
    assert set(sheets.keys()) == {"General", "Como local", "Como visitante", "Forma reciente"}
    assert "A" in sheets["General"].columns


def test_build_html_report_contains_key_sections_and_escapes_html():
    context = {
        "file_name": "E0.csv",
        "analysis_date": "2026-07-20 10:00",
        "competition": "E0",
        "season_labels": "2025-2026",
        "n_matches": 380,
        "period": "2025-08-15 → 2026-05-24",
        "home_team": "<script>alert(1)</script>",
        "away_team": "Bournemouth",
        "model_choice": "Poisson",
        "prob_home_win": 55.5,
        "prob_draw": 20.9,
        "prob_away_win": 23.6,
        "expected_home_goals": 2.1,
        "expected_away_goals": 1.3,
        "top_score": "2-1",
        "confidence": "alta",
        "explanation": "Explicación de prueba.",
        "warnings": ["Advertencia de prueba"],
        "comparison_rows": [{"Métrica": "Partidos jugados", "A": 5, "B": 5}],
        "score_matrix": pd.DataFrame([[10.0, 5.0]], index=["0"], columns=["0", "1"]),
    }
    report = report_generator.build_html_report(context)
    assert "<script>alert(1)</script>" not in report  # debe estar escapado
    assert "&lt;script&gt;" in report
    assert "Explicación de prueba." in report
    assert "Advertencia de prueba" in report
    assert "Ninguna probabilidad" in report


# --------------------------------------------------------------------------
# prediction_history.py
# --------------------------------------------------------------------------


@pytest.fixture
def history_env(tmp_path, monkeypatch):
    history_path = tmp_path / "prediction_history.csv"
    monkeypatch.setattr(prediction_history, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prediction_history, "HISTORY_PATH", history_path)
    return history_path


def test_load_history_creates_empty_file_with_columns(history_env):
    df = prediction_history.load_history()
    assert df.empty
    assert list(df.columns) == prediction_history.HISTORY_COLUMNS
    assert history_env.exists()


def test_append_prediction_assigns_incrementing_ids(history_env):
    id1 = prediction_history.append_prediction({"home_team": "A", "away_team": "B", "predicted_outcome": "H"})
    id2 = prediction_history.append_prediction({"home_team": "C", "away_team": "D", "predicted_outcome": "A"})
    assert id1 == 1
    assert id2 == 2

    df = prediction_history.load_history()
    assert len(df) == 2
    assert df.iloc[0]["model_version"] == prediction_history.MODEL_VERSION


def test_pending_records_excludes_resolved(history_env):
    prediction_history.append_prediction({"home_team": "A", "away_team": "B", "predicted_outcome": "H"})
    record_id = prediction_history.append_prediction({"home_team": "C", "away_team": "D", "predicted_outcome": "A"})
    prediction_history.update_actual_result(record_id, actual_home_goals=0, actual_away_goals=2)

    df = prediction_history.load_history()
    pending = prediction_history.pending_records(df)
    assert len(pending) == 1
    assert pending.iloc[0]["home_team"] == "A"


def test_update_actual_result_computes_correctness(history_env):
    record_id = prediction_history.append_prediction({"home_team": "A", "away_team": "B", "predicted_outcome": "H"})
    prediction_history.update_actual_result(record_id, actual_home_goals=2, actual_away_goals=0)

    df = prediction_history.load_history()
    row = df[df["id"] == record_id].iloc[0]
    assert row["actual_outcome"] == "H"
    assert str(row["correct"]) in ("True", "1")


def test_update_actual_result_unknown_id_raises(history_env):
    with pytest.raises(ValueError):
        prediction_history.update_actual_result(999, 1, 0)


def test_compute_performance_with_no_resolved_predictions(history_env):
    prediction_history.append_prediction({"home_team": "A", "away_team": "B", "predicted_outcome": "H"})
    df = prediction_history.load_history()
    perf = prediction_history.compute_performance(df)
    assert perf["total_predictions"] == 1
    assert perf["resolved"] == 0
    assert perf["accuracy"] is None


def test_compute_performance_accuracy_and_by_model(history_env):
    id1 = prediction_history.append_prediction({"home_team": "A", "away_team": "B", "predicted_outcome": "H", "model_used": "Poisson"})
    id2 = prediction_history.append_prediction({"home_team": "C", "away_team": "D", "predicted_outcome": "A", "model_used": "Poisson"})
    id3 = prediction_history.append_prediction({"home_team": "E", "away_team": "F", "predicted_outcome": "H", "model_used": "Combinado"})

    prediction_history.update_actual_result(id1, 2, 0)  # acierto (H)
    prediction_history.update_actual_result(id2, 1, 1)  # fallo (predijo A, fue D)
    prediction_history.update_actual_result(id3, 3, 0)  # acierto (H)

    df = prediction_history.load_history()
    perf = prediction_history.compute_performance(df)
    assert perf["resolved"] == 3
    assert perf["accuracy"] == pytest.approx(66.7, abs=0.1)
    assert perf["by_model"]["Poisson"]["n"] == 2
    assert perf["by_model"]["Combinado"]["accuracy"] == 100.0
