"""Pruebas básicas para src/data_loader.py."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src import data_loader


def _football_data_csv_bytes() -> bytes:
    """CSV mínimo que imita el formato football-data.co.uk, incluyendo
    algunas columnas de cuotas de apuestas que deben descartarse."""
    content = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,"
        "HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,"
        "B365H,B365D,B365A,AHh,B365AHH,B365AHA,B365>2.5,B365<2.5\n"
        "E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,1,0,H,A Taylor,"
        "19,10,10,3,7,10,6,7,1,2,0,0,"
        "1.3,6,8.5,-1.5,1.83,2.03,1.36,3.2\n"
        "E0,16/08/2025,12:30,Aston Villa,Newcastle,0,0,D,0,0,D,C Pawson,"
        "3,16,3,3,13,11,3,6,1,1,1,0,"
        "2.25,3.5,2.9,-0.25,2.0,1.85,1.37,3.26\n"
    )
    return ("﻿" + content).encode("utf-8")  # BOM al inicio, como el archivo real


def test_detect_file_type_supported():
    assert data_loader.detect_file_type("partidos.csv") == "csv"
    assert data_loader.detect_file_type("partidos.xlsx") == "excel"
    assert data_loader.detect_file_type("partidos.xls") == "excel"


def test_detect_file_type_unsupported_raises():
    with pytest.raises(ValueError):
        data_loader.detect_file_type("partidos.txt")


def test_read_csv_drops_odds_columns_and_keeps_core_columns():
    file_obj = io.BytesIO(_football_data_csv_bytes())
    loaded = data_loader.read_csv(file_obj, "E0.csv")

    assert loaded.raw_df is not None
    assert len(loaded.raw_df) == 2

    # Las columnas de cuotas que NO son la referencia del mercado 1X2 deben
    # haberse descartado (hándicap asiático, over/under...).
    for odds_col in ["AHh", "B365AHH", "B365AHA", "B365>2.5", "B365<2.5"]:
        assert odds_col not in loaded.raw_df.columns
    assert set(loaded.dropped_odds_columns) == {"AHh", "B365AHH", "B365AHA", "B365>2.5", "B365<2.5"}

    # B365H/D/A sí se conservan: son la referencia del mercado (ver market_odds.py).
    for market_col in ["B365H", "B365D", "B365A"]:
        assert market_col in loaded.raw_df.columns

    # Las columnas relevantes deben conservarse.
    for core_col in ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "Referee", "HC", "HY"]:
        assert core_col in loaded.raw_df.columns


def test_read_csv_handles_bom_encoding():
    file_obj = io.BytesIO(_football_data_csv_bytes())
    loaded = data_loader.read_csv(file_obj, "E0.csv")
    # utf-8-sig debe eliminar el BOM del nombre de la primera columna.
    assert loaded.raw_df.columns[0] == "Div"


def test_dataset_overview_reports_basic_metrics():
    df = pd.DataFrame(
        {
            "HomeTeam": ["A", "B", "A"],
            "AwayTeam": ["B", "A", "B"],
            "FTHG": [1, 2, None],
            "FTAG": [0, 2, 1],
        }
    )
    overview = data_loader.dataset_overview(df)
    assert overview["n_rows"] == 3
    assert overview["n_columns"] == 4
    assert overview["missing_by_column"]["FTHG"]["count"] == 1
    assert overview["duplicate_rows"] == 0


def test_concat_multiple_files_orders_chronologically():
    season_2 = data_loader.LoadedFile(
        filename="s2.csv",
        file_type="csv",
        raw_df=pd.DataFrame({"Date": ["10/08/2025"], "HomeTeam": ["A"], "AwayTeam": ["B"], "FTHG": [1], "FTAG": [1]}),
    )
    season_1 = data_loader.LoadedFile(
        filename="s1.csv",
        file_type="csv",
        raw_df=pd.DataFrame({"Date": ["10/08/2024"], "HomeTeam": ["C"], "AwayTeam": ["D"], "FTHG": [2], "FTAG": [0]}),
    )
    # Se cargan "fuera de orden" a propósito para verificar que se reordenan.
    combined = data_loader.concat_multiple_files([season_2, season_1])
    assert list(combined["HomeTeam"]) == ["C", "A"]
