"""Pruebas sin red del respaldo de ESPN cuando Understat no responde."""

import numpy as np
import pytest

from src import competitions as comps
from src import match_model as mm
from tests.test_modelo_prediccion import _fake_league


def _espn_like(league: mm.LeagueData):
    """Los partidos de la liga de prueba con la forma de los de ESPN (ids espn:, sin xG real)."""
    m = league.matches.copy()
    m["id"] = m["id"].str.replace("us:", "espn:", regex=False)
    m["home_id"], m["away_id"] = m["home"], m["away"]
    m["home_abbr"], m["away_abbr"] = m["home"].str[:3], m["away"].str[:3]
    m["venue"] = None
    for col in mm.STAT_COLUMNS:
        m[col] = np.nan
    return m


@pytest.fixture
def understat_down(monkeypatch):
    calls = []

    def down(*args, **kwargs):
        calls.append(args)
        raise mm.PredictionError("No se pudo descargar https://understat.com/...: timeout")

    frame = _espn_like(_fake_league())
    monkeypatch.setattr(comps.understat_source, "load", down)
    monkeypatch.setattr(comps, "_fetch_espn_years", lambda *a, **k: (frame, []))
    monkeypatch.setattr(comps, "_understat_down_until", 0.0)
    return calls


def test_understat_down_falls_back_to_espn(understat_down):
    messages = []
    data = comps.load_competition(comps.BY_CODE["eng.1"], progress=messages.append, with_market=False)
    assert not comps.from_understat(data) and data.signal_name == "xG aproximado (tiros)"
    assert any("Understat no responde" in m for m in messages)
    # Las demás ligas de Understat no vuelven a esperarlo durante UNDERSTAT_RETRY_S.
    comps.load_competition(comps.BY_CODE["esp.1"], with_market=False)
    assert len(understat_down) == 1


def test_fixtures_of_an_espn_loaded_understat_league_are_espn_rows(understat_down):
    data = comps.load_competition(comps.BY_CODE["eng.1"], with_market=False)
    day = data.focus_matches[~data.focus_matches["played"]]
    rows = comps.fixture_rows_for_model(comps.BY_CODE["eng.1"], data, day)
    assert rows["id"].tolist() == day["id"].tolist()
