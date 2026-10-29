"""Pruebas sin red de las copas nacionales: registro, campo neutral deducido del estadio y partidos de
equipos sin datos (amateurs) sin pronóstico."""

import pandas as pd
import pytest

from src import competitions as comps
from src import espn_source
from src import match_model as mm
from tests.test_markets import _event
from tests.test_modelo_prediccion import _fake_league


def test_domestic_cups_registry():
    cups = [c for c in comps.COMPETITIONS if c.region == "Copas nacionales"]
    assert len(cups) >= 10 and "Copas nacionales" in comps.REGIONS
    for c in cups:
        assert c.is_cup
        country = c.code.split(".")[0]
        assert f"{country}.1" in c.pool  # la primera división del país siempre alimenta la copa
        assert all(p.split(".")[0] == country for p in c.pool)
    assert len({c.code for c in comps.COMPETITIONS}) == len(comps.COMPETITIONS)
    assert all(code in comps.BY_CODE for code in comps.DEFAULT_CODES)
    assert comps.resolve_competition("carabao").code == "eng.league_cup"


def test_espn_venue_is_kept():
    event = _event()
    event["competitions"][0]["venue"] = {"id": 55, "fullName": "Estadio"}
    row = espn_source.events_frame([espn_source.trim_event(event)], "x").iloc[0]
    assert row["venue"] == "55"
    assert espn_source.events_frame([espn_source.trim_event(_event())], "x").iloc[0]["venue"] is None


def _match(home, away, venue, competition="liga", neutral=False, played=True):
    return {"competition": competition, "home": home, "away": away, "venue": venue, "neutral": neutral,
            "played": played}


def test_mark_neutral_by_venue():
    rows = [_match("A", "X", "1")] * 3 + [_match("B", "X", "2")] * 3 + [
        _match("A", "B", "9", "copa"),  # final en otro estadio
        _match("A", "B", "1", "copa"),  # en casa de A
        _match("B", "A", "1", "copa"),  # ESPN pone de local a B, pero se juega en casa de A
        _match("Amateur", "A", "7", "copa"),  # no se conoce el estadio del amateur
        _match("A", "B", None, "copa"),
        _match("A", "B", "9", "liga", played=False),  # fuera de las competiciones indicadas
    ] + [_match("A", "C", "9", "copa")] * 2 + [_match("A", "C", "8", "liga", played=False)] * 3
    matches = pd.DataFrame(rows)
    # Los partidos de copa (finales "de local" en el estadio neutral) y los no jugados no hacen estadio.
    venues = comps.home_venues(matches, frozenset({"copa"}))
    assert venues == {"A": frozenset({"1"}), "B": frozenset({"2"})}
    out = comps.mark_neutral(matches, venues, frozenset({"copa"}))
    assert out["neutral"].tolist()[6:12] == [True, False, True, False, False, False]
    assert not matches["neutral"].any()  # no modifica la entrada


@pytest.fixture(scope="module")
def cup_model() -> mm.LeagueModel:
    """La liga de prueba más una copa en la que entra un amateur que solo jugó un partido."""
    league = _fake_league()
    m = league.matches
    last = m["datetime"].max()
    base = {"competition": "copa", "season": 2025, "extra_time": False, "neutral": False, "h_sig": None,
            "a_sig": None, "odds_h": None, "odds_d": None, "odds_a": None, "odds_over25": None, "odds_under25": None}
    cup = pd.DataFrame([
        {**base, "id": "c1", "datetime": last - pd.Timedelta(days=20), "home": "Amateur", "away": "Team5",
         "played": True, "hg": 0.0, "ag": 3.0, "h_sig": 0.4, "a_sig": 2.5},
        {**base, "id": "c2", "datetime": last - pd.Timedelta(days=19), "home": "Team1", "away": "Team2",
         "played": True, "hg": 1.0, "ag": 1.0, "h_sig": 1.2, "a_sig": 1.0},
        {**base, "id": "c3", "datetime": last + pd.Timedelta(days=5), "home": "Amateur", "away": "Team0",
         "played": False, "hg": None, "ag": None},
        {**base, "id": "c4", "datetime": last + pd.Timedelta(days=5), "home": "Team3", "away": "Team0",
         "played": False, "hg": None, "ag": None},
    ])
    matches = pd.concat([m, cup], ignore_index=True)
    data = mm.build_league_data("copa", "Copa de prueba", matches, league.sources, "xG", league.training_since,
                                train_on_focus=False, kappa_exclude=frozenset({"copa"}))
    return mm.train_league(data)


def test_cup_skips_teams_without_data(cup_model):
    fixtures = cup_model.data.focus_matches
    c3, c4 = (fixtures[fixtures["id"] == i].iloc[0] for i in ("c3", "c4"))
    with pytest.raises(mm.PredictionError, match="Sin pronóstico: Amateur casi no tiene"):
        mm.predict_match(cup_model, "Amateur", "Team0", fixture=c3)
    pred = mm.predict_match(cup_model, "Team3", "Team0", fixture=c4)
    assert pred.p_final.sum() == pytest.approx(1.0)
    assert not cup_model.trained.val_predictions["low_data"].any()  # la validación mide lo que se pronostica


def test_window_matches_counts_only_the_year_before(cup_model):
    index = cup_model.data.index
    last = cup_model.data.matches["datetime"].max()
    assert mm.window_matches(index, "Amateur", last) == 1
    assert mm.window_matches(index, "Amateur", last + pd.Timedelta(days=400)) == 0
    assert mm.window_matches(index, "Nadie", last) == 0
