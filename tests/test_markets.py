"""Pruebas sin red de córners y tarjetas (ESPN, modelo por equipo) y de los mercados de un partido."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import competitions as comps
from src import corners_cards as cc
from src import espn_source
from src import markets
from src import match_model as mm


def _event(details=None, corners=("7", "3")) -> dict:
    def competitor(side, team_id, name, score, won_corners):
        return {"homeAway": side, "score": score, "team": {"id": team_id, "displayName": name, "abbreviation": team_id},
                "statistics": [{"name": "totalShots", "displayValue": "10"},
                               {"name": "shotsOnTarget", "displayValue": "4"},
                               {"name": "wonCorners", "displayValue": won_corners}]}

    competition = {"neutralSite": False,
                   "competitors": [competitor("home", "1", "Local FC", "2", corners[0]),
                                   competitor("away", "2", "Visitante FC", "1", corners[1])]}
    if details is not None:
        competition["details"] = details
    return {"id": "9", "date": "2026-09-01T20:00Z", "season": {"year": 2026},
            "status": {"type": {"name": "STATUS_FULL_TIME", "completed": True}}, "competitions": [competition]}


def _card(team_id, red=False):
    return {"team": {"id": team_id}, "yellowCard": not red, "redCard": red}


def test_espn_corners_and_cards_per_team():
    details = [_card("1"), _card("2"), _card("2", red=True), {"team": {"id": "1"}, "scoringPlay": True}]
    row = espn_source.events_frame([espn_source.trim_event(_event(details))], "mex.1").iloc[0]
    assert (row["h_corners"], row["a_corners"]) == (7.0, 3.0)
    assert (row["h_cards"], row["a_cards"]) == (1.0, 2.0)  # amarillas y rojas cuentan 1


def test_espn_missing_details_and_zero_corners_are_missing():
    row = espn_source.events_frame([espn_source.trim_event(_event(None, corners=("0", "0")))], "x").iloc[0]
    assert np.isnan(row["h_cards"]) and np.isnan(row["a_cards"])  # sin incidencias: no se sabe
    assert np.isnan(row["h_corners"]) and np.isnan(row["a_corners"])  # 0 y 0: ESPN rellenó con ceros
    no_cards = espn_source.events_frame([espn_source.trim_event(_event([]))], "x").iloc[0]
    assert np.isnan(no_cards["h_cards"])  # lista vacía = sin incidencias publicadas
    goal_only = espn_source.events_frame([espn_source.trim_event(_event([{"team": {"id": "1"}}]))], "x").iloc[0]
    assert (goal_only["h_cards"], goal_only["a_cards"]) == (0.0, 0.0)  # hay incidencias y ninguna tarjeta


def test_attach_espn_stats_matches_by_date_and_names():
    base = {"competition": "eng.1", "played": True}
    target = pd.DataFrame([
        {**base, "id": "us:1", "datetime": pd.Timestamp("2026-08-15 14:00"), "home": "Leeds", "away": "Everton"},
        {**base, "id": "us:2", "datetime": pd.Timestamp("2026-08-15 14:00"), "home": "Arsenal", "away": "Chelsea"},
        {**base, "id": "us:3", "datetime": pd.Timestamp("2026-08-22 14:00"), "home": "Everton", "away": "Leeds"},
    ])
    espn = pd.DataFrame([
        {"played": True, "datetime": pd.Timestamp("2026-08-15 15:00"), "home": "Leeds United", "away": "Everton",
         "h_corners": 6.0, "a_corners": 4.0, "h_cards": 2.0, "a_cards": 1.0},
        {"played": True, "datetime": pd.Timestamp("2026-08-16 11:00"), "home": "Arsenal", "away": "Chelsea",
         "h_corners": 9.0, "a_corners": 2.0, "h_cards": 0.0, "a_cards": 3.0},
    ])
    out = comps.attach_espn_stats(target, espn, "eng.1").set_index("id")
    assert out.loc["us:1", ["h_corners", "a_corners", "h_cards", "a_cards"]].tolist() == [6.0, 4.0, 2.0, 1.0]
    assert out.loc["us:2", "h_corners"] == 9.0  # otro día cercano (husos distintos)
    assert np.isnan(out.loc["us:3", "h_corners"])  # la vuelta no está en ESPN


def _synthetic_data(n_rounds: int = 60, seed: int = 0, with_cards: bool = True) -> SimpleNamespace:
    """Liga de 10 equipos en la que el equipo i genera córners en proporción a (1 + i/5)."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    rows, t0 = [], pd.Timestamp("2024-01-01")
    for r in range(n_rounds):
        order = rng.permutation(10)
        for k in range(5):
            h, a = order[2 * k], order[2 * k + 1]
            rate_h, rate_a = 3.0 * (1 + h / 5) / (1 + a / 10), 2.5 * (1 + a / 5) / (1 + h / 10)
            rows.append({"id": f"m{r}-{k}", "competition": "liga", "datetime": t0 + pd.Timedelta(days=3 * r, hours=k),
                         "home": teams[h], "away": teams[a], "played": True, "extra_time": False, "neutral": False,
                         "h_corners": float(rng.poisson(rate_h)), "a_corners": float(rng.poisson(rate_a)),
                         "h_cards": float(rng.poisson(2.0)) if with_cards else 0.0,
                         "a_cards": float(rng.poisson(2.2)) if with_cards else 0.0})
    return SimpleNamespace(matches=pd.DataFrame(rows), competition="liga")


def test_corner_model_learns_team_rates_and_validates():
    data = _synthetic_data()
    model = cc.fit(data, "corners")
    assert model is not None and model.validation is not None
    v = model.validation
    assert v.more_hits / v.more_n > 0.6  # quién saca más: las tasas de los equipos se notan
    strong = cc.forecast(model, "T9", "T0")
    weak = cc.forecast(model, "T0", "T9")
    assert strong.mu_home > strong.mu_away and weak.mu_home < weak.mu_away
    more = strong.more()
    assert more.sum() == pytest.approx(1.0) and more[0] > 0.7
    overs = [strong.p_over(x) for x in (7.5, 9.5, 11.5)]
    assert overs == sorted(overs, reverse=True)


def test_played_match_uses_its_pre_match_prediction():
    data = _synthetic_data()
    model = cc.fit(data, "corners")
    last = data.matches.iloc[-1]
    pre = cc.forecast(model, last["home"], last["away"], match_id=last["id"])
    now = cc.forecast(model, last["home"], last["away"])
    mh, ma, _, _ = model.pre[last["id"]]
    assert pre.mu_home == pytest.approx(cc.shrink(mh, pre.league_home, model.beta_team))
    assert pre.mu_home != pytest.approx(now.mu_home)  # el estado final ya incluye ese partido


def test_competition_without_published_cards_is_skipped():
    assert cc.fit(_synthetic_data(with_cards=False), "cards") is None
    assert cc.fit(_synthetic_data(), "cards") is not None


def test_main_line_is_closest_half_goal():
    assert cc.main_line(9.74) == 9.5 and cc.main_line(10.28) == 10.5 and cc.main_line(4.34) == 4.5


def _fake_pred(lam_h=1.8, lam_a=0.9, base=(1.45, 1.15)) -> SimpleNamespace:
    matrix = mm.score_matrix(lam_h, lam_a, -0.05)
    p = mm.outcome_probs(matrix)
    return SimpleNamespace(home="Local", away="Visitante", lam_home=lam_h, lam_away=lam_a, matrix=matrix, p_final=p,
                           base_home=base[0], base_away=base[1], trained=SimpleNamespace(rho=-0.05))


def test_goal_selections_are_consistent_with_the_matrix():
    pred = _fake_pred()
    sels = markets.goal_selections(pred)
    by = {(s.market, s.label): s for s in sels}
    for line in markets.GOAL_LINES:
        over, under = by[("Total de goles", f"Más de {line:g}")], by[("Total de goles", f"Menos de {line:g}")]
        assert over.p + under.p == pytest.approx(1.0)
        assert over.base is not None and over.text == f"Más de {line:g} goles"
    assert by[("Ambos anotan", "Sí")].p == pytest.approx(pred.matrix[1:, 1:].sum())
    assert by[("Doble oportunidad", "1X")].p == pytest.approx(pred.p_final[0] + pred.p_final[1])
    assert by[("Goles de Local", "Más de 0.5")].p == pytest.approx(1 - pred.matrix[0, :].sum())
    assert all(0 <= s.p <= 1 for s in sels)


def test_highlight_skips_near_certain_and_double_chance():
    sels = [markets.Selection("Goles", "Total de goles", "Más de 0.5", 0.95, 0.93),
            markets.Selection("Goles", "Doble oportunidad", "X2", 0.9, 0.55),
            markets.Selection("Goles", "Total de goles", "Más de 2.5", 0.66, 0.52),
            markets.Selection("Goles", "Ambos anotan", "Sí", 0.58, 0.5)]
    best = markets.highlight(sels)
    assert best is not None and best.label == "Más de 2.5"
    assert markets.highlight(sels[:2] + sels[3:]) is None


def test_count_selections_and_checks():
    model = cc.fit(_synthetic_data(), "corners")
    fc = cc.forecast(model, "T9", "T0")
    sels = markets.count_selections(fc, "T9", "T0")
    more = [s for s in sels if s.market == "Quién saca más córners"]
    assert [s.label for s in more] == ["T9", "Iguales", "T0"] and sum(s.p for s in more) == pytest.approx(1.0)
    assert more[0].text == "T9 saca más córners"
    checks = markets.count_checks(model)
    assert [c.market for c in checks][1] == "Quién saca más córners"
    merged = markets.merge_checks([checks, checks])
    assert merged[0].n == 2 * checks[0].n


def test_merge_checks_with_different_lines():
    a = markets.Check("Más/menos de 9.5 córners", 100, 55, 10, 7, 0.7)
    b = markets.Check("Más/menos de 10.5 córners", 50, 30, 0, 0, float("nan"))
    (merged,) = markets.merge_checks([[a], [b]])
    assert merged.n == 150 and merged.hits == 85 and merged.confident_n == 10
    assert "línea de cada liga" in merged.market
