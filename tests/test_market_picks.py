"""Pruebas sin red de la regla de picks de goles, córners y tarjetas, su historial y los parlays."""

import numpy as np
import pandas as pd
import pytest

from src import corners_cards as cc
from src import market_picks as mp
from src import markets
from src import match_model as mm
from tests.test_markets import _synthetic_data


def _sel(p, base, family="Goles", market="Total de goles", label="Más de 2.5"):
    return markets.Selection(family, market, label, p, base, f"{label} ({market})")


def _pick(key, tier, p, day="2026-10-01 18:00", hit=None, family="Goles"):
    return mp.MarketPick(key, pd.Timestamp(day), "eng.1", f"L{key}", f"V{key}", tier, _sel(p, p - 0.2, family), hit)


@pytest.mark.parametrize(("p", "base", "tier"), [
    (0.95, 0.80, "Fijo"), (0.95, 0.90, None),  # casi siempre pasa: sin pick
    (0.85, 0.70, "Intermedio"), (0.85, 0.80, None),
    (0.70, 0.50, "Arriesgado"), (0.70, 0.60, None),  # al arriesgado le piden 15 puntos
    (0.60, 0.20, None),
])
def test_tier_needs_probability_and_edge_over_normal(p, base, tier):
    t = mp.tier_of(_sel(p, base))
    assert (t.name if t else None) == tier


def test_match_picks_keeps_the_biggest_edge_per_family_and_tier():
    sels = [_sel(0.92, 0.80, label="Más de 0.5"), _sel(0.91, 0.70, label="Más de 1.5"),
            _sel(0.93, 0.75, family="Córners", market="Total de córners", label="Menos de 12.5"),
            _sel(0.70, 0.50, label="Más de 2.5")]
    picks = {(tier, s.family): s.label for tier, s in mp.match_picks(sels)}
    assert picks == {("Fijo", "Goles"): "Más de 1.5", ("Fijo", "Córners"): "Menos de 12.5",
                     ("Arriesgado", "Goles"): "Más de 2.5"}


@pytest.mark.parametrize(("market", "label", "hv", "av", "expected"), [
    ("Total de goles", "Más de 2.5", 2, 1, True), ("Total de goles", "Menos de 2.5", 2, 1, False),
    ("Goles de L", "Más de 1.5", 2, 0, True), ("Goles de V", "Más de 0.5", 2, 0, False),
    ("Ambos anotan", "Sí", 1, 1, True), ("Ambos anotan", "No", 1, 1, False),
    ("Doble oportunidad", "1X", 0, 0, True), ("Doble oportunidad", "12", 0, 0, False),
    ("Ganar por 2 o más", "L", 3, 1, True), ("Ganar por 2 o más", "V", 1, 2, False),
])
def test_selection_hit_goals(market, label, hv, av, expected):
    assert mp.selection_hit(markets.Selection("Goles", market, label, 0.5), hv, av, "L", "V") is expected


def test_selection_hit_counts():
    more = markets.Selection("Córners", "Quién saca más córners", "Iguales", 0.1)
    team = markets.Selection("Córners", "Córners de V", "Más de 3.5", 0.6)
    assert mp.selection_hit(more, 5, 5, "L", "V") and mp.selection_hit(team, 2, 4, "L", "V")


def test_market_disagrees():
    assert not mp.market_disagrees(np.array([0.6, 0.25, 0.15]), None)
    assert not mp.market_disagrees(np.array([0.6, 0.25, 0.15]), np.array([0.7, 0.2, 0.1]))
    assert mp.market_disagrees(np.array([0.3, 0.2, 0.5]), np.array([0.12, 0.14, 0.74]))  # 24 puntos
    assert mp.market_disagrees(np.array([0.45, 0.3, 0.25]), np.array([0.3, 0.3, 0.4]))  # otro favorito


def test_build_parlays_one_leg_per_match_and_minimum_legs():
    picks = [_pick("a", "Fijo", 0.95), _pick("a", "Fijo", 0.97, family="Córners"), _pick("b", "Fijo", 0.92),
             _pick("c", "Intermedio", 0.85)]
    parlays = mp.build_parlays(picks)
    assert set(parlays) == {"Fija"}  # la intermedia tiene un solo partido
    fija = parlays["Fija"]
    assert [leg.key for leg in fija.legs] == ["a", "b"] and fija.legs[0].sel.p == 0.97  # la más probable de "a"
    assert fija.p == pytest.approx(0.97 * 0.92) and fija.fair_odds == pytest.approx(1 / fija.p)
    assert fija.hit is None


def test_simulate_parlays_by_day():
    picks = [_pick("a", "Fijo", 0.95, "2026-10-01 12:00", True), _pick("b", "Fijo", 0.93, "2026-10-01 20:00", True),
             _pick("c", "Fijo", 0.95, "2026-10-02 12:00", True), _pick("d", "Fijo", 0.94, "2026-10-02 15:00", False)]
    sim = mp.simulate_parlays(picks)
    assert sim["hit"].tolist() == [True, False] and (sim["legs"] == 2).all()
    summary = mp.summary(picks)
    assert summary.iloc[0][["n", "hits"]].tolist() == [4, 3]


def test_goal_history_uses_validation_rows():
    rho = -0.05
    val = pd.DataFrame([
        {"id": f"m{i}", "datetime": pd.Timestamp("2026-09-01") + pd.Timedelta(days=i), "competition": "eng.1",
         "home": "Fuerte", "away": f"Débil{i}", "hg": hg, "ag": 0, "lam_h": lh, "lam_a": 0.5,
         **dict(zip(["p_model_H", "p_model_D", "p_model_A"], mm.outcome_probs(mm.score_matrix(lh, 0.5, rho))))}
        for i, (lh, hg) in enumerate([(3.2, 3), (3.0, 2), (0.9, 0), (1.0, 1), (1.1, 0)])])
    picks = mp.goal_history(val, rho)
    assert picks and all(p.hit is not None for p in picks)
    strong = [p for p in picks if p.away in ("Débil0", "Débil1")]
    assert any(p.tier == "Fijo" for p in strong)  # 3 goles esperados contra 0.5: el fuerte marca


def test_count_history_from_validation_forecasts():
    model = cc.fit(_synthetic_data(), "corners")
    forecasts = cc.validation_forecasts(model)
    assert forecasts and all(fc.mu_home > 0 for _, fc in forecasts)
    picks = mp.count_history(model)
    assert picks and {p.sel.family for p in picks} == {"Córners"}
    assert all(isinstance(p.hit, bool) for p in picks)


def test_pick_edge_property():
    assert _pick("a", "Fijo", 0.95).edge == pytest.approx(0.2)
