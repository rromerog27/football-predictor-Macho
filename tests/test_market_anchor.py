"""Pruebas sin red del pronóstico anclado a la casa (match_model.MARKET_ANCHOR) y de los picks de consenso."""

import numpy as np
import pandas as pd
import pytest

from src import market_picks as mp
from src import markets
from src import match_model as mm
from tests.test_modelo_prediccion import _fake_league


@pytest.fixture(scope="module")
def trained():
    return mm.train_league(_fake_league())


def _pending_fixture(model, odds=None, ou=None):
    data = model.data
    pending = data.matches[~data.matches["played"]].iloc[0].copy()
    fixture = mm.find_fixture(data, pending["home"], pending["away"], now=pending["datetime"] - pd.Timedelta(days=1))
    fixture = fixture.copy()
    if odds is not None:
        fixture[["odds_h", "odds_d", "odds_a"]] = odds
    if ou is not None:
        fixture[["odds_over25", "odds_under25"]] = ou
    return fixture


def test_market_lambdas_recover_the_goals_behind_the_odds():
    m = mm.score_matrix(1.6, 1.1, -0.05)
    i, j = np.indices(m.shape)
    lam_h, lam_a = mm.market_lambdas(mm.outcome_probs(m), float(m[i + j > 2.5].sum()), -0.05)
    assert lam_h == pytest.approx(1.6, abs=0.02) and lam_a == pytest.approx(1.1, abs=0.02)


def test_forecast_uses_the_odds_and_keeps_the_own_model(trained):
    fixture = _pending_fixture(trained, odds=[1.8, 3.6, 4.5], ou=[1.9, 1.9])
    pred = mm.predict_match(trained, fixture["home"], fixture["away"], fixture=fixture)
    assert pred.anchored
    np.testing.assert_allclose(pred.p_final, pred.market["p_1x2"])
    np.testing.assert_allclose(mm.outcome_probs(pred.matrix), pred.market["p_1x2"], atol=1e-6)
    assert pred.over25 == pytest.approx(pred.market["over25"], abs=0.02)  # los goles salen de la línea de la casa
    np.testing.assert_allclose(pred.own, pred.p_model)  # el modelo propio sigue ahí para buscar valor
    assert pred.model_lambdas is not None


def test_without_odds_the_forecast_is_the_model(trained):
    fixture = _pending_fixture(trained)
    pred = mm.predict_match(trained, fixture["home"], fixture["away"], fixture=fixture)
    assert not pred.anchored and pred.market is None
    np.testing.assert_allclose(pred.p_final, pred.own)


def test_consensus_takes_the_lowest_probability():
    a = [markets.Selection("Goles", "Total de goles", "Más de 1.5", 0.92, 0.75),
         markets.Selection("Goles", "Ambos anotan", "Sí", 0.60, 0.52)]
    b = [markets.Selection("Goles", "Total de goles", "Más de 1.5", 0.85, 0.75)]
    out = markets.consensus(a, b)
    assert [s.p for s in out] == [0.85, 0.60] and out[0].base == 0.75


def test_own_goal_selections_follow_the_own_model(trained):
    fixture = _pending_fixture(trained, odds=[1.8, 3.6, 4.5])
    pred = mm.predict_match(trained, fixture["home"], fixture["away"], fixture=fixture)
    own = {(s.market, s.label): s.p for s in markets.own_goal_selections(pred)}
    assert own[("Doble oportunidad", "1X")] == pytest.approx(pred.own[0] + pred.own[1])


def test_goal_history_uses_consensus_where_there_were_odds():
    rho = -0.05

    def row(i: str, home: str, lam_h: float, lam_a: float) -> dict:
        p = mm.outcome_probs(mm.score_matrix(lam_h, lam_a, rho))
        return {"id": i, "datetime": pd.Timestamp("2026-09-01"), "competition": "eng.1", "home": home,
                "away": f"Rival {i}", "hg": 2, "ag": 0, "lam_h": lam_h, "lam_a": lam_a,
                **dict(zip(["p_model_H", "p_model_D", "p_model_A"], p))}

    # dos partidos del "Fuerte" (3 goles esperados contra 0.5) entre varios parejos que fijan el partido medio
    val = pd.DataFrame([row("x1", "Fuerte", 3.0, 0.5), row("x2", "Fuerte", 3.0, 0.5),
                        *(row(f"m{k}", f"Medio{k}", 1.2, 1.0) for k in range(8))])
    # la casa ve el partido parejo: con consenso, los picks fijos del modelo en ese partido desaparecen
    odds = pd.DataFrame([{"id": "x1", "odds_h": 2.6, "odds_d": 3.2, "odds_a": 2.8}])

    def fixed(picks):
        return {p.key for p in picks if p.tier == "Fijo"}

    assert fixed(mp.goal_history(val, rho)) == {"x1", "x2"}
    assert fixed(mp.goal_history(val, rho, odds)) == {"x2"}  # el x2 no tenía cuotas: el modelo solo
