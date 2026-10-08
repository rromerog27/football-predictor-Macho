"""Pruebas sin red del modelo con datos de Understat (src/understat_model.py)
y del script de terminal modelo_prediccion.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import modelo_prediccion
from src import understat_model as um


def _fake_season(season: int, n_teams: int = 6, rounds: int = 2, seed: int = 0,
                 leave_last_unplayed: bool = True, spacing_days: float = 3.0) -> list[dict]:
    """Calendario a vuelta completa con el formato JSON `dates` de Understat.
    Los equipos más bajos en la lista son más fuertes, para que haya señal."""
    rng = np.random.default_rng(seed + season)
    teams = [f"Team{i}" for i in range(n_teams)]
    strength = {t: 1.6 - 0.1 * i for i, t in enumerate(teams)}
    start = pd.Timestamp(f"{season}-08-10 14:00:00")
    dates, day = [], 0
    for _ in range(rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hxg = max(0.2, rng.normal(1.5 * strength[h] / strength[a], 0.3))
                axg = max(0.2, rng.normal(1.2 * strength[a] / strength[h], 0.3))
                dates.append(
                    {
                        "id": str(season * 10_000 + day),
                        "isResult": True,
                        "h": {"id": h, "title": h, "short_title": h[:3]},
                        "a": {"id": a, "title": a, "short_title": a[:3]},
                        "goals": {"h": str(rng.poisson(hxg)), "a": str(rng.poisson(axg))},
                        "xG": {"h": f"{hxg:.4f}", "a": f"{axg:.4f}"},
                        "datetime": str(start + pd.Timedelta(days=spacing_days * day)),
                    }
                )
                day += 1
    if leave_last_unplayed:
        dates[-1].update(isResult=False, goals={"h": None, "a": None}, xG={"h": None, "a": None})
    return dates


def _fake_league(current: int = 2025, n_seasons: int = 6, n_teams: int = 12) -> um.LeagueData:
    dates = {
        s: _fake_season(s, n_teams=n_teams, leave_last_unplayed=(s == current), spacing_days=2.0)
        for s in range(current - n_seasons + 1, current + 1)
    }
    matches = um.matches_frame(dates)
    played = matches[matches["played"]]
    sources = [um.SourceInfo(f"https://understat.com/getLeagueData/TEST/{s}", s, 0, True) for s in dates]
    return um.LeagueData("EPL", current, matches, um.MatchIndex(played), sources)


def test_matches_frame_parses_played_and_pending():
    df = um.matches_frame({2025: _fake_season(2025)})
    assert df["played"].sum() == len(df) - 1
    pending = df[~df["played"]].iloc[0]
    assert np.isnan(pending["hg"]) and np.isnan(pending["hxg"])
    assert df["datetime"].is_monotonic_increasing


def test_score_matrix_and_markets_match_closed_form():
    lam_h, lam_a = 1.6, 1.1
    matrix = um.score_matrix(lam_h, lam_a)
    assert matrix.sum() == pytest.approx(1.0)
    assert um.outcome_probs(matrix).sum() == pytest.approx(1.0)

    over25, btts = um.goal_markets(matrix)
    total = lam_h + lam_a
    under25 = np.exp(-total) * (1 + total + total**2 / 2)  # suma de Poisson independientes
    assert over25 == pytest.approx(1 - under25, abs=1e-6)
    assert btts == pytest.approx((1 - np.exp(-lam_h)) * (1 - np.exp(-lam_a)), abs=1e-6)


def test_reweight_matrix_reproduces_target_1x2():
    matrix = um.score_matrix(1.4, 1.2)
    target = np.array([0.40, 0.30, 0.30])
    adjusted = um.reweight_matrix(matrix, target)
    assert adjusted.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(um.outcome_probs(adjusted), target, atol=1e-12)


def test_fit_ratings_recovers_multiplicative_strengths():
    teams = [f"T{i}" for i in range(6)]
    attack = dict(zip(teams, [1.5, 1.2, 1.0, 0.9, 0.8, 0.7]))
    defense = dict(zip(teams, [0.6, 0.8, 1.0, 1.1, 1.2, 1.4]))
    rows = [
        {"home": h, "away": a, "hxg": 1.5 * attack[h] * defense[a], "axg": 1.2 * attack[a] * defense[h],
         "hg": 1.0, "ag": 1.0}
        for h in teams for a in teams if h != a
    ]
    window = pd.DataFrame(rows)
    ratings = um.fit_ratings(window)
    i = [ratings.index[t] for t in teams]

    # El ajuste reproduce el xG observado y conserva el orden de las fuerzas reales.
    fitted_h = ratings.mu_home * ratings.attack[window["home"].map(ratings.index)] \
        * ratings.defense[window["away"].map(ratings.index)]
    np.testing.assert_allclose(fitted_h, window["hxg"], rtol=0.03)
    assert list(np.argsort(-ratings.attack[i])) == list(range(6))
    assert list(np.argsort(ratings.defense[i])) == list(range(6))


def test_snapshot_uses_only_matches_before_cutoff():
    matches = um.matches_frame({2025: _fake_season(2025, rounds=3)})  # >50 partidos previos
    played = matches[matches["played"]].reset_index(drop=True)
    index = um.MatchIndex(played)
    target = played.iloc[70]

    snap = um.match_snapshot(index, target["home"], target["away"], target["datetime"], {}, with_rows=True)
    assert snap is not None
    h, a, _ = snap
    assert (h["recent_rows"]["datetime"] < target["datetime"]).all()
    assert len(h["recent_rows"]) == h["n_recent"] == um.N_RECENT
    long = um.team_long_frame(played)
    assert h["n_window"] == ((long["team"] == target["home"]) & (long["datetime"] < target["datetime"])).sum()

    # Cambiar partidos posteriores no altera la instantánea (sin fuga de información).
    tampered = played.copy()
    tampered.loc[tampered["datetime"] >= target["datetime"], ["hxg", "axg", "hg", "ag"]] = 9.0
    snap2 = um.match_snapshot(um.MatchIndex(tampered), target["home"], target["away"], target["datetime"], {})
    for key in ("att_recent", "def_recent", "att_long", "def_long", "ppg5", "ppg5_venue", "rest_days"):
        assert snap2[0][key] == pytest.approx(h[key])
        assert snap2[1][key] == pytest.approx(a[key])


@pytest.fixture(scope="module")
def trained_league() -> um.LeagueModel:
    return um.train_league(_fake_league())


def test_predict_match_end_to_end(trained_league):
    pred = um.predict_match(trained_league, "team0", "Team11")
    for probs in (pred.p_poisson, pred.p_logistic, pred.p_final):
        assert probs.sum() == pytest.approx(1.0)
    # El ensemble es una mezcla convexa de ambos modelos.
    w = trained_league.trained.w_poisson
    np.testing.assert_allclose(pred.p_final, w * pred.p_poisson + (1 - w) * pred.p_logistic)
    assert pred.p_final[0] > pred.p_final[2]  # Team0 es el más fuerte de la liga sintética
    assert pred.lam_home > pred.lam_away
    assert 0 < pred.over25 < 1 and 0 < pred.btts < 1
    assert pred.final_score is None


def test_predict_match_uses_pending_fixture(trained_league):
    data = trained_league.data
    pending = data.matches[~data.matches["played"]].iloc[0]
    now = pending["datetime"] - pd.Timedelta(days=1)
    fixture = um.find_fixture(data, pending["home"], pending["away"], now=now)
    pred = um.predict_match(trained_league, pending["home"], pending["away"], fixture=fixture)
    assert pred.match_id == pending["id"]
    assert pred.cutoff == pending["datetime"]
    assert pred.understat_url.endswith(f"/match/{pending['id']}")


def test_predict_played_fixture_reports_result_and_ignores_it(trained_league):
    data = trained_league.data
    played = data.matches[data.matches["played"]].iloc[-1]
    pred = um.predict_match(trained_league, played["home"], played["away"], fixture=played)
    assert pred.final_score == (int(played["hg"]), int(played["ag"]))
    assert (pred.home_snap["recent_rows"]["datetime"] < played["datetime"]).all()


def test_fixtures_between(trained_league):
    data = trained_league.data
    first = data.matches["datetime"].iloc[0]
    day = um.fixtures_between(data, first.normalize(), first.normalize() + pd.Timedelta(days=1))
    assert len(day) >= 1 and (day["datetime"].dt.normalize() == first.normalize()).all()


def test_cli_report_has_required_sections(trained_league):
    pred = um.predict_match(trained_league, "Team0", "Team5")
    text = modelo_prediccion.report(pred, trained_league.data.sources, detail=True)
    for header in ("FUENTES DE DATOS", "RESULTADOS DE LOS MODELOS", "PREDICCIÓN FINAL COMBINADA",
                   "Poisson Puro -> 1:", "Regresión Logística -> 1:", "Over 2.5", "Ambos Anotan"):
        assert header in text


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Leeds", "Leeds"),
        ("leeds united", "Leeds"),
        ("Man Utd", "Manchester United"),
        ("spurs", "Tottenham"),
        ("newcastle", "Newcastle United"),
        ("Nott'm Forest", "Nottingham Forest"),
    ],
)
def test_resolve_team(query, expected):
    teams = ["Arsenal", "Leeds", "Manchester United", "Manchester City", "Tottenham",
             "Newcastle United", "Nottingham Forest"]
    assert um.resolve_team(query, teams) == expected


def test_resolve_team_unknown_raises():
    with pytest.raises(um.PredictionError):
        um.resolve_team("Real Madrid", ["Arsenal", "Leeds"])


@pytest.mark.parametrize(("name", "code"), [("Premier League", "EPL"), ("LaLiga", "La_liga"), ("serie a", "Serie_A")])
def test_resolve_league(name, code):
    assert um.resolve_league(name) == code
