"""Pruebas sin red de modelo_prediccion.py (scraping de Understat + Poisson/logística)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import modelo_prediccion as mp


def _fake_understat_dates(n_teams: int = 6, rounds: int = 2, seed: int = 0) -> list[dict]:
    """Calendario a doble vuelta con el formato JSON `dates` de Understat.
    El último partido queda sin jugar (como un partido futuro)."""
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(n_teams)]
    start = pd.Timestamp("2025-08-16 14:00:00")
    dates, match_id, day = [], 1, 0
    for _ in range(rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hxg, axg = rng.uniform(0.3, 2.5), rng.uniform(0.2, 2.0)
                dates.append(
                    {
                        "id": str(match_id),
                        "isResult": True,
                        "h": {"id": h, "title": h, "short_title": h[:3]},
                        "a": {"id": a, "title": a, "short_title": a[:3]},
                        "goals": {"h": str(rng.poisson(hxg)), "a": str(rng.poisson(axg))},
                        "xG": {"h": f"{hxg:.4f}", "a": f"{axg:.4f}"},
                        "datetime": str(start + pd.Timedelta(days=3 * day)),
                    }
                )
                match_id += 1
                day += 1
    last = dates[-1]
    last.update(isResult=False, goals={"h": None, "a": None}, xG={"h": None, "a": None})
    return dates


def test_matches_frame_parses_played_and_pending():
    df = mp.matches_frame({2025: _fake_understat_dates()})
    assert df["played"].sum() == len(df) - 1
    pending = df[~df["played"]].iloc[0]
    assert np.isnan(pending["hg"]) and np.isnan(pending["hxg"])
    assert df["datetime"].is_monotonic_increasing


def test_score_matrix_and_markets_match_closed_form():
    lam_h, lam_a = 1.6, 1.1
    matrix = mp.score_matrix(lam_h, lam_a)
    assert matrix.sum() == pytest.approx(1.0)
    assert mp.outcome_probs(matrix).sum() == pytest.approx(1.0)

    over25, btts = mp.goal_markets(matrix)
    total = lam_h + lam_a
    under25 = np.exp(-total) * (1 + total + total**2 / 2)  # suma de Poisson independientes
    assert over25 == pytest.approx(1 - under25, abs=1e-6)
    assert btts == pytest.approx((1 - np.exp(-lam_h)) * (1 - np.exp(-lam_a)), abs=1e-6)


def test_reweight_matrix_reproduces_target_1x2():
    matrix = mp.score_matrix(1.4, 1.2)
    target = np.array([0.40, 0.30, 0.30])
    adjusted = mp.reweight_matrix(matrix, target)
    assert adjusted.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(mp.outcome_probs(adjusted), target, atol=1e-12)


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
    ratings = mp.fit_ratings(window)
    i = [ratings.index[t] for t in teams]

    # El ajuste reproduce el xG observado y conserva el orden de las fuerzas reales.
    fitted_h = ratings.mu_home * ratings.attack[window["home"].map(ratings.index)] \
        * ratings.defense[window["away"].map(ratings.index)]
    np.testing.assert_allclose(fitted_h, window["hxg"], rtol=0.03)
    assert list(np.argsort(-ratings.attack[i])) == list(range(6))
    assert list(np.argsort(ratings.defense[i])) == list(range(6))


def test_snapshot_uses_only_matches_before_cutoff():
    played_all = mp.matches_frame({2025: _fake_understat_dates(rounds=3)})  # >50 partidos previos
    played = played_all[played_all["played"]].reset_index(drop=True)
    long = mp.team_long_frame(played)
    target = played.iloc[70]

    snap = mp.match_snapshot(played, long, target["home"], target["away"], target["datetime"], {})
    assert snap is not None
    h, a, _ = snap
    assert (h["recent_rows"]["datetime"] < target["datetime"]).all()
    assert h["n_window"] == ((long["team"] == target["home"]) & (long["datetime"] < target["datetime"])).sum()

    # Cambiar partidos posteriores no altera la instantánea (sin fuga de información).
    tampered = played.copy()
    later = tampered["datetime"] >= target["datetime"]
    tampered.loc[later, ["hxg", "axg", "hg", "ag"]] = 9.0
    snap2 = mp.match_snapshot(tampered, mp.team_long_frame(tampered), target["home"], target["away"],
                              target["datetime"], {})
    for key in ("att_recent", "def_recent", "att_long", "def_long", "ppg5", "rest_days"):
        assert snap2[0][key] == pytest.approx(h[key])
        assert snap2[1][key] == pytest.approx(a[key])


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
    assert mp.resolve_team(query, teams) == expected


def test_resolve_team_unknown_raises():
    with pytest.raises(SystemExit):
        mp.resolve_team("Real Madrid", ["Arsenal", "Leeds"])


@pytest.mark.parametrize(("name", "code"), [("Premier League", "EPL"), ("LaLiga", "La_liga"), ("serie a", "Serie_A")])
def test_resolve_league(name, code):
    assert mp.resolve_league(name) == code
