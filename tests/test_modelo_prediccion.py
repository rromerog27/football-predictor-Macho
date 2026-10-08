"""Pruebas sin red del modelo (src/match_model.py), sus fuentes (Understat, ESPN),
el registro de competiciones y el script de terminal modelo_prediccion.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import modelo_prediccion
from src import competitions as comps
from src import espn_source, understat_source
from src import match_model as mm


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


def _fake_league(current: int = 2025, n_seasons: int = 6, n_teams: int = 12) -> mm.LeagueData:
    dates = {
        s: _fake_season(s, n_teams=n_teams, leave_last_unplayed=(s == current), spacing_days=2.0)
        for s in range(current - n_seasons + 1, current + 1)
    }
    matches = understat_source.matches_frame(dates, "eng.1")
    sources = [mm.SourceInfo(f"https://understat.com/getLeagueData/TEST/{s}", str(s), 0, True) for s in dates]
    return mm.build_league_data("eng.1", "Liga de prueba", matches, sources, "xG",
                                pd.Timestamp(f"{current - n_seasons + 2}-07-01"))


def _espn_event(event_id: str = "1", status: str = "STATUS_FULL_TIME", completed: bool = True,
                home_score: str = "2", away_score: str = "1", with_odds: bool = False) -> dict:
    """Evento con la forma del marcador de ESPN (solo los campos que se leen)."""
    def competitor(side, name, abbr, score, shots, sot):
        return {"homeAway": side, "score": score,
                "team": {"id": abbr, "displayName": name, "abbreviation": abbr},
                "statistics": [{"name": "totalShots", "displayValue": shots},
                               {"name": "shotsOnTarget", "displayValue": sot}]}

    competition = {
        "neutralSite": False,
        "competitors": [competitor("home", "Club América", "AME", home_score, "14", "6"),
                        competitor("away", "Monterrey", "MTY", away_score, "9", "3")],
    }
    if with_odds:
        competition["odds"] = [{
            "moneyline": {"home": {"open": {"odds": "-110"}, "close": {"odds": "+100"}},
                          "away": {"close": {"odds": "+250"}}},
            "drawOdds": {"moneyLine": 240},
            "overUnder": 2.5,
            "total": {"over": {"close": {"odds": "-120"}}, "under": {"close": {"odds": "EVEN"}}},
        }]
    return {"id": event_id, "date": "2026-10-11T03:10Z", "season": {"year": 2026},
            "status": {"type": {"name": status, "completed": completed}}, "competitions": [competition]}


# --------------------------------------------------------------------------
# Fuentes
# --------------------------------------------------------------------------


def test_understat_matches_frame_parses_played_and_pending():
    df = understat_source.matches_frame({2025: _fake_season(2025)}, "eng.1")
    assert set(mm.STANDARD_COLUMNS) <= set(df.columns)
    assert df["played"].sum() == len(df) - 1
    pending = df[~df["played"]].iloc[0]
    assert np.isnan(pending["hg"]) and np.isnan(pending["h_sig"])
    assert df["datetime"].is_monotonic_increasing


@pytest.mark.parametrize(("american", "decimal"), [("+150", 2.5), ("-275", 1 + 100 / 275), ("EVEN", 2.0), (225, 3.25)])
def test_american_to_decimal(american, decimal):
    assert espn_source.american_to_decimal(american) == pytest.approx(decimal)


def test_espn_event_parsing_with_shots_and_odds():
    events = [espn_source.trim_event(_espn_event(with_odds=True))]
    df = espn_source.events_frame(events, "mex.1")
    row = df.iloc[0]
    assert row["home"] == "Club América"
    assert row["played"] and (row["hg"], row["ag"]) == (2.0, 1.0)
    assert row["h_sig"] == pytest.approx(espn_source.pseudo_xg(14, 6))
    assert row["datetime"] == pd.Timestamp("2026-10-11 03:10")
    assert (row["odds_h"], row["odds_d"], row["odds_a"]) == pytest.approx((2.0, 3.4, 3.5))  # cierre, no apertura
    assert (row["odds_over25"], row["odds_under25"]) == pytest.approx((1 + 100 / 120, 2.0))


def test_espn_drops_postponed_and_marks_extra_time():
    assert espn_source.trim_event(_espn_event(status="STATUS_POSTPONED", completed=False)) is None
    aet = espn_source.events_frame([espn_source.trim_event(_espn_event(status="STATUS_FINAL_AET"))], "x")
    assert bool(aet["extra_time"].iloc[0])


def test_disambiguate_names_only_for_real_collisions():
    df = pd.DataFrame({"home": ["Nacional", "Nacional"], "home_id": ["1", "2"], "home_abbr": ["NAC", "CNA"],
                       "away": ["Peñarol", "Millonarios"], "away_id": ["3", "4"], "away_abbr": ["PEN", "MIL"]})
    out = espn_source.disambiguate_names(df)
    assert set(out["home"]) == {"Nacional (NAC)", "Nacional (CNA)"}
    assert list(out["away"]) == ["Peñarol", "Millonarios"]


# --------------------------------------------------------------------------
# Poisson, Dixon-Coles y mercados
# --------------------------------------------------------------------------


def test_score_matrix_and_markets_match_closed_form():
    lam_h, lam_a = 1.6, 1.1
    matrix = mm.score_matrix(lam_h, lam_a)
    assert matrix.sum() == pytest.approx(1.0)
    over25, btts = mm.goal_markets(matrix)
    total = lam_h + lam_a
    under25 = np.exp(-total) * (1 + total + total**2 / 2)  # suma de Poisson independientes
    assert over25 == pytest.approx(1 - under25, abs=1e-6)
    assert btts == pytest.approx((1 - np.exp(-lam_h)) * (1 - np.exp(-lam_a)), abs=1e-6)


def test_dixon_coles_moves_mass_to_draws_when_rho_negative():
    base = mm.score_matrix(1.3, 1.1)
    dc = mm.score_matrix(1.3, 1.1, rho=-0.1)
    assert dc.sum() == pytest.approx(1.0)
    assert mm.outcome_probs(dc)[1] > mm.outcome_probs(base)[1]
    np.testing.assert_allclose(mm.dixon_coles_tau(2, 3, 1.3, 1.1, -0.1), 1.0)  # solo afecta a 0-0/1-0/0-1/1-1


def test_fit_rho_is_near_zero_for_independent_poisson():
    rng = np.random.default_rng(1)
    lam_h, lam_a = rng.uniform(0.8, 2.0, 4000), rng.uniform(0.6, 1.6, 4000)
    rho = mm.fit_rho(rng.poisson(lam_h), rng.poisson(lam_a), lam_h, lam_a)
    assert abs(rho) < 0.05


def test_reweight_matrix_reproduces_target_1x2():
    matrix = mm.score_matrix(1.4, 1.2, rho=-0.05)
    target = np.array([0.40, 0.30, 0.30])
    np.testing.assert_allclose(mm.outcome_probs(mm.reweight_matrix(matrix, target)), target, atol=1e-12)


def test_market_probs_removes_margin():
    fixture = pd.Series({"odds_h": 2.0, "odds_d": 3.4, "odds_a": 3.5, "odds_over25": 1.8, "odds_under25": 2.0})
    market = mm.market_probs(fixture)
    assert market["p_1x2"].sum() == pytest.approx(1.0)
    assert market["margin"] == pytest.approx(1 / 2 + 1 / 3.4 + 1 / 3.5 - 1)
    assert market["over25"] == pytest.approx((1 / 1.8) / (1 / 1.8 + 1 / 2.0))
    assert mm.market_probs(pd.Series({"odds_h": np.nan, "odds_d": 3.0, "odds_a": 3.0})) is None


# --------------------------------------------------------------------------
# Fuerzas y no fuga de información
# --------------------------------------------------------------------------


def test_fit_ratings_recovers_multiplicative_strengths():
    teams = [f"T{i}" for i in range(6)]
    attack = dict(zip(teams, [1.5, 1.2, 1.0, 0.9, 0.8, 0.7]))
    defense = dict(zip(teams, [0.6, 0.8, 1.0, 1.1, 1.2, 1.4]))
    rows = [
        {"home": h, "away": a, "hxg": 1.5 * attack[h] * defense[a], "axg": 1.2 * attack[a] * defense[h],
         "hg": 1.0, "ag": 1.0}
        for h in teams for a in teams if h != a
    ]
    ratings = mm.fit_ratings(pd.DataFrame(rows), "hxg", "axg")
    i = [ratings.index[t] for t in teams]
    # El suavizado acerca las fuerzas a 1, pero conserva el orden real.
    assert list(np.argsort(-ratings.attack[i])) == list(range(6))
    assert list(np.argsort(ratings.defense[i])) == list(range(6))


def test_fit_ratings_survives_team_without_goals():
    rows = [{"home": "A", "away": "B", "hg": 0.0, "ag": 2.0}, {"home": "B", "away": "C", "hg": 1.0, "ag": 0.0},
            {"home": "C", "away": "A", "hg": 3.0, "ag": 0.0}]
    ratings = mm.fit_ratings(pd.DataFrame(rows), "hg", "ag")
    assert np.isfinite(ratings.attack).all() and (ratings.attack > 0).all()


def test_snapshot_uses_only_matches_before_cutoff():
    matches = understat_source.matches_frame({2025: _fake_season(2025, rounds=3)}, "eng.1")  # >50 partidos previos
    played = matches[matches["played"]].reset_index(drop=True)
    index = mm.MatchIndex(played)
    target = played.iloc[70]

    snap = mm.match_snapshot(index, target["home"], target["away"], target["datetime"], {}, with_rows=True)
    assert snap is not None
    h, a, _ = snap
    assert (h["recent_rows"]["datetime"] < target["datetime"]).all()
    assert len(h["recent_rows"]) == h["n_recent"] == mm.N_RECENT

    # Cambiar partidos posteriores no altera la instantánea (sin fuga de información).
    tampered = played.copy()
    tampered.loc[tampered["datetime"] >= target["datetime"], ["h_sig", "a_sig", "hg", "ag"]] = 9.0
    snap2 = mm.match_snapshot(mm.MatchIndex(tampered), target["home"], target["away"], target["datetime"], {})
    for key in ("att_recent_sig", "def_recent_gls", "att_long_sig", "def_long_gls", "ppg5", "ppg5_venue",
                "rest_days"):
        assert snap2[0][key] == pytest.approx(h[key])
        assert snap2[1][key] == pytest.approx(a[key])


def test_neutral_venue_uses_average_base():
    gh, ga = mm.base_goals(1.6, 1.2, np.array([False, True]))
    assert (gh[0], ga[0]) == (1.6, 1.2)
    assert gh[1] == ga[1] == pytest.approx(1.4)


# --------------------------------------------------------------------------
# Entrenamiento y predicción de punta a punta
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_league() -> mm.LeagueModel:
    return mm.train_league(_fake_league())


def test_predict_match_end_to_end(trained_league):
    t = trained_league.trained
    assert t.shrink in mm.SHRINK_GRID and t.signal_weight in mm.SIGNAL_WEIGHT_GRID
    assert mm.RHO_BOUNDS[0] <= t.rho <= mm.RHO_BOUNDS[1]
    assert t.ll_ensemble < t.ll_baseline  # la liga sintética tiene señal real

    pred = mm.predict_match(trained_league, "team0", "Team11")
    for probs in (pred.p_poisson, pred.p_logistic, pred.p_final):
        assert probs.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pred.p_final, t.w_poisson * pred.p_poisson + (1 - t.w_poisson) * pred.p_logistic)
    assert pred.p_final[0] > pred.p_final[2]  # Team0 es el más fuerte de la liga sintética
    assert pred.lam_home > pred.lam_away
    assert pred.final_score is None and pred.market is None and not pred.low_data


def test_goals_only_competition_trains_with_fixed_weight():
    data = _fake_league()
    goals_only = mm.build_league_data(data.competition, data.name, data.matches, data.sources, "goles",
                                      data.training_since)
    assert not goals_only.index.has_signal
    assert (goals_only.index.played["h_sig"] == goals_only.index.played["hg"]).all()
    assert mm.train_league(goals_only).trained.signal_weight == 1.0


def test_predict_match_uses_pending_fixture_and_its_odds(trained_league):
    data = trained_league.data
    pending = data.matches[~data.matches["played"]].iloc[0].copy()
    now = pending["datetime"] - pd.Timedelta(days=1)
    fixture = mm.find_fixture(data, pending["home"], pending["away"], now=now)
    assert fixture is not None and fixture["id"] == pending["id"]
    fixture = fixture.copy()
    fixture[["odds_h", "odds_d", "odds_a"]] = [1.8, 3.6, 4.5]
    pred = mm.predict_match(trained_league, pending["home"], pending["away"], fixture=fixture)
    assert pred.match_id == pending["id"] and pred.cutoff == pending["datetime"]
    assert pred.market is not None and pred.market["p_1x2"].sum() == pytest.approx(1.0)


def test_predict_played_fixture_reports_result_and_ignores_it(trained_league):
    played = trained_league.data.matches[trained_league.data.matches["played"]].iloc[-1]
    pred = mm.predict_match(trained_league, played["home"], played["away"], fixture=played)
    assert pred.final_score == (int(played["hg"]), int(played["ag"]))
    assert (pred.home_snap["recent_rows"]["datetime"] < played["datetime"]).all()


def test_cli_report_has_required_sections(trained_league):
    pending = trained_league.data.matches[~trained_league.data.matches["played"]].iloc[0].copy()
    pending[["odds_h", "odds_d", "odds_a", "odds_over25", "odds_under25"]] = [2.1, 3.3, 3.6, 1.9, 1.9]
    pred = mm.predict_match(trained_league, pending["home"], pending["away"], fixture=pending)
    text = modelo_prediccion.report(pred, trained_league.data, detail=True)
    for header in ("FUENTES DE DATOS", "RESULTADOS DE LOS MODELOS", "PREDICCIÓN FINAL COMBINADA",
                   "Poisson (Dixon-Coles", "Regresión Logística -> 1:", "Over 2.5", "Ambos Anotan", "MERCADO"):
        assert header in text


# --------------------------------------------------------------------------
# Competiciones y nombres
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "code"), [("Premier League", "eng.1"), ("EPL", "eng.1"), ("LaLiga", "esp.1"),
                                            ("Liga MX", "mex.1"), ("champions", "uefa.champions"),
                                            ("libertadores", "conmebol.libertadores"), ("argentina", "arg.1")])
def test_resolve_competition(name, code):
    assert comps.resolve_competition(name).code == code


def test_competition_registry_is_consistent():
    assert len(comps.BY_CODE) == len(comps.COMPETITIONS)
    assert set(comps.DEFAULT_CODES) <= set(comps.BY_CODE)
    assert all(c.region in comps.REGIONS for c in comps.COMPETITIONS)
    assert all(c.is_cup == bool(c.pool) for c in comps.COMPETITIONS)


def test_espn_days_cover_a_european_local_day():
    # Un día en Madrid (UTC+2) empieza la tarde anterior en la costa este de EE. UU.
    start = pd.Timestamp("2026-10-09 22:00")
    assert comps.espn_days(start, start + pd.Timedelta(days=1)) == ["20261009", "20261010"]


def test_match_to_source_pairs_by_time_and_names():
    espn = pd.DataFrame({"id": ["espn:1", "espn:2"], "datetime": pd.to_datetime(["2026-10-10 11:30", "2026-10-10 14:00"]),
                         "home": ["Arsenal", "Manchester United"], "away": ["Leeds United", "Tottenham Hotspur"]})
    source = pd.DataFrame({"id": ["us:9", "us:8"], "datetime": pd.to_datetime(["2026-10-10 11:30", "2026-10-10 17:30"]),
                           "home": ["Arsenal", "Manchester United"], "away": ["Leeds", "Tottenham"]})
    assert comps.match_to_source(espn, source) == {"espn:1": "us:9"}  # el segundo está a 3.5 h: no se empareja
    assert comps.match_to_source(espn, source, max_gap=pd.Timedelta(hours=4)) == {"espn:1": "us:9", "espn:2": "us:8"}
    rival = source.assign(away=["Chelsea", "Tottenham"])
    assert comps.match_to_source(espn, rival) == {}  # misma hora y local, pero otro visitante


@pytest.mark.parametrize(
    ("query", "expected"),
    [("Leeds", "Leeds"), ("leeds united", "Leeds"), ("Man Utd", "Manchester United"), ("spurs", "Tottenham"),
     ("newcastle", "Newcastle United"), ("Nott'm Forest", "Nottingham Forest")],
)
def test_resolve_team(query, expected):
    teams = ["Arsenal", "Leeds", "Manchester United", "Manchester City", "Tottenham",
             "Newcastle United", "Nottingham Forest"]
    assert mm.resolve_team(query, teams) == expected


def test_resolve_team_unknown_raises():
    with pytest.raises(mm.PredictionError):
        mm.resolve_team("Real Madrid", ["Arsenal", "Leeds"])
