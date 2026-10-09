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


def test_espn_zero_shots_for_both_teams_means_no_statistics():
    event = _espn_event()
    for c in event["competitions"][0]["competitors"]:
        c["statistics"] = [{"name": "totalShots", "displayValue": "0"}, {"name": "shotsOnTarget", "displayValue": "0"}]
    row = espn_source.events_frame([espn_source.trim_event(event)], "arg.2").iloc[0]
    assert row["played"] and np.isnan(row["h_sig"]) and np.isnan(row["a_sig"])  # el modelo usará los goles


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
    assert t.ll_model < t.ll_baseline  # la liga sintética tiene señal real

    pred = mm.predict_match(trained_league, "team0", "Team11")
    assert pred.p_final.sum() == pytest.approx(1.0)
    # sin alineaciones, el 1X2 final es el de Poisson con el favorito calibrado
    np.testing.assert_allclose(pred.p_final, mm.calibrate_favorite(pred.p_poisson))
    np.testing.assert_allclose(pred.p_model, pred.p_final)
    assert 0 < pred.over25 < 1 and pred.p_before_lineups is None
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


def test_calibrate_favorite_only_stretches_clear_favorites():
    even = np.array([0.45, 0.30, 0.25])
    np.testing.assert_allclose(mm.calibrate_favorite(even), even)  # nadie pasa del umbral: sin cambios
    fav = np.array([0.15, 0.25, 0.60])
    q = mm.calibrate_favorite(fav)
    assert q.sum() == pytest.approx(1.0) and q[2] > fav[2]
    assert q[0] / q[1] == pytest.approx(fav[0] / fav[1])  # el resto se reparte en proporción
    strong = mm.calibrate_favorite(np.array([0.80, 0.12, 0.08]))
    assert strong[0] > q[2] and strong[0] < 1  # más estirado cuanto más claro, sin llegar a 1
    rows = mm.calibrate_favorite(np.vstack([even, fav]))  # también por filas
    np.testing.assert_allclose(rows, np.vstack([even, q]))
    at_threshold = np.array([mm.FAVORITE_THRESHOLD, 0.25, 0.75 - mm.FAVORITE_THRESHOLD])
    np.testing.assert_allclose(mm.calibrate_favorite(at_threshold), at_threshold)  # continuo en el umbral


def test_cli_report_has_required_sections(trained_league):
    pending = trained_league.data.matches[~trained_league.data.matches["played"]].iloc[0].copy()
    pending[["odds_h", "odds_d", "odds_a", "odds_over25", "odds_under25"]] = [2.1, 3.3, 3.6, 1.9, 1.9]
    pred = mm.predict_match(trained_league, pending["home"], pending["away"], fixture=pending)
    text = modelo_prediccion.report(pred, trained_league.data, detail=True)
    for header in ("FUENTES DE DATOS", "MODELO (POISSON CON DIXON-COLES)", "PREDICCIÓN FINAL",
                   "Poisson (Dixon-Coles", "Over 2.5", "Ambos Anotan", "MERCADO"):
        assert header in text
    assert "Logística" not in text


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


# --------------------------------------------------------------------------
# Calibraciones (competición, recién llegados, total de goles)
# --------------------------------------------------------------------------


def test_competition_kappa_ratio_with_prior_and_exclusion():
    comp = np.array(["liga"] * 100 + ["copa"] * 100)
    hg, ag = np.full(200, 2.0), np.full(200, 1.0)
    lam_h, lam_a = np.full(200, 1.0), np.full(200, 1.0)
    kappa = mm.competition_kappa(comp, hg, ag, lam_h, lam_a, exclude=frozenset({"copa"}))
    assert ("copa", "h") not in kappa
    assert 1.0 < kappa[("liga", "h")] < 2.0  # goles reales el doble de lo esperado, suavizado hacia 1
    assert kappa[("liga", "a")] == pytest.approx(1.0)
    lh, la = mm.apply_kappa(np.array(["liga", "copa"]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), kappa)
    assert lh[0] == pytest.approx(kappa[("liga", "h")]) and lh[1] == 1.0


def test_newcomer_factors_detect_weaker_promoted_teams():
    rng = np.random.default_rng(3)
    n = 2000
    new_h = rng.random(n) < 0.2
    new_a = (rng.random(n) < 0.2) & ~new_h
    lam_h, lam_a = np.full(n, 1.5), np.full(n, 1.2)
    true_h = lam_h * np.where(new_h, 0.8, 1.0) * np.where(new_a, 1.3, 1.0)
    true_a = lam_a * np.where(new_a, 0.8, 1.0) * np.where(new_h, 1.3, 1.0)
    own, rival = mm.newcomer_factors(rng.poisson(true_h), rng.poisson(true_a), lam_h, lam_a, new_h, new_a)
    assert own == pytest.approx(0.8, abs=0.06) and rival == pytest.approx(1.3, abs=0.08)


def test_shrink_totals_keeps_ratio_and_moves_total_to_mean():
    comp = np.array(["liga", "liga"])
    lam_h, lam_a = np.array([2.4, 0.9]), np.array([1.2, 0.6])
    same = mm.shrink_totals(comp, lam_h, lam_a, 1.0, {"liga": 2.5})
    np.testing.assert_allclose(same[0], lam_h)
    flat_h, flat_a = mm.shrink_totals(comp, lam_h, lam_a, 0.0, {"liga": 2.5})
    np.testing.assert_allclose(flat_h + flat_a, [2.5, 2.5])
    np.testing.assert_allclose(flat_h / flat_a, lam_h / lam_a)


# --------------------------------------------------------------------------
# Recién ascendidos: nombres entre divisiones
# --------------------------------------------------------------------------


def _season_rows(home: str, away: str, date: str) -> dict:
    return {"home": home, "away": away, "datetime": pd.Timestamp(date)}


def test_map_team_names_links_same_club_across_divisions():
    upper = pd.DataFrame([_season_rows("Leeds", "Arsenal", "2025-09-01"), _season_rows("Sheffield United", "Arsenal", "2023-09-01"),
                          _season_rows("Burnley", "Arsenal", "2025-10-01")])
    lower = pd.DataFrame([_season_rows("Leeds United", "Hull City", "2024-09-01"),
                          _season_rows("Sheffield Wednesday", "Hull City", "2024-10-01"),
                          _season_rows("Burnley", "Hull City", "2025-11-01")])  # misma temporada: no es el mismo club
    mapping = comps.map_team_names(lower, upper)
    assert mapping == {"Leeds United": "Leeds"}


def test_team_similarity_examples():
    assert mm.team_similarity("Club Leon", "León") == 1.0
    assert mm.team_similarity("Wolves", "Wolverhampton Wanderers") == 1.0  # alias
    assert mm.team_similarity("Leeds", "Leeds United") >= comps.MIN_MAPPING_SIMILARITY
    assert mm.team_similarity("Sheffield United", "Sheffield Wednesday") < comps.MIN_MAPPING_SIMILARITY


# --------------------------------------------------------------------------
# Backtest contra el mercado
# --------------------------------------------------------------------------


def test_football_data_parsing_prefers_pinnacle_and_drops_corrupt_odds():
    from src import football_data_source as fds
    raw = pd.DataFrame({
        "Date": ["15/08/2025", "16/08/2025", "17/08/2025"], "HomeTeam": ["Liverpool", "Man United", "Wolves"],
        "AwayTeam": ["Bournemouth", "Arsenal", "Leeds"], "FTHG": [4, 0, 1], "FTAG": [2, 1, 1],
        "PSCH": [1.29, 4.0, 9.0], "PSCD": [6.55, 3.8, 9.0], "PSCA": [9.75, 1.9, 9.0],  # 3.ª: suma 0.33, corrupta
        "AvgCH": [1.29, 4.1, 2.6], "AvgCD": [6.0, 3.7, 3.3], "AvgCA": [8.7, 1.85, 2.9],
        "PC>2.5": [1.5, 2.0, 2.1], "PC<2.5": [2.6, 1.85, 1.75],
    })
    df = fds._standardize(raw, "HomeTeam", "AwayTeam", "FTHG", "FTAG")
    assert list(df["odds_source"]) == ["Pinnacle (cierre)", "Pinnacle (cierre)", "media del mercado (cierre)"]
    assert df.loc[2, "odds_h"] == 2.6
    assert df.loc[0, "date"] == pd.Timestamp("2025-08-15")
    assert df["odds_over25"].notna().all()


def test_match_predictions_uses_date_score_and_names():
    from src import backtest
    preds = pd.DataFrame({"datetime": pd.to_datetime(["2025-08-16 19:00", "2025-08-17 14:00"]),
                          "home": ["Manchester United", "Wolverhampton Wanderers"], "away": ["Arsenal", "Leeds"],
                          "hg": [0.0, 1.0], "ag": [1.0, 1.0]})
    fd_rows = pd.DataFrame({"date": pd.to_datetime(["2025-08-17", "2025-08-17", "2025-08-17"]),
                            "home": ["Man United", "Wolves", "Chelsea"], "away": ["Arsenal", "Leeds", "Fulham"],
                            "hg": [0.0, 1.0, 1.0], "ag": [1.0, 1.0, 1.0], "odds_h": [4.0, 2.6, 1.5]})
    matched = backtest.match_predictions(preds, fd_rows)
    assert list(matched["home_fd"]) == ["Man United", "Wolves"]


def test_backtest_run_with_synthetic_odds(trained_league, monkeypatch):
    from src import backtest
    preds = trained_league.trained.val_predictions
    p = preds[[f"p_model_{c}" for c in mm.CLASSES]].to_numpy()
    odds = 1 / (p * 1.05)  # mercado "justo" igual al modelo con un 5% de margen
    fake = pd.DataFrame({"date": preds["datetime"].dt.normalize(), "home": preds["home"], "away": preds["away"],
                         "hg": preds["hg"], "ag": preds["ag"], "odds_h": odds[:, 0], "odds_d": odds[:, 1],
                         "odds_a": odds[:, 2], "odds_over25": np.nan, "odds_under25": np.nan,
                         "odds_source": "sintéticas"})
    monkeypatch.setattr(backtest.fd, "load", lambda code, start, end: (fake, ["https://example.invalid"]))
    r = backtest.run(trained_league)
    assert r.n_matched == r.n_val == len(preds)
    assert r.ll_market == pytest.approx(r.ll_model, abs=1e-9)  # mismas probabilidades una vez quitado el margen
    assert 0.0 <= r.alpha <= 1.0 and len(r.roi) == len(backtest.ROI_THRESHOLDS)


# --------------------------------------------------------------------------
# Señal de mercado (cuotas de cierre de partidos anteriores)
# --------------------------------------------------------------------------


def test_implied_goals_recovers_lambdas_from_1x2_and_over_under():
    from src import market_signal as ms
    lam_h, lam_a = np.array([1.9, 1.1, 0.7, 2.6]), np.array([0.8, 1.2, 1.9, 0.5])
    line = np.array([2.5, 2.5, 3.5, 1.5])
    p_h, p_a, p_over = ms._model_probs(lam_h, lam_a, line)
    got_h, got_a = ms.implied_goals(p_h, p_a, p_over, 2.7, line)
    np.testing.assert_allclose(got_h, lam_h, rtol=1e-3)
    np.testing.assert_allclose(got_a, lam_a, rtol=1e-3)
    # Sin Over/Under el reparto local/visitante sigue saliendo del 1X2.
    only_h, only_a = ms.implied_goals(p_h, p_a, np.full(4, np.nan), 2.7, line)
    np.testing.assert_allclose(np.log(only_h / only_a), np.log(lam_h / lam_a), atol=0.2)


def test_map_teams_by_results_ignores_names():
    from src import market_signal as ms
    rng = np.random.default_rng(3)
    ours_names = ["Manchester United", "Wolverhampton Wanderers", "Nottingham Forest", "Atletico Madrid"]
    theirs_names = {"Manchester United": "Man United", "Wolverhampton Wanderers": "Wolves",
                    "Nottingham Forest": "Nott'm Forest", "Atletico Madrid": "Ath Madrid"}
    rows = []
    for k in range(24):
        h, a = rng.choice(ours_names, 2, replace=False)
        rows.append({"date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=7 * k), "home": h, "away": a,
                     "hg": float(rng.integers(0, 4)), "ag": float(rng.integers(0, 4))})
    ours = pd.DataFrame(rows)
    theirs = ours.assign(home=ours["home"].map(theirs_names), away=ours["away"].map(theirs_names),
                         date=ours["date"] + pd.Timedelta(days=1))
    assert ms.map_teams(ours, theirs) == {v: k for k, v in theirs_names.items()}


def _with_market(data: mm.LeagueData, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Partidos de la liga sintética con los λ "verdaderos" como señal de mercado (desde `since`)."""
    m = data.matches.copy()
    s = {f"Team{i}": 1.6 - 0.1 * i for i in range(12)}
    played = m["played"] & ((m["datetime"] >= since) if since is not None else True)
    m["h_mkt"] = np.where(played, 1.5 * m["home"].map(s) / m["away"].map(s), np.nan)
    m["a_mkt"] = np.where(played, 1.2 * m["away"].map(s) / m["home"].map(s), np.nan)
    return m


def test_market_signal_is_learned_when_history_has_odds():
    data = _fake_league()
    with_mkt = mm.build_league_data(data.competition, data.name, _with_market(data), data.sources, "xG",
                                    data.training_since)
    assert with_mkt.market_mode == "learn" and with_mkt.index.has_market
    t = mm.train_league(with_mkt).trained
    assert t.market_weight in mm.MARKET_WEIGHT_GRID and t.market_weight >= 0.5  # λ exactos: el mercado manda
    assert mm.train_league(data).trained.market_weight == 0.0


def test_market_signal_with_recent_odds_only_uses_fixed_weight():
    data = _fake_league()
    last = data.matches["datetime"].max()
    recent_only = _with_market(data, since=last - pd.Timedelta(days=200))
    fixed = mm.build_league_data(data.competition, data.name, recent_only, data.sources, "xG", data.training_since)
    assert fixed.market_mode == "fixed"
    model = mm.train_league(fixed)
    assert model.trained.market_weight == mm.MARKET_WEIGHT_DEFAULT
    pred = mm.predict_match(model, "Team0", "Team11", cutoff=last + pd.Timedelta(days=1))
    assert pred.home_snap["mkt_share"] >= 0.8 and pred.lam_home > pred.lam_away


def test_market_weight_scales_with_recent_odds_share():
    snap = {"n_recent": 10, "mkt_share": 0.0}
    for kind, value in (("sig", 1.3), ("gls", 1.1), ("mkt", 2.0)):
        snap[f"att_recent_{kind}"] = snap[f"att_long_{kind}"] = value
    without = mm.strength(snap, 5.0, 0.5, "att")
    assert mm.strength(snap, 5.0, 0.5, "att", market_weight=0.8) == pytest.approx(without)
    snap["mkt_share"] = 1.0
    assert mm.strength(snap, 5.0, 0.5, "att", market_weight=1.0) == pytest.approx(2.0)


def test_espn_past_odds_parsing_and_cache(tmp_path, monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, event_id):
            self.event_id = event_id

        def raise_for_status(self):
            pass

        def json(self):
            if self.event_id == "2":
                return {"pickcenter": []}
            return {"pickcenter": [{"provider": {"name": "DraftKings"}, "overUnder": 3.5, "overOdds": 150.0,
                                    "underOdds": -200.0, "homeTeamOdds": {"moneyLine": -150},
                                    "awayTeamOdds": {"moneyLine": 400}, "drawOdds": {"moneyLine": 280.0}}]}

    class FakeSession:
        def get(self, url, params=None, headers=None, timeout=None):
            calls.append(params["event"])
            return FakeResponse(params["event"])

    monkeypatch.setattr(espn_source, "CACHE_DIR", tmp_path)
    odds, from_cache = espn_source.fetch_past_odds("col.1", ["1", "2"], FakeSession())
    assert not from_cache and odds["2"] is None
    assert odds["1"]["home"] == pytest.approx(1 + 100 / 150) and odds["1"]["away"] == pytest.approx(5.0)
    assert odds["1"]["line"] == 3.5 and odds["1"]["under"] == pytest.approx(1.5)
    again, from_cache = espn_source.fetch_past_odds("col.1", ["1", "2"], FakeSession())
    assert from_cache and again == odds and sorted(calls) == ["1", "2"]  # la segunda vez, de la caché


def test_backtest_uses_espn_odds_where_football_data_has_none(trained_league, monkeypatch):
    from src import backtest
    preds = trained_league.trained.val_predictions
    p = preds[[f"p_model_{c}" for c in mm.CLASSES]].to_numpy()
    odds = {str(i).removeprefix("espn:"): {"home": 1 / (r[0] * 1.05), "draw": 1 / (r[1] * 1.05),
                                           "away": 1 / (r[2] * 1.05), "line": 2.5, "over": 1.9, "under": 1.9}
            for i, r in zip("espn:" + preds["id"].astype(str), p)}
    model = mm.LeagueModel(trained_league.data, trained_league.trained)
    model.trained.val_predictions = preds.assign(id="espn:" + preds["id"].astype(str), competition="col.1")
    model.data.competition = "col.1"
    try:
        monkeypatch.setattr(backtest.espn_source, "fetch_past_odds",
                            lambda code, ids: ({i: odds.get(i) for i in ids}, True))
        r = backtest.run(model)
    finally:
        model.trained.val_predictions = preds
        model.data.competition = "eng.1"
    assert r.odds_source == "DraftKings (ESPN)" and r.n_matched == len(preds)
    assert r.ll_market == pytest.approx(r.ll_model, abs=1e-9)


# --------------------------------------------------------------------------
# Alineaciones (rotaciones del once titular)
# --------------------------------------------------------------------------


def test_rotation_weights_missing_regulars_by_starts():
    from src import lineups
    regulars = [f"p{i}" for i in range(11)]
    previous = [regulars] * 8 + [regulars[:10] + ["sub"]] * 2  # p10 fue titular 8 de 10
    value, missing = lineups.rotation(previous, set(regulars))
    assert value == 0.0 and missing == []
    value, missing = lineups.rotation(previous, set(regulars[2:]) | {"sub", "x"})
    assert missing == ["p0", "p1"] and value == pytest.approx(20 / (10 * 10 + 8))


def test_shift_home_away_moves_probability_to_away():
    p = np.array([0.5, 0.25, 0.25])
    assert mm.shift_home_away(p, 0.0) == pytest.approx(p)
    q = mm.shift_home_away(p, 0.3)
    assert q.sum() == pytest.approx(1.0) and q[0] < p[0] and q[2] > p[2]
    assert np.log(q[0] / q[2]) == pytest.approx(np.log(p[0] / p[2]) - 0.6)


def test_parse_lineups_needs_both_elevens():
    def team(side, n):
        return {"homeAway": side, "roster": [{"starter": i < n, "athlete": {"id": f"{side}{i}", "displayName": f"J{i}"}}
                                             for i in range(18)]}
    xi = espn_source.parse_lineups({"rosters": [team("home", 11), team("away", 11)]})
    assert len(xi["home"]) == 11 and xi["away"][0] == ["away0", "J0"]
    assert espn_source.parse_lineups({"rosters": [team("home", 11)]}) is None
    assert espn_source.parse_lineups({"rosters": [team("home", 11), team("away", 3)]}) is None
    assert espn_source.parse_lineups({}) is None


def test_match_lineups_detects_rotation(monkeypatch):
    from src import lineups
    kickoff = pd.Timestamp("2026-10-10 15:00")
    events = [{"id": f"e{k}", "date": f"2026-09-{k + 1:02d}T15:00Z", "completed": True,
               "home": {"id": "H" if k % 2 == 0 else "X"}, "away": {"id": "A" if k % 2 == 0 else "Y"}}
              for k in range(12)]
    events += [{"id": f"f{k}", "date": f"2026-09-{k + 1:02d}T18:00Z", "completed": True,
                "home": {"id": "Y"}, "away": {"id": "H" if k % 2 else "A"}} for k in range(12)]
    regular = {t: [[f"{t}{i}", f"{t} jugador {i}"] for i in range(11)] for t in ("H", "A")}

    def fake_lineups(slug, ids, completed, session=None):
        out = {}
        for e in ids:
            if e == "now":  # el local sale con 4 suplentes; el visitante, con su once habitual
                subs = [[f"Hs{i}", f"suplente {i}"] for i in range(4)]
                out[e] = {"home": regular["H"][:7] + subs, "away": regular["A"]}
            else:
                ev = next(x for x in events if x["id"] == e)
                out[e] = {s: regular.get(ev[s]["id"], [[f"o{i}", "otro"] for i in range(11)]) for s in ("home", "away")}
        return out

    monkeypatch.setattr(lineups.espn_source, "fetch_year",
                        lambda session, slug, year, cur: (events if year == 2026 else [], None))
    monkeypatch.setattr(lineups.espn_source, "fetch_lineups", fake_lineups)
    info = lineups.match_lineups("eng.1", "now", "H", "A", kickoff)
    assert info.home.n_previous == lineups.N_PREVIOUS and info.away.rotation == 0.0
    assert info.home.rotation == pytest.approx(4 / 11) and len(info.home.missing) == 4
    assert info.shift > 0  # el local rota: el 1X2 se mueve hacia el visitante


def test_fixture_lineups_only_for_leagues_near_kickoff(monkeypatch):
    from src import lineups
    calls = []
    monkeypatch.setattr(lineups, "match_lineups", lambda *a, **k: calls.append(a) or "xi")
    now = pd.Timestamp("2026-10-10 12:00")
    row = pd.Series({"datetime": now + pd.Timedelta(hours=1), "id": "espn:123", "home_id": "1", "away_id": "2"})
    assert lineups.fixture_lineups("eng.1", False, row, now) == "xi" and calls[0][:4] == ("eng.1", "123", "1", "2")
    assert lineups.fixture_lineups("uefa.champions", True, row, now) is None  # copas: no
    tomorrow = row.copy()
    tomorrow["datetime"] = now + pd.Timedelta(days=1)  # aún no hay alineaciones: ni se consultan
    assert lineups.fixture_lineups("eng.1", False, tomorrow, now) is None
    us_row = pd.Series({"datetime": now, "id": "us:9", "espn_id": "espn:77", "espn_home_id": "5", "espn_away_id": "6"})
    assert lineups.espn_refs(us_row) == ("77", "5", "6")
    assert lineups.espn_refs(pd.Series({"datetime": now, "id": "us:9"})) is None


def test_predict_match_applies_lineup_shift(trained_league):
    from src import lineups
    rotated = lineups.LineupInfo(lineups.TeamLineup(["a"] * 11, 0.5, ["x"], 10),
                                 lineups.TeamLineup(["b"] * 11, 0.0, [], 10))
    base = mm.predict_match(trained_league, "Team0", "Team11")
    pred = mm.predict_match(trained_league, "Team0", "Team11", lineups=rotated)
    np.testing.assert_allclose(pred.p_before_lineups, base.p_final)
    assert pred.p_final[0] < base.p_final[0] and pred.p_final.sum() == pytest.approx(1.0)
    text = modelo_prediccion.report(pred, trained_league.data, detail=False)
    assert "ALINEACIONES CONFIRMADAS" in text and "con el ajuste por alineaciones" in text
