"""Pruebas de la sección FC 27 Mercado: lectura de FUT.GG, historial y señales.

Usan fragmentos de HTML con la misma forma que las páginas reales de FUT.GG,
así que no necesitan red.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src import fc27_history, fc27_market, fc27_signals

NOW = datetime(2026, 10, 1, 21, 0, tzinfo=timezone.utc)


def _mover(pct, price, ea_id, ovr, name, rarity, created):
    return (
        f'$R[1]={{upgrade:null,chemistry:0,momentumPercentage:{pct},currentDbPrice:{price},showIgsBlob:!1,'
        f'id:1,eaId:{ea_id},overall:{ovr},commonName:"{name}",cardName:"{name}",rarityName:"{rarity}",'
        f'url:"/players/1-x/27-{ea_id}/",uniqueClub:$R[2]={{id:4,url:"/clubs/1-x/"}},createdAt:"{created}",'
        f'dominantColor:"0f0d0d"}},'
    )


MOMENTUM_HTML = "".join([
    # FUT.GG guarda la variación con el signo invertido: 6 = el precio bajó un 6%.
    _mover(6, 920000, 50579475, 91, "Michael Olise", "Team of the week", "2026-09-23T17:00:00Z"),
    _mover(-30, 2300000, 226764, 90, "George Best", "Base Icon", "2026-09-15T17:00:00Z"),
    _mover(-4, 60000, 241651, 87, "Viktor Gy\\u00f6keres", "Team of the week", "2026-09-30T17:00:00Z"),
    _mover("null", 1000, 1, 80, "Sin dato", "Team of the week", "2026-09-30T17:00:00Z"),
])

CHEAPEST_HTML = (
    '{price:650,eaId:227174,name:"Cash",overall:81,position:3,url:"/x"},'
    '{price:650,eaId:230142,name:"Oyarzabal",overall:84,position:25,url:"/x"},'
    '{price:700,eaId:230143,name:"Catley",overall:84,position:5,url:"/x"},'
    '{price:1800,eaId:230144,name:"Foord",overall:85,position:12,url:"/x"},'
    '{price:3800,eaId:230145,name:"Katoto",overall:86,position:25,url:"/x"},'
    '{price:5000,eaId:230146,name:"Saka",overall:87,position:23,url:"/x"},'
)

SBC_HTML = (
    '$R[92]={id:5274,game:"27",eaId:39,slug:"27-39-bundesliga-potm-september",categoryEaId:1,'
    'name:"Bundesliga POTM September",description:"x",endTime:"2026-10-29T15:00:01Z",'
    'createdAt:"2026-10-01T15:00:23Z",isRepeatable:!1,url:"/sbc/players/27-39-bundesliga-potm-september/",'
    'scoreRequirement:700000,cost:565410,costPc:603285,awards:$R[95]=[$R[96]={id:24505,game:"27",count:1,'
    'isUntradeable:!0,other:null,pack:null,playerEaId:84133907,player:$R[97]={upgrade:null,eaId:84133907,'
    'overall:91,commonName:"Michael Olise",cardName:"Olise",rarityName:"POTM Bundesliga"}}]},'
    '$R[98]={id:5275,game:"27",eaId:31,slug:"27-31-83-upgrade",categoryEaId:2,name:"83+ Upgrade",'
    'endTime:"2026-10-06T17:00:00Z",createdAt:"2026-09-29T17:00:21Z",isRepeatable:!0,'
    'url:"/sbc/upgrades/27-31-83-upgrade/",scoreRequirement:2500,cost:2210,costPc:2345,awards:$R[99]=[]},'
)


@pytest.fixture
def snap() -> fc27_market.MarketSnapshot:
    return fc27_market.MarketSnapshot(
        NOW,
        fc27_market.parse_momentum(MOMENTUM_HTML),
        fc27_market.parse_cheapest(CHEAPEST_HTML),
        fc27_market.parse_sbcs(SBC_HTML),
    )


# ---------------------------------------------------------------------------
# Lectura de FUT.GG
# ---------------------------------------------------------------------------


def test_parse_momentum_flips_sign_and_skips_missing(snap):
    movers = snap.movers.set_index("name")
    assert len(movers) == 3  # la carta con momentum null se descarta
    assert movers.loc["Michael Olise", "pct_24h"] == -6.0
    assert movers.loc["George Best", "pct_24h"] == 30.0
    assert movers.loc["Michael Olise", "url"] == "/players/1-x/27-50579475/"
    assert "Viktor Gyökeres" in movers.index  # escapes unicode decodificados


def test_parse_sbcs_reads_cost_score_and_award(snap):
    potm = snap.sbcs.set_index("slug").loc["27-39-bundesliga-potm-september"]
    assert potm["score_requirement"] == 700000
    assert potm["cost"] == 565410 and potm["cost_pc"] == 603285
    assert potm["award_name"] == "Michael Olise" and potm["award_overall"] == 91
    assert bool(potm["award_untradeable"]) is True
    assert potm["category"] == "players"
    upgrade = snap.sbcs.set_index("slug").loc["27-31-83-upgrade"]
    assert bool(upgrade["repeatable"]) is True
    assert pd.isna(upgrade["award_name"])


def test_fodder_table_picks_cheapest_per_point(snap):
    table = fc27_market.fodder_table(snap.cheapest).set_index("overall")
    assert table.loc[84, "min_price"] == 650
    assert table.loc[84, "price"] == 675  # mediana de los más baratos (650 y 700)
    assert table.loc[84, "coins_per_point"] == pytest.approx(675 / 830, abs=1e-3)
    assert bool(table.loc[84, "is_best"]) is True
    assert 81 not in table.index  # sin Item Score conocido
    assert fc27_market.fodder_floor_price(snap.cheapest) == 650


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------


def _snap_at(ts: datetime, price: int) -> fc27_market.MarketSnapshot:
    movers = pd.DataFrame([{
        "ea_id": 7, "name": "Test", "overall": 88, "rarity": "Team of the week", "price": price,
        "pct_24h": 0.0, "url": "/players/7/", "created_at": "2026-09-01T00:00:00Z",
    }])
    empty = pd.DataFrame(columns=fc27_market.CHEAPEST_COLUMNS)
    return fc27_market.MarketSnapshot(ts, movers, empty, pd.DataFrame(columns=fc27_market.SBC_COLUMNS))


def test_save_snapshot_respects_min_gap():
    conn = fc27_history.connect(":memory:")
    assert fc27_history.save_snapshot(conn, _snap_at(NOW, 100))
    assert not fc27_history.save_snapshot(conn, _snap_at(NOW + timedelta(minutes=5), 100))
    assert fc27_history.save_snapshot(conn, _snap_at(NOW + timedelta(minutes=15), 100))
    assert fc27_history.snapshot_count(conn) == 2


def test_price_changes_uses_own_history():
    conn = fc27_history.connect(":memory:")
    fc27_history.save_snapshot(conn, _snap_at(NOW - timedelta(hours=6), 100_000))
    fc27_history.save_snapshot(conn, _snap_at(NOW - timedelta(hours=1), 110_000))
    fc27_history.save_snapshot(conn, _snap_at(NOW, 121_000))
    ch = fc27_history.price_changes(conn, NOW).set_index("ea_id")
    assert ch.loc[7, "pct_1h"] == pytest.approx(10.0)
    assert ch.loc[7, "pct_6h"] == pytest.approx(21.0)
    assert pd.isna(ch.loc[7, "pct_168h"])  # sin datos de hace 7 días: no se inventa


def test_load_latest_snapshot_roundtrip(snap):
    conn = fc27_history.connect(":memory:")
    fc27_history.save_snapshot(conn, snap)
    back = fc27_history.load_latest_snapshot(conn)
    assert back is not None
    assert set(back.movers["name"]) == set(snap.movers["name"])
    assert back.sbcs.iloc[0]["cost"] == snap.sbcs.iloc[0]["cost"]


def test_evaluate_signals_applies_ea_tax():
    conn = fc27_history.connect(":memory:")
    fc27_history.save_snapshot(conn, _snap_at(NOW, 100_000))
    fc27_history.save_snapshot(conn, _snap_at(NOW + timedelta(hours=72), 104_000))
    sig = pd.DataFrame([{"key": "card:7", "ea_id": 7, "name": "Test", "signal": "COMPRAR", "price": 100_000,
                         "market_score": 75, "risk_score": 30, "horizon_h": 72}])
    assert fc27_history.record_signals(conn, sig, NOW) == 1
    assert fc27_history.record_signals(conn, sig, NOW + timedelta(hours=1)) == 0  # no se duplica
    res = fc27_history.evaluate_signals(conn, NOW + timedelta(hours=80))
    assert len(res) == 1
    # +4% bruto no cubre el 5% de impuesto: la compra no acierta.
    assert not bool(res.iloc[0]["hit"])
    assert res.iloc[0]["net_return_pct"] == pytest.approx(-1.2)


# ---------------------------------------------------------------------------
# Señales
# ---------------------------------------------------------------------------


def test_classify_and_decide_signal():
    assert fc27_signals.classify(90) == "Señal muy fuerte"
    assert fc27_signals.classify(70) == "Interesante"
    assert fc27_signals.classify(39) == "Riesgo elevado"
    assert fc27_signals.decide_signal(72, 40) == "COMPRAR"
    assert fc27_signals.decide_signal(72, 50) == "VIGILAR"
    assert fc27_signals.decide_signal(60, 75) == "RIESGO"
    assert fc27_signals.decide_signal(45, 30) is None


def test_sbc_substitute_flags_more_expensive_card(snap):
    signals = fc27_signals.build_signals(snap, None, {"events": [], "notes": []}, NOW)
    olise = signals[signals["name"] == "Michael Olise"].iloc[0]
    assert olise["utility"] == 0
    assert olise["signal"] == "RIESGO"
    assert olise["risks"][0].startswith("Sustituto")  # el riesgo más fuerte va primero


def test_parabolic_rise_raises_risk(snap):
    signals = fc27_signals.build_signals(snap, None, {"events": [], "notes": []}, NOW)
    best = signals[signals["name"] == "George Best"].iloc[0]
    assert best["risk_score"] >= 70
    assert best["signal"] == "RIESGO"


def test_fodder_signal_at_floor_is_buy(snap):
    signals = fc27_signals.build_signals(snap, None, {"events": [], "notes": []}, NOW)
    fodder = signals[signals["key"] == "fodder:84"].iloc[0]
    assert fodder["price"] == 675
    assert fodder["risk_score"] == 20  # 675 está a menos del 8% del suelo (650)
    assert fodder["market_score"] >= 55
    plan = fc27_signals.trade_plan(fodder)
    assert set(plan) == {"zona", "objetivo", "invalidacion"}


def test_scores_stay_in_range(snap):
    signals = fc27_signals.build_signals(snap, None, {"events": [], "notes": []}, NOW)
    assert signals["market_score"].between(0, 100).all()
    assert signals["risk_score"].between(0, 100).all()


def test_promo_soon_lowers_upcoming_for_special_cards(snap):
    analyst = {"events": [{"when": (NOW + timedelta(hours=20)).isoformat(), "type": "promo",
                           "label": "Promo", "status": "ALTA", "source": "test"}], "notes": []}
    calm = fc27_signals.build_signals(snap, None, {"events": [], "notes": []}, NOW).set_index("key")
    busy = fc27_signals.build_signals(snap, None, analyst, NOW).set_index("key")
    key = "card:241651"
    assert busy.loc[key, "upcoming"] < calm.loc[key, "upcoming"]
    assert busy.loc[key, "risk_score"] > calm.loc[key, "risk_score"]


def test_expired_notes_are_hidden():
    analyst = {"notes": [
        {"player": "A", "expires": (NOW + timedelta(days=1)).isoformat()},
        {"player": "B", "expires": (NOW - timedelta(days=1)).isoformat()},
    ]}
    assert [n["player"] for n in fc27_signals.active_notes(analyst, NOW)] == ["A"]


def test_alerts_include_new_sbc_and_substitute(snap):
    signals = fc27_signals.build_signals(snap, None, {"events": [], "notes": []}, NOW)
    titles = [a.title for a in fc27_signals.build_alerts(snap, signals, None, {"events": []}, NOW)]
    assert any("Nuevo SBC: Bundesliga POTM September" in t for t in titles)
    assert any("Sustituto más barato para Michael Olise" in t for t in titles)


def test_fodder_table_ignores_single_outlier_listing():
    rows = [{"ea_id": i, "name": f"P{i}", "overall": 86, "price": p}
            for i, p in enumerate([2000, 3800, 3900, 3900, 4000, 4100])]
    table = fc27_market.fodder_table(pd.DataFrame(rows))
    assert table.iloc[0]["min_price"] == 2000
    assert table.iloc[0]["price"] == 3900


def test_watchlist_tracks_price_since_added():
    conn = fc27_history.connect(":memory:")
    fc27_history.save_snapshot(conn, _snap_at(NOW, 100_000))
    card = {"ea_id": 7, "name": "Test", "overall": 88, "rarity": "Team of the week", "price": 100_000}
    assert fc27_history.add_to_watchlist(conn, card, NOW)
    assert not fc27_history.add_to_watchlist(conn, card, NOW)  # no se duplica
    fc27_history.save_snapshot(conn, _snap_at(NOW + timedelta(hours=2), 110_000))
    wl = fc27_history.watchlist(conn).iloc[0]
    assert wl["last_price"] == 110_000
    assert wl["pct_since_added"] == pytest.approx(10.0)
    assert wl["net_if_sold_pct"] == pytest.approx(4.5)  # 110.000 × 0,95 / 100.000 − 1
    fc27_history.remove_from_watchlist(conn, 7)
    assert fc27_history.watchlist_ids(conn) == set()


def test_pc_trade_math_includes_ea_tax():
    m = fc27_history.pc_trade_math(100_000, 120_000)
    assert m["break_even"] == 105_263      # 100.000 / 0,95
    assert m["target"] == 115_789          # +10% neto
    assert m["stop"] == 90_000
    assert m["net_now"] == 14_000          # 120.000 × 0,95 − 100.000
    assert m["net_now_pct"] == pytest.approx(14.0)
    assert fc27_history.pc_trade_math(None, 120_000)["target"] is None


def test_watchlist_stores_user_pc_prices():
    conn = fc27_history.connect(":memory:")
    fc27_history.add_to_watchlist(conn, {"ea_id": 7, "name": "Test", "price": 100_000}, NOW)
    fc27_history.set_pc_prices(conn, 7, 150_000, 171_000, NOW)
    row = fc27_history.watchlist(conn).iloc[0]
    assert row["buy_price_pc"] == 150_000 and row["price_pc"] == 171_000
    assert row["pc_net_now"] == round(171_000 * 0.95 - 150_000)


def test_migration_adds_pc_columns_to_old_database(tmp_path):
    import sqlite3
    db = tmp_path / "old.sqlite"
    old = sqlite3.connect(db)
    old.execute("CREATE TABLE watchlist (ea_id INTEGER PRIMARY KEY, name TEXT NOT NULL, overall INTEGER, "
                "rarity TEXT, url TEXT, added_at TEXT NOT NULL, added_price INTEGER)")
    old.execute("INSERT INTO watchlist VALUES (1, 'Viejo', 85, 'TOTW', NULL, '2026-10-01T00:00:00+00:00', 5000)")
    old.commit(); old.close()
    conn = fc27_history.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(watchlist)")}
    assert {"buy_price_pc", "price_pc", "price_pc_at"} <= cols
    assert fc27_history.watchlist(conn).iloc[0]["name"] == "Viejo"  # no se pierden datos


def test_trade_log_profit_includes_ea_tax():
    conn = fc27_history.connect(":memory:")
    win = fc27_history.open_trade(conn, "Fodder de 84", 650, NOW, quantity=100, signal_at_buy="COMPRAR")
    loss = fc27_history.open_trade(conn, "Olise TOTW", 900_000, NOW, signal_at_buy="RIESGO")
    fc27_history.open_trade(conn, "Haaland", 110_000, NOW)  # sigue abierta
    fc27_history.close_trade(conn, win, 800, NOW + timedelta(days=2))
    fc27_history.close_trade(conn, loss, 850_000, NOW + timedelta(days=1))
    df = fc27_history.trades(conn).set_index("id")
    assert df.loc[win, "net_profit"] == 800 * 0.95 * 100 - 650 * 100      # +11.000
    assert df.loc[loss, "net_profit"] == 850_000 * 0.95 - 900_000         # −92.500
    assert df.loc[win, "break_even"] == round(650 / 0.95)
    s = fc27_history.trade_summary(fc27_history.trades(conn))
    assert s["closed"] == 2 and s["open"] == 1
    assert s["net_profit"] == 11_000 - 92_500
    assert s["win_rate_pct"] == 50.0
    assert s["capital_in_open"] == 110_000
    by_signal = fc27_history.trade_results_by_signal(fc27_history.trades(conn)).set_index("signal_at_buy")
    assert by_signal.loc["COMPRAR", "net_profit"] == 11_000


def test_trade_validation_and_delete():
    conn = fc27_history.connect(":memory:")
    with pytest.raises(ValueError):
        fc27_history.open_trade(conn, "", 1000, NOW)
    with pytest.raises(ValueError):
        fc27_history.open_trade(conn, "X", 0, NOW)
    tid = fc27_history.open_trade(conn, "X", 1000, NOW)
    fc27_history.close_trade(conn, tid, 1200, NOW)
    fc27_history.close_trade(conn, tid, 5000, NOW)  # una operación cerrada no se vuelve a cerrar
    assert fc27_history.trades(conn).iloc[0]["sell_price"] == 1200
    fc27_history.delete_trade(conn, tid)
    assert fc27_history.trades(conn).empty
    assert fc27_history.trade_summary(fc27_history.trades(conn))["roi_pct"] is None


def test_vigilar_needs_a_concrete_reason():
    assert fc27_signals.decide_signal(60, 30, has_reason=False) is None
    assert fc27_signals.decide_signal(60, 30, has_reason=True) == "VIGILAR"


def test_headline_alerts_only_substitutes_and_followed_cards():
    alerts = [
        fc27_signals.Alert("crítica", "CONFIRMADO", "Sustituto", "x", "sustituto", 1),
        fc27_signals.Alert("crítica", "CONFIRMADO", "Icono cae", "x", "movimiento_1h", 2),
        fc27_signals.Alert("aviso", "CONFIRMADO", "Mi carta cae", "x", "movimiento_1h", 3),
    ]
    titles = [a.title for a in fc27_signals.headline_alerts(alerts, followed={3})]
    assert titles == ["Sustituto", "Mi carta cae"]


def test_price_series_returns_chronological_prices():
    conn = fc27_history.connect(":memory:")
    fc27_history.save_snapshot(conn, _snap_at(NOW - timedelta(hours=2), 100))
    fc27_history.save_snapshot(conn, _snap_at(NOW - timedelta(hours=1), 120))
    fc27_history.save_snapshot(conn, _snap_at(NOW, 110))
    series = fc27_history.price_series(conn, [7, 999], NOW - timedelta(hours=3))
    assert series == {7: [100, 120, 110]}
    assert fc27_history.price_series(conn, [], NOW) == {}


def test_sparkline_svg_direction_color():
    from src import ui_theme
    up = ui_theme.sparkline_svg([1, 2, 3])
    down = ui_theme.sparkline_svg([3, 2, 1])
    assert up.startswith("<svg") and ui_theme.GREEN in up
    assert "#DC2626" in down
    assert ui_theme.sparkline_svg([5]) == ""
