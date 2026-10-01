"""Historial local del mercado de FC 27 (SQLite).

Cada vez que la app (o `scripts/fc27_snapshot.py`) descarga datos de FUT.GG,
se guarda una instantánea. Con varias instantáneas se pueden calcular las
variaciones de 1h, 6h, 3 días y 7 días que FUT.GG no ofrece, y comprobar a
posteriori si las señales de compra/riesgo acertaron (calibración).

El archivo vive en `data/fc27_market.sqlite` y está en `.gitignore`: es
historial personal, no se versiona.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.fc27_market import CHEAPEST_COLUMNS, MOVER_COLUMNS, SBC_COLUMNS, MarketSnapshot

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fc27_market.sqlite"

# Separación mínima entre dos instantáneas guardadas, para no llenar la base
# con copias casi idénticas si la página se recarga muchas veces seguidas.
MIN_SNAPSHOT_GAP_MIN = 10

# Ventanas de variación calculadas con el historial propio (en horas).
CHANGE_WINDOWS_H = (1, 6, 24, 72, 168)

EA_TAX = 0.05

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS prices (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
    source TEXT NOT NULL,
    ea_id INTEGER NOT NULL,
    name TEXT, overall INTEGER, rarity TEXT,
    price INTEGER NOT NULL, pct_24h REAL, url TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_prices_ea ON prices(ea_id, snapshot_id);
CREATE TABLE IF NOT EXISTS sbcs (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
    slug TEXT NOT NULL, name TEXT, category TEXT, url TEXT,
    end_time TEXT, created_at TEXT, repeatable INTEGER,
    score_requirement INTEGER, cost INTEGER, cost_pc INTEGER,
    award_ea_id INTEGER, award_name TEXT, award_overall INTEGER,
    award_rarity TEXT, award_untradeable INTEGER
);
CREATE TABLE IF NOT EXISTS watchlist (
    ea_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    overall INTEGER, rarity TEXT, url TEXT,
    added_at TEXT NOT NULL,
    added_price INTEGER
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    key TEXT NOT NULL,
    ea_id INTEGER,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    price INTEGER NOT NULL,
    market_score INTEGER, risk_score INTEGER, horizon_h INTEGER
);
"""


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Abre (y crea si hace falta) la base del historial."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.executescript(_SCHEMA)
    return conn


def last_snapshot_time(conn: sqlite3.Connection) -> datetime | None:
    row = conn.execute("SELECT MAX(fetched_at) FROM snapshots").fetchone()
    return _parse(row[0]) if row and row[0] else None


def snapshot_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]


def save_snapshot(conn: sqlite3.Connection, snap: MarketSnapshot, min_gap_min: int = MIN_SNAPSHOT_GAP_MIN) -> bool:
    """Guarda la instantánea si han pasado al menos `min_gap_min` minutos desde
    la anterior. Devuelve True si se guardó."""
    if snap.movers.empty and snap.cheapest.empty:
        return False
    last = last_snapshot_time(conn)
    if last is not None and snap.fetched_at - last < timedelta(minutes=min_gap_min):
        return False
    with conn:
        cur = conn.execute("INSERT OR IGNORE INTO snapshots(fetched_at) VALUES (?)", (_iso(snap.fetched_at),))
        if cur.rowcount == 0:
            return False
        sid = cur.lastrowid
        movers = snap.movers.assign(source="momentum")
        cheapest = snap.cheapest.assign(source="cheapest", rarity=None, pct_24h=None, url=None, created_at=None)
        for df in (movers, cheapest):
            if df.empty:
                continue
            conn.executemany(
                "INSERT INTO prices(snapshot_id, source, ea_id, name, overall, rarity, price, pct_24h, url, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (sid, r.source, int(r.ea_id), r.name, _int_or_none(r.overall), r.rarity, int(r.price),
                     _float_or_none(r.pct_24h), r.url, r.created_at)
                    for r in df.itertuples(index=False)
                ],
            )
        if not snap.sbcs.empty:
            cols = SBC_COLUMNS
            conn.executemany(
                f"INSERT INTO sbcs(snapshot_id, {', '.join(cols)}) VALUES (?{', ?' * len(cols)})",
                [(sid, *[_db_value(getattr(r, c)) for c in cols]) for r in snap.sbcs.itertuples(index=False)],
            )
    return True


def load_latest_snapshot(conn: sqlite3.Connection) -> MarketSnapshot | None:
    """Reconstruye la última instantánea guardada (modo sin conexión)."""
    row = conn.execute("SELECT id, fetched_at FROM snapshots ORDER BY fetched_at DESC LIMIT 1").fetchone()
    if not row:
        return None
    sid, fetched_at = row
    prices = pd.read_sql_query("SELECT * FROM prices WHERE snapshot_id = ?", conn, params=(sid,))
    movers = prices[prices["source"] == "momentum"][MOVER_COLUMNS].reset_index(drop=True)
    cheapest = prices[prices["source"] == "cheapest"][CHEAPEST_COLUMNS].reset_index(drop=True)
    sbcs = pd.read_sql_query("SELECT * FROM sbcs WHERE snapshot_id = ?", conn, params=(sid,))[SBC_COLUMNS]
    sbcs["repeatable"] = sbcs["repeatable"].astype(bool)
    return MarketSnapshot(_parse(fetched_at), movers, cheapest, sbcs, [])


def price_history(conn: sqlite3.Connection, ea_id: int) -> pd.DataFrame:
    """Serie de precios guardada de una carta: columnas fetched_at, price."""
    df = pd.read_sql_query(
        "SELECT s.fetched_at, MIN(p.price) AS price FROM prices p JOIN snapshots s ON s.id = p.snapshot_id "
        "WHERE p.ea_id = ? GROUP BY s.id ORDER BY s.fetched_at",
        conn, params=(int(ea_id),),
    )
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    return df


def _all_prices_since(conn: sqlite3.Connection, since: datetime) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT s.fetched_at, p.ea_id, MIN(p.price) AS price FROM prices p JOIN snapshots s ON s.id = p.snapshot_id "
        "WHERE s.fetched_at >= ? GROUP BY s.id, p.ea_id",
        conn, params=(_iso(since),),
    )
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    return df


def price_changes(conn: sqlite3.Connection, now: datetime, windows_h: tuple[int, ...] = CHANGE_WINDOWS_H) -> pd.DataFrame:
    """Variación % del precio de cada carta en cada ventana, usando solo el
    historial propio.

    Para la ventana de `w` horas se toma el último precio guardado entre
    `now - w - tolerancia` y `now - w`, con una tolerancia del 25% de la
    ventana (mínimo 30 minutos). Si no hay instantánea en ese tramo, el valor
    queda vacío: no se interpola.

    Columnas: ea_id, price_now y pct_<w>h por cada ventana.
    """
    tol = {w: timedelta(hours=max(0.5, w * 0.25)) for w in windows_h}
    oldest = now - timedelta(hours=max(windows_h)) - max(tol.values())
    df = _all_prices_since(conn, oldest)
    out_cols = ["ea_id", "price_now"] + [f"pct_{w}h" for w in windows_h]
    if df.empty:
        return pd.DataFrame(columns=out_cols)
    latest_ts = df["fetched_at"].max()
    current = df[df["fetched_at"] == latest_ts][["ea_id", "price"]].rename(columns={"price": "price_now"})
    for w in windows_h:
        target = latest_ts - timedelta(hours=w)
        window = df[(df["fetched_at"] <= target) & (df["fetched_at"] >= target - tol[w])]
        past = window.sort_values("fetched_at").groupby("ea_id", as_index=False).last()[["ea_id", "price"]]
        merged = current.merge(past, on="ea_id", how="left")
        current[f"pct_{w}h"] = ((merged["price_now"] / merged["price"] - 1) * 100).round(2).values
    return current[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mi lista (cartas que el usuario sigue)
# ---------------------------------------------------------------------------


def add_to_watchlist(conn: sqlite3.Connection, card: dict, now: datetime) -> bool:
    """Añade una carta a "Mi lista". `card` necesita ea_id, name y price;
    overall, rarity y url son opcionales. Devuelve False si ya estaba."""
    with conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO watchlist(ea_id, name, overall, rarity, url, added_at, added_price) "
            "VALUES (?,?,?,?,?,?,?)",
            (int(card["ea_id"]), card["name"], _int_or_none(card.get("overall")), card.get("rarity"),
             card.get("url"), _iso(now), _int_or_none(card.get("price"))),
        )
    return cur.rowcount == 1


def remove_from_watchlist(conn: sqlite3.Connection, ea_id: int) -> None:
    with conn:
        conn.execute("DELETE FROM watchlist WHERE ea_id = ?", (int(ea_id),))


def watchlist_ids(conn: sqlite3.Connection) -> set[int]:
    return {row[0] for row in conn.execute("SELECT ea_id FROM watchlist")}


def watchlist(conn: sqlite3.Connection) -> pd.DataFrame:
    """Cartas de "Mi lista" con su último precio guardado.

    Columnas: ea_id, name, overall, rarity, url, added_at, added_price,
    last_price, last_seen, pct_since_added (bruto) y net_if_sold_pct (vender
    ahora, tras el 5% de EA).
    """
    df = pd.read_sql_query(
        "SELECT w.*, "
        " (SELECT MIN(p.price) FROM prices p WHERE p.ea_id = w.ea_id AND p.snapshot_id = "
        "   (SELECT MAX(p2.snapshot_id) FROM prices p2 WHERE p2.ea_id = w.ea_id)) AS last_price, "
        " (SELECT s.fetched_at FROM snapshots s WHERE s.id = "
        "   (SELECT MAX(p3.snapshot_id) FROM prices p3 WHERE p3.ea_id = w.ea_id)) AS last_seen "
        "FROM watchlist w ORDER BY w.added_at DESC",
        conn,
    )
    added = pd.to_numeric(df["added_price"], errors="coerce")
    last = pd.to_numeric(df["last_price"], errors="coerce")
    df["pct_since_added"] = ((last / added - 1) * 100).round(2)
    df["net_if_sold_pct"] = ((last * (1 - EA_TAX) / added - 1) * 100).round(2)
    return df


# ---------------------------------------------------------------------------
# Señales y calibración
# ---------------------------------------------------------------------------


def record_signals(conn: sqlite3.Connection, signals: pd.DataFrame, created_at: datetime, min_gap_h: int = 6) -> int:
    """Guarda las señales COMPRAR / RIESGO para evaluarlas más adelante.

    No repite la misma señal (misma carta y tipo) si ya se registró en las
    últimas `min_gap_h` horas. Devuelve cuántas se guardaron.
    """
    if signals.empty:
        return 0
    since = _iso(created_at - timedelta(hours=min_gap_h))
    recent = {
        (k, kind) for k, kind in conn.execute("SELECT key, kind FROM signals WHERE created_at >= ?", (since,))
    }
    rows = [
        (_iso(created_at), r.key, _int_or_none(r.ea_id), r.name, r.signal, int(r.price),
         int(r.market_score), int(r.risk_score), int(r.horizon_h))
        for r in signals.itertuples(index=False)
        if r.signal in ("COMPRAR", "RIESGO") and (r.key, r.signal) not in recent and r.price and r.price > 0
    ]
    with conn:
        conn.executemany(
            "INSERT INTO signals(created_at, key, ea_id, name, kind, price, market_score, risk_score, horizon_h) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def evaluate_signals(conn: sqlite3.Connection, now: datetime) -> pd.DataFrame:
    """Resultado de cada señal cuyo horizonte ya pasó.

    COMPRAR acierta si vender al final del horizonte deja beneficio después
    del 5% de impuesto de EA. RIESGO acierta si el precio bajó.
    Columnas: created_at, name, kind, price, price_after, net_return_pct, hit.
    """
    sig = pd.read_sql_query("SELECT * FROM signals", conn)
    cols = ["created_at", "name", "kind", "price", "price_after", "net_return_pct", "hit"]
    if sig.empty:
        return pd.DataFrame(columns=cols)
    snaps = pd.read_sql_query("SELECT id, fetched_at FROM snapshots", conn)
    snaps["fetched_at"] = pd.to_datetime(snaps["fetched_at"], utc=True)
    results = []
    for s in sig.itertuples(index=False):
        due = _parse(s.created_at) + timedelta(hours=int(s.horizon_h))
        if due > now or s.ea_id is None or pd.isna(s.ea_id):
            continue
        tol = timedelta(hours=max(1, int(s.horizon_h) * 0.25))
        cand = snaps[(snaps["fetched_at"] >= due) & (snaps["fetched_at"] <= due + tol)].sort_values("fetched_at")
        price_after = None
        for sid in cand["id"]:
            row = conn.execute(
                "SELECT MIN(price) FROM prices WHERE snapshot_id = ? AND ea_id = ?", (int(sid), int(s.ea_id))
            ).fetchone()
            if row and row[0]:
                price_after = row[0]
                break
        if price_after is None:
            continue
        if s.kind == "COMPRAR":
            net = price_after * (1 - EA_TAX) / s.price - 1
            hit = net > 0
        else:
            net = price_after / s.price - 1
            hit = price_after < s.price
        results.append((s.created_at, s.name, s.kind, s.price, price_after, round(net * 100, 2), hit))
    return pd.DataFrame(results, columns=cols)


def _int_or_none(v):
    return None if v is None or pd.isna(v) else int(v)


def _float_or_none(v):
    return None if v is None or pd.isna(v) else float(v)


def _db_value(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v
