"""Guarda una instantánea del mercado de FC 27 sin abrir la app.

Pensado para programarlo (cron, Programador de tareas de Windows) y que el
historial crezca aunque no tengas la app abierta. Ejemplo cada 30 minutos:

    */30 * * * * cd /ruta/al/proyecto && .venv/bin/python scripts/fc27_snapshot.py

Imprime un resumen corto y termina con código 1 si FUT.GG no respondió.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import fc27_history, fc27_market, fc27_signals  # noqa: E402


def main() -> int:
    snap = fc27_market.fetch_market_snapshot()
    for e in snap.errors:
        print(f"Aviso: {e}")
    if snap.is_empty:
        print("FUT.GG no respondió; no se guardó nada.")
        return 1
    conn = fc27_history.connect()
    saved = fc27_history.save_snapshot(conn, snap)
    if not saved:
        print("Ya hay una instantánea de hace menos de 10 minutos; no se guardó otra.")
        return 0
    changes = fc27_history.price_changes(conn, snap.fetched_at)
    signals = fc27_signals.build_signals(snap, changes, fc27_signals.load_analyst_file(), snap.fetched_at)
    recorded = fc27_history.record_signals(conn, signals, snap.fetched_at)
    counts = signals["signal"].value_counts()
    print(
        f"Instantánea guardada ({fc27_signals.format_when(snap.fetched_at)}): {len(snap.movers)} cartas, "
        f"{len(snap.sbcs)} SBC. Señales: {counts.get('COMPRAR', 0)} comprar, {counts.get('VIGILAR', 0)} vigilar, "
        f"{counts.get('RIESGO', 0)} riesgo ({recorded} registradas para calibrar). "
        f"Total en el historial: {fc27_history.snapshot_count(conn)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
