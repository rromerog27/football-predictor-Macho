"""Fuente Understat: xG partido a partido de las 6 ligas que cubre.

Descarga `https://understat.com/getLeagueData/<liga>/<temporada>` (el JSON que
usa su web) y lo convierte en las columnas estándar de `src/match_model.py`,
con el xG como señal de calidad de ocasiones. Caché en disco: las temporadas
pasadas no caducan; la temporada en curso, a las 3 horas.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.match_model import PredictionError, SourceInfo

UNDERSTAT = "https://understat.com"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) football-predictor-macho",
    "X-Requested-With": "XMLHttpRequest",
}
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "understat_cache"
CURRENT_SEASON_CACHE_TTL_S = 3 * 3600


def season_label(season: int) -> str:
    return f"{season}/{(season + 1) % 100:02d}"


def current_season_for(now: pd.Timestamp) -> int:
    """Understat nombra cada temporada por su año de inicio (2026 = 2026/27)."""
    return now.year if now.month >= 7 else now.year - 1


def fetch_season(session: requests.Session, league: str, season: int, current_season: int,
                 refresh: bool = False) -> tuple[list[dict], SourceInfo]:
    url = f"{UNDERSTAT}/getLeagueData/{league}/{season}"
    cache_file = CACHE_DIR / f"{league}_{season}.json"
    if not refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if season < current_season or age < CURRENT_SEASON_CACHE_TTL_S:
            dates = json.loads(cache_file.read_text(encoding="utf-8"))
            return dates, SourceInfo(url, season_label(season), sum(bool(m.get("isResult")) for m in dates), True)

    headers = {**HTTP_HEADERS, "Referer": f"{UNDERSTAT}/league/{league}/{season}"}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = session.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            dates = resp.json()["dates"]
            break
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    else:
        raise PredictionError(f"No se pudo descargar {url}: {last_error}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(dates), encoding="utf-8")
    return dates, SourceInfo(url, season_label(season), sum(bool(m.get("isResult")) for m in dates), False)


def matches_frame(dates_by_season: dict[int, list[dict]], competition: str) -> pd.DataFrame:
    """JSON `dates` de Understat → columnas estándar (horas en UTC; señal = xG)."""
    rows = []
    for season, dates in dates_by_season.items():
        for m in dates:
            played = bool(m.get("isResult"))
            xg = m.get("xG") or {}
            rows.append(
                {
                    "id": f"us:{m['id']}",
                    "competition": competition,
                    "season": season,
                    "datetime": pd.Timestamp(m["datetime"]),
                    "home": m["h"]["title"],
                    "away": m["a"]["title"],
                    "played": played,
                    "extra_time": False,
                    "neutral": False,
                    "hg": float(m["goals"]["h"]) if played else np.nan,
                    "ag": float(m["goals"]["a"]) if played else np.nan,
                    "h_sig": float(xg["h"]) if played and xg.get("h") is not None else np.nan,
                    "a_sig": float(xg["a"]) if played and xg.get("a") is not None else np.nan,
                    "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan,
                    "odds_over25": np.nan, "odds_under25": np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["datetime", "id"]).reset_index(drop=True)


def load(league: str, competition: str, seasons: int, refresh: bool = False, now: pd.Timestamp | None = None,
         progress=None) -> tuple[pd.DataFrame, list[SourceInfo], int]:
    """Temporada en curso y `seasons` anteriores. Devuelve (partidos, fuentes, temporada actual)."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    current = current_season_for(now)
    session = requests.Session()
    dates_by_season, sources = {}, []
    for season in range(current, current - seasons - 1, -1):
        if progress:
            progress(f"Descargando Understat {league} {season_label(season)}...")
        dates, info = fetch_season(session, league, season, current, refresh)
        if dates:
            dates_by_season[season] = dates
            sources.append(info)
    if not dates_by_season:
        raise PredictionError(f"Understat no devolvió partidos de {league}.")
    return matches_frame(dates_by_season, competition), sources, current
