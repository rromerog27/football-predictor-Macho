"""Fuente ESPN: resultados, tiros, calendario y cuotas de ~40 ligas y copas.

Usa el marcador público de ESPN
(`https://site.api.espn.com/apis/site/v2/sports/soccer/<slug>/scoreboard`):
`dates=AAAA&limit=1000` devuelve el año natural completo y `dates=AAAAMMDD` un
día. Cada evento trae goles, tiros y tiros a puerta de los partidos jugados,
si se jugó en campo neutral y, en los partidos por jugar, las cuotas de
DraftKings (moneyline 1X2 y Over/Under).

Como no hay xG, la señal de calidad de ocasiones es un **xG aproximado**:
`XG_PER_SHOT_ON_TARGET · tiros a puerta + XG_PER_SHOT_OFF_TARGET · tiros fuera`.
Los coeficientes salen de una regresión del xG de Understat sobre los tiros de
ESPN en 5.838 partidos-equipo de las 5 grandes ligas (2025-2026), cruzados por
fecha y nombre: el xG aproximado correlaciona 0.73 con el xG real (los goles,
0.61), con coeficientes estables entre ligas (0.21-0.25 por tiro a puerta).

Caché en disco recortada a los campos usados: los años pasados no caducan; el
año en curso, a las 3 horas.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.match_model import PredictionError, SourceInfo
from src.utils import write_atomic

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) football-predictor-macho"}
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "espn_cache"
CURRENT_YEAR_CACHE_TTL_S = 3 * 3600
XG_PER_SHOT_ON_TARGET = 0.2295
XG_PER_SHOT_OFF_TARGET = 0.0647
EXTRA_TIME_STATUSES = {"STATUS_FINAL_AET", "STATUS_FINAL_PEN"}
_ODDS_LOCKS: dict[str, threading.Lock] = {}  # una descarga de fichas (cuotas, alineaciones) por competición a la vez
_ODDS_LOCKS_GUARD = threading.Lock()
DROPPED_STATUS_WORDS = ("POSTPONED", "CANCELED", "CANCELLED", "ABANDONED", "SUSPENDED", "FORFEIT")


def american_to_decimal(value) -> float:
    """Cuota americana (+150, -275, "EVEN") → decimal (2.50, 1.36, 2.00)."""
    if value is None:
        return np.nan
    text = str(value).strip().upper()
    if text in ("EVEN", "EV"):
        return 2.0
    try:
        ml = float(text)
    except ValueError:
        return np.nan
    if ml == 0:
        return np.nan
    return 1 + ml / 100 if ml > 0 else 1 + 100 / abs(ml)


def _line_odds(node: dict | None) -> float:
    """Cuota de cierre (o de apertura si aún no hay cierre) de un nodo {open, close}."""
    if not node:
        return np.nan
    for key in ("close", "open"):
        if (node.get(key) or {}).get("odds") is not None:
            return american_to_decimal(node[key]["odds"])
    return np.nan


def _parse_odds(competition: dict) -> dict | None:
    odds_list = [o for o in (competition.get("odds") or []) if o]
    if not odds_list:
        return None
    o = odds_list[0]
    ml = o.get("moneyline") or {}
    parsed = {
        "home": _line_odds(ml.get("home")),
        "draw": american_to_decimal((o.get("drawOdds") or {}).get("moneyLine")),
        "away": _line_odds(ml.get("away")),
        "over25": np.nan,
        "under25": np.nan,
    }
    total = o.get("total") or {}
    if o.get("overUnder") == 2.5:
        parsed["over25"] = _line_odds(total.get("over"))
        parsed["under25"] = _line_odds(total.get("under"))
    return parsed


def _stat(competitor: dict, name: str) -> float | None:
    for s in competitor.get("statistics") or []:
        if s.get("name") == name:
            try:
                return float(s.get("displayValue"))
            except (TypeError, ValueError):
                return None
    return None


def trim_event(event: dict) -> dict | None:
    """Evento de ESPN → dict compacto con lo que usa el modelo (None si se descarta)."""
    status = (event.get("status") or {}).get("type") or {}
    name = status.get("name", "")
    if any(word in name for word in DROPPED_STATUS_WORDS):
        return None
    competition = (event.get("competitions") or [{}])[0]
    sides = {c.get("homeAway"): c for c in competition.get("competitors") or []}
    if "home" not in sides or "away" not in sides:
        return None

    def side(c: dict) -> dict:
        team = c.get("team") or {}
        score = c.get("score")
        return {
            "id": str(team.get("id")),
            "name": team.get("displayName") or team.get("name"),
            "abbr": team.get("abbreviation") or "",
            "score": float(score) if score not in (None, "") else None,
            "shots": _stat(c, "totalShots"),
            "sot": _stat(c, "shotsOnTarget"),
        }

    return {
        "id": str(event.get("id")),
        "date": event.get("date"),
        "status": name,
        "completed": bool(status.get("completed")),
        "neutral": bool(competition.get("neutralSite")),
        "season": (event.get("season") or {}).get("year"),
        "home": side(sides["home"]),
        "away": side(sides["away"]),
        "odds": _parse_odds(competition),
    }


def _get_events(session: requests.Session, slug: str, params: dict) -> list[dict]:
    url = f"{ESPN}/{slug}/scoreboard"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = session.get(url, params=params, headers=HTTP_HEADERS, timeout=60)
            resp.raise_for_status()
            return resp.json().get("events", [])
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise PredictionError(f"No se pudo descargar {url}?dates={params.get('dates')}: {last_error}")


def fetch_year(session: requests.Session, slug: str, year: int, current_year: int,
               refresh: bool = False) -> tuple[list[dict], SourceInfo]:
    url = f"{ESPN}/{slug}/scoreboard?dates={year}"
    cache_file = CACHE_DIR / f"{slug}_{year}.json"
    if not refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if year < current_year or age < CURRENT_YEAR_CACHE_TTL_S:
            events = json.loads(cache_file.read_text(encoding="utf-8"))
            return events, SourceInfo(url, str(year), sum(e["completed"] for e in events), True)

    raw = _get_events(session, slug, {"dates": str(year), "limit": 1000})
    if len(raw) >= 1000:  # el tope de la API: se baja mes a mes
        raw = [e for month in range(1, 13)
               for e in _get_events(session, slug, {"dates": f"{year}{month:02d}", "limit": 1000})]
    events = [t for e in raw if (t := trim_event(e)) is not None]
    write_atomic(cache_file, json.dumps(events))
    return events, SourceInfo(url, str(year), sum(e["completed"] for e in events), False)


def fetch_day(session: requests.Session, slug: str, day: str) -> list[dict]:
    """Partidos de un día (AAAAMMDD, fecha de ESPN en EE. UU.), sin caché en disco."""
    return [t for e in _get_events(session, slug, {"dates": day}) if (t := trim_event(e)) is not None]


def fetch_current(session: requests.Session, slug: str) -> list[dict]:
    """Marcador por defecto de ESPN: la jornada en curso o la próxima de la competición."""
    return [t for e in _get_events(session, slug, {}) if (t := trim_event(e)) is not None]


def _summary(session: requests.Session, slug: str, event_id: str) -> dict:
    """Ficha de un partido de ESPN (`.../summary?event=<id>`); lanza si la petición falla."""
    url = f"{ESPN}/{slug}/summary"
    for attempt in range(3):
        try:
            resp = session.get(url, params={"event": event_id}, headers=HTTP_HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def _summary_odds(session: requests.Session, slug: str, event_id: str) -> dict | None:
    """Cuotas previas (DraftKings) de un partido según su ficha de ESPN: 1X2 y la línea de
    Over/Under que ofrecía (2.5, 3.5...). None si no tiene; lanza si la petición falla."""
    data = _summary(session, slug, event_id)
    picks = [p for p in data.get("pickcenter") or [] if p]
    if not picks:
        return None
    o = picks[0]
    odds = {
        "home": american_to_decimal((o.get("homeTeamOdds") or {}).get("moneyLine")),
        "draw": american_to_decimal((o.get("drawOdds") or {}).get("moneyLine")),
        "away": american_to_decimal((o.get("awayTeamOdds") or {}).get("moneyLine")),
        "line": o.get("overUnder"),
        "over": american_to_decimal(o.get("overOdds")),
        "under": american_to_decimal(o.get("underOdds")),
    }
    if not all(np.isfinite(odds[k]) and odds[k] > 1 for k in ("home", "draw", "away")):
        return None
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in odds.items()}


def fetch_past_odds(slug: str, event_ids: list[str], session: requests.Session | None = None
                    ) -> tuple[dict[str, dict | None], bool]:
    """Cuotas previas de partidos ya jugados (ids de ESPN sin prefijo) y si todas salieron de la
    caché. ESPN las guarda en la ficha de cada partido desde finales de 2025: una petición por
    partido, con caché en disco sin caducidad (los partidos jugados no cambian). Las peticiones
    que fallan se reintentan en la siguiente carga."""
    cache_file = CACHE_DIR / f"odds_{slug}.json"
    with _ODDS_LOCKS_GUARD:
        lock = _ODDS_LOCKS.setdefault(slug, threading.Lock())
    with lock:  # otra competición con la misma liga en su pool puede estar descargándola
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
        missing = [e for e in dict.fromkeys(event_ids) if e not in cache]
        if missing:
            _download_odds(slug, missing, cache, cache_file, session)
    return {e: cache.get(e) for e in event_ids}, not missing


def _download_odds(slug: str, missing: list[str], cache: dict, cache_file: Path,
                   session: requests.Session | None) -> None:
    """Descarga las cuotas de `missing` en paralelo, las añade a `cache` y lo guarda."""
    session = session or requests.Session()

    def job(event_id: str):
        try:
            return event_id, _summary_odds(session, slug, event_id), True
        except (requests.RequestException, ValueError):
            return event_id, None, False

    with ThreadPoolExecutor(8) as pool:
        for event_id, odds, ok in pool.map(job, missing):
            if ok:
                cache[event_id] = odds
    write_atomic(cache_file, json.dumps(cache))


def parse_lineups(summary: dict) -> dict | None:
    """Titulares de cada equipo en la ficha de ESPN: {"home": [[id, nombre], ...], "away": [...]}.
    None si ESPN aún no publica las alineaciones (o no las tiene)."""
    xi = {}
    for team in summary.get("rosters") or []:
        starters = [[str(p["athlete"]["id"]), p["athlete"].get("displayName", "")]
                    for p in team.get("roster") or [] if p.get("starter") and p.get("athlete")]
        if team.get("homeAway") in ("home", "away") and len(starters) >= 10:
            xi[team["homeAway"]] = starters
    return xi if len(xi) == 2 else None


def fetch_lineups(slug: str, event_ids: list[str], completed: bool,
                  session: requests.Session | None = None) -> dict[str, dict | None]:
    """Titulares de varios partidos (ids de ESPN sin prefijo). Los de partidos terminados se guardan
    en disco sin caducidad; los de partidos por jugar no (las alineaciones salen ~1 h antes)."""
    cache_file = CACHE_DIR / f"xi_{slug}.json"
    with _ODDS_LOCKS_GUARD:
        lock = _ODDS_LOCKS.setdefault(f"xi:{slug}", threading.Lock())
    with lock:
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if completed and cache_file.exists() else {}
        missing = [e for e in dict.fromkeys(event_ids) if e not in cache]
        if missing:
            session = session or requests.Session()

            def job(event_id: str):
                try:
                    return event_id, parse_lineups(_summary(session, slug, event_id)), True
                except (requests.RequestException, ValueError):
                    return event_id, None, False

            with ThreadPoolExecutor(8) as pool:
                fetched = {e: xi for e, xi, ok in pool.map(job, missing) if ok}
            cache.update(fetched)
            if completed and fetched:
                write_atomic(cache_file, json.dumps(cache))
    return {e: cache.get(e) for e in event_ids}


def pseudo_xg(shots, sot):
    shots, sot = np.asarray(shots, dtype=float), np.asarray(sot, dtype=float)
    return XG_PER_SHOT_ON_TARGET * sot + XG_PER_SHOT_OFF_TARGET * np.clip(shots - sot, 0, None)


def events_frame(events: list[dict], competition: str) -> pd.DataFrame:
    """Eventos recortados → columnas estándar (señal = xG aproximado si hay tiros)."""
    rows = []
    for e in events:
        h, a, odds = e["home"], e["away"], e["odds"] or {}
        played = e["completed"] and h["score"] is not None and a["score"] is not None
        # Sin estadística, o 0 tiros de ambos equipos (ESPN rellena con ceros en algunas ligas): sin señal.
        has_shots = (all(v is not None for v in (h["shots"], h["sot"], a["shots"], a["sot"]))
                     and h["shots"] + a["shots"] > 0)
        rows.append(
            {
                "id": f"espn:{e['id']}",
                "competition": competition,
                "season": e["season"],
                "datetime": pd.Timestamp(e["date"]).tz_convert(None),
                "home": h["name"], "away": a["name"],
                "home_id": h["id"], "away_id": a["id"],
                "home_abbr": h["abbr"], "away_abbr": a["abbr"],
                "played": played,
                "extra_time": e["status"] in EXTRA_TIME_STATUSES,
                "neutral": e["neutral"],
                "hg": h["score"] if played else np.nan,
                "ag": a["score"] if played else np.nan,
                "h_sig": float(pseudo_xg(h["shots"], h["sot"])) if played and has_shots else np.nan,
                "a_sig": float(pseudo_xg(a["shots"], a["sot"])) if played and has_shots else np.nan,
                "odds_h": odds.get("home", np.nan), "odds_d": odds.get("draw", np.nan),
                "odds_a": odds.get("away", np.nan),
                "odds_over25": odds.get("over25", np.nan), "odds_under25": odds.get("under25", np.nan),
            }
        )
    columns = ["id", "competition", "season", "datetime", "home", "away", "home_id", "away_id", "home_abbr",
               "away_abbr", "played", "extra_time", "neutral", "hg", "ag", "h_sig", "a_sig", "odds_h", "odds_d",
               "odds_a", "odds_over25", "odds_under25"]
    return pd.DataFrame(rows, columns=columns)


def disambiguate_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Si dos equipos distintos (ids) comparten nombre en el conjunto, añade su abreviatura."""
    ids = pd.concat([frame[["home", "home_id", "home_abbr"]].set_axis(["name", "id", "abbr"], axis=1),
                     frame[["away", "away_id", "away_abbr"]].set_axis(["name", "id", "abbr"], axis=1)])
    dup = ids.groupby("name")["id"].nunique()
    dup = set(dup[dup > 1].index)
    if not dup:
        return frame
    label = {(r.name, r.id): f"{r.name} ({r.abbr or r.id})" for r in ids.drop_duplicates(["name", "id"]).itertuples()
             if r.name in dup}
    frame = frame.copy()
    frame["home"] = [label.get((n, i), n) for n, i in zip(frame["home"], frame["home_id"])]
    frame["away"] = [label.get((n, i), n) for n, i in zip(frame["away"], frame["away_id"])]
    return frame
