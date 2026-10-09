"""Alineaciones: ajuste de la predicción cuando un equipo rota su once titular.

Una buena parte de la ventaja de las cuotas de cierre sobre el modelo son las
alineaciones, que el mercado conoce ~1 h antes del partido: la diferencia
entre el mercado y el modelo correlaciona −0.3/−0.45 con las rotaciones.

Rotación de un equipo = titularidades (en sus 10 partidos anteriores de la
competición) de sus 11 jugadores más habituales que no salen de titulares,
divididas por las titularidades de esos 11: 0 es el once habitual y 0.5 que
falta la mitad de lo habitual. El 1X2 final se desplaza en log-odds
local/visitante en ±LINEUP_BETA · (rotación local − rotación visitante) / 2.

LINEUP_BETA es común a todas las ligas: estimado con validación cruzada en dos
mitades sobre 5.871 partidos de validación de 17 ligas (β = 0.66 y 1.06 en
cada mitad, 0.87 con todo; log loss −0.0021). Por liga el efecto es demasiado
ruidoso para estimarlo aparte. Solo se aplica en ligas (en copas, los
partidos previos de cada equipo están en otras competiciones).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
import requests

from src import espn_source
from src.match_model import PredictionError, utc_now

LINEUP_BETA = 0.85
N_PREVIOUS = 10  # partidos anteriores que definen los habituales
MIN_PREVIOUS = 5  # con menos partidos previos con alineación, la rotación no se calcula
LOOKAHEAD = pd.Timedelta(hours=2)  # antes, ESPN no publica alineaciones: no se consultan
LIVE_WINDOW = pd.Timedelta(hours=3)  # tras el inicio, se siguen mostrando mientras dura el partido


@dataclass
class TeamLineup:
    starters: list[str]  # nombres de los titulares
    rotation: float | None  # 0 = once habitual; None sin partidos previos suficientes
    missing: list[str] = field(default_factory=list)  # habituales que no son titulares (más habituales primero)
    n_previous: int = 0


@dataclass
class LineupInfo:
    home: TeamLineup
    away: TeamLineup

    @property
    def shift(self) -> float:
        """Desplazamiento del log-odds a favor del visitante (negativo: a favor del local)."""
        return LINEUP_BETA * ((self.home.rotation or 0.0) - (self.away.rotation or 0.0)) / 2


def rotation(previous: list[list[str]], starters: set[str]) -> tuple[float, list[str]]:
    """(rotación, ids de los habituales ausentes) del once `starters` frente a los partidos `previous`."""
    counts = Counter(p for xi in previous for p in xi)
    top = counts.most_common(11)
    missing = [p for p, _ in top if p not in starters]
    return sum(counts[p] for p in missing) / sum(c for _, c in top), missing


def _previous_events(slug: str, team_id: str, before: pd.Timestamp,
                     session: requests.Session) -> list[tuple[str, str]]:
    """(id, "home"/"away") de los últimos N_PREVIOUS partidos terminados del equipo en la competición
    antes de `before`."""
    current_year = utc_now().year
    events = [e for year in (before.year - 1, before.year)
              for e in espn_source.fetch_year(session, slug, year, current_year)[0]]
    mine = []
    for e in events:
        when = pd.Timestamp(e["date"]).tz_convert(None)
        if e["completed"] and when < before and team_id in (e["home"]["id"], e["away"]["id"]):
            mine.append((when, e["id"], "home" if e["home"]["id"] == team_id else "away"))
    return [(event_id, side) for _, event_id, side in sorted(mine)[-N_PREVIOUS:]]


def match_lineups(slug: str, event_id: str, home_id: str, away_id: str, kickoff: pd.Timestamp,
                  session: requests.Session | None = None) -> LineupInfo | None:
    """Alineaciones del partido y rotación de cada equipo; None si ESPN aún no las publica."""
    session = session or requests.Session()
    finished = kickoff < utc_now() - pd.Timedelta(hours=3)
    xi = espn_source.fetch_lineups(slug, [event_id], completed=finished, session=session)[event_id]
    if xi is None:
        return None
    teams = {}
    for side, team_id in (("home", str(home_id)), ("away", str(away_id))):
        previous = _previous_events(slug, team_id, kickoff, session)
        lineups = espn_source.fetch_lineups(slug, [pid for pid, _ in previous], completed=True, session=session)
        previous_xis = [[pid for pid, _ in lineups[eid][s]] for eid, s in previous if lineups.get(eid)]
        names = {pid: name for eid, s in previous if lineups.get(eid) for pid, name in lineups[eid][s]}
        names.update(dict(xi[side]))
        starters = [name for _, name in xi[side]]
        if len(previous_xis) < MIN_PREVIOUS:
            teams[side] = TeamLineup(starters, None, [], len(previous_xis))
            continue
        value, missing = rotation(previous_xis, {pid for pid, _ in xi[side]})
        teams[side] = TeamLineup(starters, value, [names.get(p, p) for p in missing], len(previous_xis))
    return LineupInfo(teams["home"], teams["away"])


def in_window(kickoff: pd.Timestamp, now: pd.Timestamp | None = None) -> bool:
    """Si tiene sentido consultar las alineaciones: desde LOOKAHEAD antes del inicio hasta
    LIVE_WINDOW después (más tarde, el ajuste ya no sirve para decidir nada)."""
    now = now if now is not None else utc_now()
    return -LIVE_WINDOW <= kickoff - now <= LOOKAHEAD


def espn_refs(fixture: pd.Series) -> tuple[str, str, str] | None:
    """(id del partido, id del local, id del visitante) en ESPN de una fila del calendario: columnas
    `espn_id`/`espn_home_id`/`espn_away_id` (ligas de Understat) o `id` "espn:..." con `home_id`/`away_id`."""
    def text(key: str) -> str | None:
        value = fixture.get(key)
        return value if isinstance(value, str) and value else None

    event = text("espn_id") or text("id")
    home, away = text("espn_home_id") or text("home_id"), text("espn_away_id") or text("away_id")
    if not event or not event.startswith("espn:") or not home or not away:
        return None
    return event.removeprefix("espn:"), home, away


def fixture_lineups(competition: str, is_cup: bool, fixture: pd.Series, now: pd.Timestamp | None = None,
                    session: requests.Session | None = None) -> LineupInfo | None:
    """Alineaciones de un partido del calendario si aplican: liga, dentro de la ventana de
    `in_window` y con ids de ESPN. None si no aplica, si ESPN aún no las publica o si la consulta falla."""
    refs = espn_refs(fixture) if fixture is not None else None
    if is_cup or refs is None or not in_window(fixture["datetime"], now):
        return None
    try:
        return match_lineups(competition, *refs, fixture["datetime"], session)
    except (requests.RequestException, PredictionError, KeyError, ValueError):
        return None
