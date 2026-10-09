"""Partidos del día — predicciones automáticas para ~33 ligas y copas.

Página registrada en `app.py` (página de inicio). Lista los partidos de la
fecha elegida (en la zona horaria del navegador) de las competiciones
elegidas y muestra para cada uno la predicción del ensemble Poisson
(Dixon-Coles) + regresión logística de `src/match_model.py`, con las
probabilidades del mercado cuando ESPN publica cuotas. La sección "Analizar
un partido" corre el mismo modelo para cualquier cruce de una competición.

Cada competición se entrena una sola vez y queda en caché unas horas (la
primera visita tarda unos segundos por competición con partidos ese día);
predecir un partido es instantáneo. Este archivo solo presenta los datos: el
modelo vive en `src/match_model.py` y las fuentes en `src/competitions.py`.
"""

from __future__ import annotations

import html
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import requests
import streamlit as st

from src import backtest
from src import competitions as comps
from src import football_data_source as fd
from src import lineups
from src import match_model as mm

MODEL_TTL_S = 3 * 3600  # igual que la caché del año/temporada en curso: el modelo ve los resultados nuevos
DAY_TTL_S = 10 * 60
LINEUP_TTL_S = 5 * 60  # las alineaciones salen ~1 h antes del inicio: se vuelven a mirar cada 5 minutos
# Diferencia modelo − mercado que se señala: la media es ~5 puntos por resultado; 10 o más
# aparece en ~1 de cada 10 partidos (medido en una jornada de 39 partidos con cuotas).
VALUE_THRESHOLD = 0.10
TIMEZONES = [
    "Europe/Madrid", "Europe/London", "America/Mexico_City", "America/Bogota", "America/Lima",
    "America/Santiago", "America/Argentina/Buenos_Aires", "America/New_York", "UTC",
]
ORDERED_CODES = [c.code for region in comps.REGIONS for c in comps.COMPETITIONS if c.region == region]
ESPN_LOGO = "https://a.espncdn.com/i/teamlogos/soccer/500/{}.png"  # escudo por id de equipo de ESPN
WEEKDAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
          "noviembre", "diciembre"]
HIGHLIGHTS = ["Más claros", "Más parejos", "Vs mercado", "XI confirmados"]
SORTS = ["Hora", "Liga", "Más claros"]
N_HIGHLIGHTS = 4

PAGE_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: var(--fc-bg); }
/* Barra 1X2: escala divergente local (azul) ↔ visitante (naranja) con empate neutro. Validados con el
   validador de paleta (CVD ΔE ≥ 24, contraste ≥ 3:1) en claro (#2a78d6/#eb6834 sobre blanco) y en oscuro
   (#3b82e8/#e8693a sobre la superficie oscura); el gris es el punto medio de cada modo. */
:root { --pd-home: #2a78d6; --pd-draw: #CBD5E1; --pd-away: #eb6834; }
@media (prefers-color-scheme: dark) { :root { --pd-home: #3b82e8; --pd-draw: #475569; --pd-away: #e8693a; } }

/* -- Cabecera compacta -- */
.pd-head { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 4px 16px;
  margin-bottom: 10px; }
.pd-title { font-size: clamp(1.45rem, 2.4vw, 1.85rem); font-weight: 800; letter-spacing: -.02em; line-height: 1.1;
  color: var(--fc-text); }
.pd-sub { color: var(--fc-muted); font-size: .88rem; }
.pd-label { font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--fc-faint); }
.pd-note { font-size: .8rem; color: var(--fc-muted); line-height: 1.5; margin-bottom: 1rem; }
.pd-empty { border: 1px dashed var(--fc-border-strong); border-radius: var(--fc-radius); padding: 18px 20px;
  color: var(--fc-muted); font-size: .9rem; line-height: 1.5; background: var(--fc-surface); margin-bottom: 1rem; }
.pd-summary { font-size: .84rem; color: var(--fc-muted); margin: 2px 0 14px; }
.pd-summary b { color: var(--fc-text); font-weight: 700; }
.pd-section { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin: 18px 0 8px; }
.pd-section h3 { font-size: 1rem; font-weight: 800; color: var(--fc-text); margin: 0; padding: 0; letter-spacing: -.01em; }

/* -- Controles: fecha (‹ ›), buscador y ligas en una fila que se reparte en el móvil -- */
.st-key-pd_controls { gap: 8px; }
.st-key-pd_controls [data-testid="stDateInput"] { min-width: 150px; }

/* -- Escudos (ESPN); si la imagen no carga, se ven las iniciales (::before de la imagen rota) -- */
.pd-crest { width: 22px; height: 22px; flex: none; object-fit: contain; position: relative; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center; }
.pd-crest::before { content: attr(data-i); position: absolute; inset: 0; display: flex; align-items: center;
  justify-content: center; border-radius: 50%; background: var(--fc-surface-2); border: 1px solid var(--fc-border);
  color: var(--fc-muted); font-size: .55rem; font-weight: 800; letter-spacing: -.02em; }
span.pd-crest { background: var(--fc-surface-2); border: 1px solid var(--fc-border); color: var(--fc-muted);
  font-size: .55rem; font-weight: 800; }

/* -- Barra 1X2: segmentos separados por 2px de superficie, extremos redondeados a 4px -- */
.pd-bar { display: flex; gap: 2px; height: 8px; }
.pd-bar i { display: block; height: 100%; min-width: 3px; }
.pd-bar i:first-child { border-radius: 4px 0 0 4px; }
.pd-bar i:last-child { border-radius: 0 4px 4px 0; }
.pd-legend { display: flex; justify-content: space-between; gap: 6px; font-size: .78rem; color: var(--fc-muted);
  font-variant-numeric: tabular-nums; margin-top: 6px; }
.pd-legend span { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.pd-legend b { color: var(--fc-text); font-weight: 700; }
.pd-sw { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }

/* -- Píldoras -- */
.pd-pill { display: inline-flex; align-items: center; gap: 4px; font-size: .68rem; font-weight: 700; letter-spacing: .02em;
  padding: 3px 9px; border-radius: 999px; white-space: nowrap; max-width: 100%; overflow: hidden; text-overflow: ellipsis;
  background: var(--fc-surface-2); color: var(--fc-muted); border: 1px solid var(--fc-border); }
.pd-pill.ok { background: var(--fc-green-soft); color: var(--fc-green-ink); border-color: color-mix(in srgb, var(--fc-green) 25%, transparent); }
.pd-pill.miss { background: var(--fc-red-soft); color: var(--fc-red-ink); border-color: color-mix(in srgb, var(--fc-red) 25%, transparent); }
.pd-pill.pick { background: var(--fc-blue-soft); color: var(--fc-blue-ink); border-color: color-mix(in srgb, var(--fc-blue) 25%, transparent); }
.pd-pill.val { background: var(--fc-yellow-soft); color: var(--fc-yellow-ink);
  border-color: color-mix(in srgb, var(--fc-yellow) 30%, transparent); }
.pd-pill.live { background: var(--fc-red-soft); color: var(--fc-red-ink); border-color: transparent; }

/* -- Destacados -- */
.pd-hl { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 6px; }
.pd-hl-card { display: block; text-decoration: none !important; background: var(--fc-surface); border: 1px solid var(--fc-border);
  border-radius: var(--fc-radius); box-shadow: var(--fc-shadow); padding: 12px 14px; color: var(--fc-text) !important;
  transition: box-shadow var(--fc-ease), transform var(--fc-ease); min-width: 0; }
.pd-hl-card:hover { box-shadow: var(--fc-shadow-hover); transform: translateY(-1px); }
.pd-hl-top { font-size: .72rem; color: var(--fc-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pd-hl-teams { display: flex; flex-direction: column; gap: 4px; margin: 8px 0 10px; }
.pd-hl-team { display: flex; align-items: center; gap: 8px; font-size: .88rem; font-weight: 700; min-width: 0; }
.pd-hl-team span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pd-hl-main { display: flex; align-items: baseline; gap: 8px; margin-bottom: 8px; min-width: 0; }
.pd-hl-main b { font-size: 1.45rem; font-weight: 800; letter-spacing: -.02em; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pd-hl-main span { font-size: .78rem; color: var(--fc-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pd-hl-foot { font-size: .72rem; color: var(--fc-muted); margin-top: 6px; font-variant-numeric: tabular-nums; }

/* -- Tabla de partidos: cada fila se despliega con el análisis -- */
.pd-table { background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius);
  box-shadow: var(--fc-shadow); overflow: hidden; }
.pd-cols, .pd-row > summary { display: grid; align-items: center; gap: 14px;
  grid-template-columns: 86px minmax(0, 1.5fr) minmax(170px, 1fr) 52px minmax(120px, .8fr) 18px; padding: 10px 16px; }
.pd-cols { font-size: .68rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--fc-faint);
  border-bottom: 1px solid var(--fc-border); padding-top: 9px; padding-bottom: 9px; background: var(--fc-surface-2); }
.pd-cols .pd-c-probs { display: grid; grid-template-columns: repeat(3, 1fr); text-align: center; }
.pd-group { padding: 8px 16px; font-size: .74rem; font-weight: 800; letter-spacing: .04em; text-transform: uppercase;
  color: var(--fc-muted); background: var(--fc-surface-2); border-bottom: 1px solid var(--fc-border); }
.pd-row { border-bottom: 1px solid var(--fc-border); }
.pd-row:last-child { border-bottom: 0; }
.pd-row > summary { cursor: pointer; list-style: none; transition: background-color var(--fc-ease); }
.pd-row > summary::-webkit-details-marker { display: none; }
.pd-row > summary:hover { background: color-mix(in srgb, var(--fc-blue) 4%, transparent); }
.pd-row[open] > summary { background: color-mix(in srgb, var(--fc-blue) 6%, transparent); }
.pd-row:target > summary { box-shadow: inset 3px 0 0 var(--fc-blue); }
.pd-r-time { font-variant-numeric: tabular-nums; line-height: 1.25; min-width: 0; }
.pd-r-time b { display: block; font-size: .9rem; color: var(--fc-text); font-weight: 700; }
.pd-r-time span { display: block; font-size: .66rem; color: var(--fc-faint); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.pd-r-teams { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.pd-r-team { display: flex; align-items: center; gap: 8px; min-width: 0; font-size: .9rem; color: var(--fc-text); }
.pd-r-team span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pd-r-team em { font-style: normal; font-weight: 800; margin-left: auto; font-variant-numeric: tabular-nums; }
.pd-r-team.win span { font-weight: 700; }
.pd-r-probs { min-width: 0; }
.pd-r-nums { display: grid; grid-template-columns: repeat(3, 1fr); text-align: center; font-size: .86rem;
  font-variant-numeric: tabular-nums; color: var(--fc-muted); margin-bottom: 5px; }
.pd-r-nums b { color: var(--fc-text); font-weight: 800; }
.pd-r-over { font-size: .86rem; font-variant-numeric: tabular-nums; color: var(--fc-text); text-align: right; }
.pd-r-pick { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 4px; min-width: 0; }
.pd-r-pick .pd-pill { display: inline-block; }
.pd-chev { color: var(--fc-faint); font-size: .8rem; transition: transform var(--fc-ease); text-align: right; }
.pd-row[open] .pd-chev { transform: rotate(90deg); }
.pd-r-body { padding: 4px 16px 16px; border-top: 1px dashed var(--fc-border); }
.pd-r-body .pd-detail { max-width: 760px; }
.pd-r-market { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; font-size: .78rem; color: var(--fc-muted);
  margin: 10px 0 2px; font-variant-numeric: tabular-nums; }
.pd-r-market b { color: var(--fc-text); }

/* -- Tarjeta (sección "Analizar un partido") -- */
[class*="st-key-pd_card_"] { background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius);
  box-shadow: var(--fc-shadow); padding: 14px 16px 4px; gap: 4px; }
.st-key-pd_manual_form, .st-key-pd_perf_form { background: var(--fc-surface); border: 1px solid var(--fc-border);
  border-radius: var(--fc-radius); box-shadow: var(--fc-shadow); padding: 14px 16px 6px; }
.pd-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; }
.pd-time { font-size: .8rem; font-weight: 700; color: var(--fc-text); font-variant-numeric: tabular-nums; }
.pd-teams { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); align-items: center; gap: 8px;
  margin-bottom: 12px; }
.pd-team { min-width: 0; }
.pd-team.away { text-align: right; }
.pd-team-name { font-size: 1rem; font-weight: 700; color: var(--fc-text); line-height: 1.2; overflow-wrap: anywhere; }
.pd-team-xg { font-size: .74rem; color: var(--fc-muted); margin-top: 2px; font-variant-numeric: tabular-nums; }
.pd-score { font-size: 1.25rem; font-weight: 800; color: var(--fc-text); font-variant-numeric: tabular-nums; white-space: nowrap; }
.pd-vs { font-size: .72rem; font-weight: 700; color: var(--fc-faint); text-transform: uppercase; letter-spacing: .08em; }
.pd-markets { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin: 12px 0; }
.pd-market { background: var(--fc-surface-2); border-radius: var(--fc-radius-sm); padding: 7px 9px; min-width: 0; }
.pd-market span { display: block; font-size: .72rem; font-weight: 600; color: var(--fc-muted); white-space: nowrap; }
.pd-market b { display: block; font-size: .98rem; color: var(--fc-text); font-variant-numeric: tabular-nums; }
.pd-mkt { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; font-size: .76rem; color: var(--fc-muted);
  font-variant-numeric: tabular-nums; margin: -2px 0 12px; }
.pd-mkt b { color: var(--fc-text); font-weight: 700; }
.pd-mkt .pd-label { margin-right: 2px; }
.pd-mkt .pd-pill.val { margin-left: auto; }
.pd-warn { font-size: .74rem; color: var(--fc-yellow-ink); margin: -4px 0 10px; }
.pd-info { font-size: .74rem; color: var(--fc-muted); margin: -4px 0 10px; }

/* -- Análisis (dentro de la fila o junto a la tarjeta) -- */
.pd-detail { font-size: .82rem; color: var(--fc-muted); line-height: 1.5; }
.pd-detail table { width: 100%; border-collapse: collapse; margin: 8px 0 10px; font-variant-numeric: tabular-nums; }
.pd-detail th { text-align: left; font-weight: 600; color: var(--fc-faint); font-size: .7rem; text-transform: uppercase;
  letter-spacing: .05em; padding: 4px 4px 4px 0; border-bottom: 1px solid var(--fc-border); }
.pd-detail td { padding: 5px 4px 5px 0; border-bottom: 1px solid var(--fc-border); color: var(--fc-text); }
.pd-detail th:not(:first-child), .pd-detail td:not(:first-child) { text-align: right; }
.pd-ts { padding: 6px 0 8px; border-bottom: 1px solid var(--fc-border); margin-bottom: 8px; }
.pd-ts-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.pd-ts-head b { color: var(--fc-text); font-weight: 700; }
.pd-ts-meta { display: flex; flex-wrap: wrap; gap: 2px 12px; margin-top: 4px; font-size: .78rem; font-variant-numeric: tabular-nums; }
.pd-ts-meta span { white-space: nowrap; }
.pd-form { display: inline-flex; gap: 3px; }
.pd-form i { font-style: normal; font-size: .66rem; font-weight: 800; width: 17px; height: 17px; border-radius: 4px;
  display: inline-flex; align-items: center; justify-content: center; }
.pd-form .G { background: var(--fc-green-soft); color: var(--fc-green-ink); }
.pd-form .E { background: var(--fc-surface-2); color: var(--fc-muted); }
.pd-form .P { background: var(--fc-red-soft); color: var(--fc-red-ink); }
.pd-detail a { color: var(--fc-blue); font-weight: 600; text-decoration: none; }

/* -- Móvil: cada fila en dos líneas (hora · equipos · pronóstico / barra · over) y destacados deslizables -- */
@media (max-width: 760px) {
  .pd-cols { display: none; }
  .pd-row > summary { grid-template-columns: 48px minmax(0, 1fr) 96px; grid-template-areas:
      "time teams pick" "probs probs over"; gap: 8px 10px; padding: 10px 12px; }
  .pd-r-time { grid-area: time; } .pd-r-teams { grid-area: teams; } .pd-r-pick { grid-area: pick; }
  .pd-r-probs { grid-area: probs; } .pd-r-over { grid-area: over; align-self: end; font-size: .78rem; }
  .pd-r-over::before { content: "+2.5 "; color: var(--fc-faint); font-size: .7rem; }
  .pd-chev { display: none; }
  .pd-r-body { padding: 4px 12px 14px; }
  .pd-hl { grid-auto-flow: column; grid-template-columns: none; grid-auto-columns: 78%; overflow-x: auto;
    scroll-snap-type: x mandatory; padding-bottom: 6px; }
  .pd-hl-card { scroll-snap-align: start; }
  .pd-section .pd-note { display: none; }
  .st-key-pd_query { flex: 1 1 150px !important; width: auto !important; min-width: 0; }
}
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Datos (con caché)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=DAY_TTL_S, show_spinner=False)
def _day_listing(codes: tuple[str, ...], start: pd.Timestamp, end: pd.Timestamp
                 ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Partidos de cada competición en [start, end) según ESPN, descargados en paralelo."""
    def job(code: str):
        try:
            return code, comps.day_fixtures(comps.BY_CODE[code], start, end, requests.Session()), None
        except mm.PredictionError as exc:
            return code, None, str(exc)

    with ThreadPoolExecutor(comps.MAX_WORKERS) as pool:
        results = list(pool.map(job, codes))
    return ({c: f for c, f, _ in results if f is not None and len(f)},
            {c: e for c, _, e in results if e})


@st.cache_data(ttl=DAY_TTL_S, show_spinner=False)
def _next_kickoff(codes: tuple[str, ...], after: pd.Timestamp) -> pd.Timestamp | None:
    def job(code: str):
        try:
            return comps.next_kickoff(comps.BY_CODE[code], after, requests.Session())
        except mm.PredictionError:
            return None

    with ThreadPoolExecutor(comps.MAX_WORKERS) as pool:
        times = [t for t in pool.map(job, codes) if t is not None]
    return min(times) if times else None


@st.cache_resource(ttl=MODEL_TTL_S, show_spinner=False, max_entries=40)
def _competition_model(code: str) -> mm.LeagueModel:
    """Descarga los datos de la competición y entrena Poisson + logística (lo caro: una vez cada pocas horas)."""
    return mm.train_league(comps.load_competition(comps.BY_CODE[code]))


@st.cache_resource(ttl=MODEL_TTL_S, show_spinner=False, max_entries=40)
def _backtest(code: str) -> backtest.BacktestResult:
    return backtest.run(_competition_model(code))


@st.cache_data(ttl=LINEUP_TTL_S, show_spinner=False)
def _lineups(code: str, event_id: str, home_id: str, away_id: str, kickoff: pd.Timestamp) -> lineups.LineupInfo | None:
    """Alineaciones del partido (None si ESPN aún no las publica o la consulta falla)."""
    fixture = pd.Series({"datetime": kickoff, "espn_id": f"espn:{event_id}", "espn_home_id": home_id,
                         "espn_away_id": away_id})
    return lineups.fixture_lineups(code, comps.BY_CODE[code].is_cup, fixture)


def _row_lineups(code: str, row: pd.Series) -> lineups.LineupInfo | None:
    refs = lineups.espn_refs(row)
    if refs is None or comps.BY_CODE[code].is_cup or not lineups.in_window(row["datetime"]):
        return None
    return _lineups(code, *refs, row["datetime"])


def _model_or_error(code: str) -> mm.LeagueModel | None:
    name = comps.BY_CODE[code].name
    with st.spinner(f"Preparando el modelo de {name} (la primera vez tarda unos segundos)…"):
        try:
            return _competition_model(code)
        except mm.PredictionError as exc:
            st.warning(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001 — una competición caída no debe tumbar la página
            st.warning(f"{name}: no se pudo preparar el modelo ({exc}).")
    return None


def _browser_timezone() -> str | None:
    """Zona horaria del navegador, si Python la conoce (Chrome usa nombres antiguos como
    "America/Buenos_Aires", que sin el paquete tzdata no existen en algunos sistemas)."""
    try:
        tz = st.context.timezone
        ZoneInfo(tz)
        return tz
    except (AttributeError, TypeError, ValueError, ZoneInfoNotFoundError):
        return None


def _day_bounds_utc(day: date, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(day).tz_localize(tz).tz_convert("UTC").tz_localize(None)
    end = pd.Timestamp(day + timedelta(days=1)).tz_localize(tz).tz_convert("UTC").tz_localize(None)
    return start, end


def _local(ts: pd.Timestamp, tz: str) -> pd.Timestamp:
    return ts.tz_localize("UTC").tz_convert(tz)


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------


def _pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def _esc(text: str) -> str:
    return html.escape(str(text))


def _form_html(rows: pd.DataFrame) -> str:
    letters = mm.form_string(rows).split()
    return "<span class='pd-form'>" + "".join(f"<i class='{c}'>{c}</i>" for c in letters) + "</span>"


def _pick_pill(pred: mm.MatchPrediction) -> str:
    """Pronóstico (resultado más probable) y, si el partido ya se jugó, si acertó."""
    best = int(pred.p_final.argmax())
    pick = ("1 · " + pred.home, "X · Empate", "2 · " + pred.away)[best]
    if pred.final_score is None:
        if pred.p_final[best] < 0.45:
            return "<span class='pd-pill' title='Ningún resultado supera el 45%'>Parejo</span>"
        return f"<span class='pd-pill pick' title='Resultado más probable'>{_esc(pick)}</span>"
    hg, ag = pred.final_score
    actual = 0 if hg > ag else (1 if hg == ag else 2)
    if actual == best:
        return f"<span class='pd-pill ok' title='Pronóstico: {_esc(pick)}'>✓ Acertó</span>"
    return f"<span class='pd-pill miss' title='Pronóstico: {_esc(pick)}'>✗ Falló</span>"


def _bar_html(pred: mm.MatchPrediction) -> str:
    names = (pred.home, "Empate", pred.away)
    colors = ("var(--pd-home)", "var(--pd-draw)", "var(--pd-away)")
    label = ", ".join(f"{n} {p * 100:.1f}%" for n, p in zip(names, pred.p_final))
    segments = "".join(
        f"<i style='flex:{p:.4f};background:{c}' title='{_esc(n)}: {p * 100:.1f}%'></i>"
        for n, p, c in zip(names, pred.p_final, colors)
    )
    legend = "".join(
        f"<span><i class='pd-sw' style='background:{c}'></i>{k} <b>{_pct(p)}</b></span>"
        for k, p, c in zip(("1", "X", "2"), pred.p_final, colors)
    )
    return (f"<div class='pd-bar' role='img' aria-label='Probabilidades 1X2: {_esc(label)}'>{segments}</div>"
            f"<div class='pd-legend'>{legend}</div>")


def _market_html(pred: mm.MatchPrediction) -> str:
    """Probabilidades del mercado y, si el modelo da VALUE_THRESHOLD o más a algún resultado, cuál."""
    if not pred.market:
        return ""
    pm = pred.market["p_1x2"]
    diff = pred.p_final - pm
    best = int(diff.argmax())
    chip = ""
    if diff[best] >= VALUE_THRESHOLD:
        key = ("1", "X", "2")[best]
        chip = (f"<span class='pd-pill val' title='El modelo da {diff[best] * 100:.1f} puntos más que el mercado "
                f"a este resultado. No es una recomendación de apuesta.'>{key} +{diff[best] * 100:.0f} vs mercado</span>")
    probs = "".join(f"<span>{k} <b>{_pct(p)}</b></span>" for k, p in zip(("1", "X", "2"), pm))
    return (f"<div class='pd-mkt' title='Cuotas de DraftKings vía ESPN, sin el margen de la casa "
            f"({pred.market['margin'] * 100:.1f}%)'><span class='pd-label'>Mercado</span>{probs}{chip}</div>")


def _rotation_text(team) -> str:
    if team.rotation is None:
        return "sin historial suficiente"
    k = len(team.missing)
    return "once habitual" if k == 0 else f"{k} habitual{'es' if k > 1 else ''} fuera"


def _lineup_html(pred: mm.MatchPrediction) -> str:
    """Aviso de alineaciones confirmadas: rotación de cada equipo y cuánto movió el 1X2."""
    lu = pred.lineups
    if lu is None:
        return ""
    delta = (pred.p_final - pred.p_before_lineups) * 100
    moved = "sin cambio"
    if abs(delta[0]) >= 0.5 or abs(delta[2]) >= 0.5:
        moved = f"local {delta[0]:+.0f} pts · visitante {delta[2]:+.0f} pts"
    title = (f"{_esc(pred.home)}: {_esc(', '.join(lu.home.missing) or 'once habitual')} — "
             f"{_esc(pred.away)}: {_esc(', '.join(lu.away.missing) or 'once habitual')}")
    return (f"<div class='pd-info' title='Habituales que no son titulares. {title}'>👥 XI confirmados: "
            f"{_esc(pred.home)} {_rotation_text(lu.home)}, {_esc(pred.away)} {_rotation_text(lu.away)} "
            f"→ {moved}</div>")


def _card_summary_html(pred: mm.MatchPrediction, tz: str, show_date: bool = False) -> str:
    if pred.kickoff is None:
        when = f"Datos al {_local(pred.cutoff, tz):%d/%m}"
    else:
        when = _local(pred.kickoff, tz).strftime("%d/%m %H:%M" if show_date else "%H:%M")
    when = f"{when} · {comps.BY_CODE[pred.competition].name}" + (" · neutral" if pred.neutral else "")
    middle = (f"<div class='pd-score'>{pred.final_score[0]} - {pred.final_score[1]}</div>"
              if pred.final_score else "<div class='pd-vs'>vs</div>")
    i, j, p = pred.top_scores[0]
    warn = ""
    if pred.low_data:
        few = min(pred.home_snap["n_window"], pred.away_snap["n_window"])
        warn = (f"<div class='pd-warn' title='Partidos de los últimos 12 meses en los datos descargados'>"
                f"⚠ Pocos datos de algún equipo ({few} partidos): tómalo con cautela.</div>")
    if pred.newcomers:
        warn += (f"<div class='pd-info' title='Menos de {mm.NEWCOMER_MATCHES} partidos en la competición en 12 meses: "
                 f"su fuerza sale sobre todo de su liga anterior, con la calibración de ascendidos'>"
                 f"↑ Recién llegado: {_esc(', '.join(pred.newcomers))}</div>")
    return (
        f"<div class='pd-top'><span class='pd-time'>{_esc(when)}</span>{_pick_pill(pred)}</div>"
        "<div class='pd-teams'>"
        f"<div class='pd-team'><div class='pd-team-name'>{_esc(pred.home)}</div>"
        f"<div class='pd-team-xg' title='Goles esperados por el modelo'>λ {pred.lam_home:.2f}</div></div>"
        f"{middle}"
        f"<div class='pd-team away'><div class='pd-team-name'>{_esc(pred.away)}</div>"
        f"<div class='pd-team-xg' title='Goles esperados por el modelo'>λ {pred.lam_away:.2f}</div></div>"
        "</div>"
        f"{_bar_html(pred)}"
        "<div class='pd-markets'>"
        f"<div class='pd-market'><span>Over 2.5</span><b>{_pct(pred.over25)}</b></div>"
        f"<div class='pd-market'><span>Ambos anotan</span><b>{_pct(pred.btts)}</b></div>"
        f"<div class='pd-market' title='Marcador exacto más probable ({p * 100:.1f}%)'>"
        f"<span>Marcador</span><b>{i}-{j}</b></div>"
        "</div>"
        f"{_market_html(pred)}{_lineup_html(pred)}{warn}"
    )


def _detail_html(pred: mm.MatchPrediction) -> str:
    t, h, a = pred.trained, pred.home_snap, pred.away_snap
    att_h, def_h = pred.attack_defense("home")
    att_a, def_a = pred.attack_defense("away")
    signal = pred.signal_short

    def row(name: str, probs) -> str:
        return f"<tr><td>{name}</td>" + "".join(f"<td>{p * 100:.1f}%</td>" for p in probs) + "</tr>"

    def team_block(name: str, snap: dict, att: float, dfn: float, venue: str) -> str:
        recent = snap["recent_rows"]
        odds = recent[recent["mkt_real"]] if "mkt_real" in recent else recent.iloc[0:0]
        market = (f"<span title='Goles esperados a favor / en contra según las cuotas de cierre de "
                  f"{len(odds)} de sus últimos {len(recent)} partidos'>Mercado {odds['mf'].mean():.2f} / "
                  f"{odds['ma'].mean():.2f}</span>" if len(odds) else "")
        return (f"<div class='pd-ts'><div class='pd-ts-head'><b>{_esc(name)}</b>{_form_html(snap['form_rows'])}</div>"
                f"<div class='pd-ts-meta'><span title='Puntos en los últimos 5 partidos {venue}'>"
                f"{int(snap['venue_rows']['pts'].sum())} pts {venue}</span>"
                f"<span title='{signal} a favor / en contra, media de los últimos {len(recent)} partidos'>"
                f"{signal} {recent['sf'].mean():.2f} / {recent['sa'].mean():.2f}</span>{market}"
                f"<span title='Ataque / defensa ajustados por rival (1.00 = media)'>"
                f"Atq {att:.2f} · Def {dfn:.2f}</span>"
                f"<span title='Días desde su último partido en los datos descargados'>"
                f"{snap['rest_days']:.0f} d de descanso</span>"
                "</div></div>")

    market_row = row("Mercado", pred.market["p_1x2"]) if pred.market else ""
    scores = " · ".join(f"{i}-{j} ({p * 100:.1f}%)" for i, j, p in pred.top_scores)
    over_market = (f" · mercado {pred.market['over25'] * 100:.1f}%"
                   if pred.market and pred.market["over25"] is not None else "")
    url = comps.match_url(pred.match_id)
    link = f"<a href='{url}' target='_blank' rel='noopener'>Ver partido ↗</a>" if url else ""
    ensemble = pred.p_before_lineups if pred.lineups is not None else pred.p_final
    lineup_row = row("Con alineaciones", pred.p_final) if pred.lineups is not None else ""
    lineup_note = ""
    if pred.lineups is not None:
        parts = [f"{_esc(name)}: {_esc(', '.join(team.missing)) if team.missing else _rotation_text(team)}"
                 for name, team in ((pred.home, pred.lineups.home), (pred.away, pred.lineups.away))]
        lineup_note = f"<div><b>Habituales que no son titulares:</b> {' · '.join(parts)}</div>"
    return (
        "<div class='pd-detail'>"
        "<table><tr><th>Modelo</th><th>1</th><th>X</th><th>2</th></tr>"
        f"{row('Poisson (DC)', pred.p_poisson)}{row('Logística', pred.p_logistic)}"
        f"{row(f'Ensemble ({t.w_poisson:.0%}/{1 - t.w_poisson:.0%})', ensemble)}{lineup_row}{market_row}</table>"
        f"{team_block(pred.home, h, att_h, def_h, 'en casa')}"
        f"{team_block(pred.away, a, att_a, def_a, 'fuera')}"
        f"{lineup_note}<div><b>Marcadores más probables:</b> {scores}</div>"
        f"<div><b>Over 2.5:</b> modelo {pred.over25 * 100:.1f}% (Poisson {pred.over25_poisson * 100:.1f}%)"
        f"{over_market} · <b>Ambos anotan:</b> {pred.btts * 100:.1f}%</div>"
        f"<div style='margin-top:6px'>{link}</div>"
        "</div>"
    )


def _match_card(pred: mm.MatchPrediction, tz: str, key: str, show_date: bool = False,
                with_detail: bool = True) -> None:
    with st.container(key=f"pd_card_{key}"):
        st.markdown(_card_summary_html(pred, tz, show_date), unsafe_allow_html=True)
        if with_detail:
            with st.expander("Ver análisis"):
                st.markdown(_detail_html(pred), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Panel del día: escudos, destacados y tabla
# ---------------------------------------------------------------------------


@dataclass
class DayItem:
    """Un partido del día con su predicción (o el motivo por el que no la hay)."""
    key: str
    code: str
    row: pd.Series
    pred: mm.MatchPrediction | None
    error: str | None
    home_id: str | None
    away_id: str | None


def _initials(name: str) -> str:
    words = [w for w in re.split(r"[\s\-.()]+", str(name)) if w and w[0].isalnum()]
    return ("".join(w[0] for w in words[:2]) or str(name)[:2]).upper()


def _crest(team_id: str | None, name: str) -> str:
    """Escudo de ESPN; si no hay id o la imagen no carga, las iniciales del equipo."""
    initials = _esc(_initials(name))
    if team_id:
        return f"<img class='pd-crest' src='{ESPN_LOGO.format(team_id)}' alt='' data-i='{initials}' loading='lazy'>"
    return f"<span class='pd-crest'>{initials}</span>"


def _team_ids(row: pd.Series) -> tuple[str | None, str | None]:
    def text(key: str) -> str | None:
        value = row.get(key)
        return value if isinstance(value, str) and value else None

    return text("espn_home_id") or text("home_id"), text("espn_away_id") or text("away_id")


def _bar(p) -> str:
    names = ("Local", "Empate", "Visitante")
    colors = ("var(--pd-home)", "var(--pd-draw)", "var(--pd-away)")
    return ("<div class='pd-bar' role='img' aria-label='1X2: " + ", ".join(f"{n} {x * 100:.0f}%" for n, x in zip(names, p))
            + "'>" + "".join(f"<i style='flex:{x:.4f};background:{c}' title='{n}: {x * 100:.1f}%'></i>"
                             for n, x, c in zip(names, p, colors)) + "</div>")


def _value_edge(pred: mm.MatchPrediction) -> tuple[int, float] | None:
    """(resultado, puntos) donde el modelo más supera al mercado, si hay cuotas."""
    if not pred.market:
        return None
    diff = pred.p_final - pred.market["p_1x2"]
    best = int(diff.argmax())
    return best, float(diff[best])


def _flags_html(pred: mm.MatchPrediction) -> str:
    out = ""
    edge = _value_edge(pred)
    if edge and edge[1] >= VALUE_THRESHOLD:
        out += (f"<span class='pd-pill val' title='El modelo da {edge[1] * 100:.1f} puntos más que el mercado a este "
                f"resultado. No es una recomendación de apuesta.'>+{edge[1] * 100:.0f} al {('1', 'X', '2')[edge[0]]}</span>")
    if pred.lineups is not None:
        out += "<span class='pd-pill' title='Alineaciones confirmadas: el 1X2 tiene en cuenta las rotaciones'>XI</span>"
    return out


def _outcome_label(pred: mm.MatchPrediction, k: int) -> str:
    return (f"gana {pred.home}", "empate", f"gana {pred.away}")[k]


def _status(item: DayItem, tz: str, now: pd.Timestamp) -> tuple[str, str]:
    """(texto principal, clase) de la columna de hora: hora local, "En juego" o "Final"."""
    kickoff = item.row["datetime"]
    if item.pred is not None and item.pred.final_score is not None:
        return "Final", ""
    if kickoff <= now <= kickoff + pd.Timedelta(hours=2, minutes=15):
        return "En juego", "live"
    return f"{_local(kickoff, tz):%H:%M}", ""


def _row_html(item: DayItem, tz: str, now: pd.Timestamp, show_league: bool) -> str:
    row, pred = item.row, item.pred
    league = comps.BY_CODE[item.code].name
    status, cls = _status(item, tz, now)
    main = "<span class='pd-pill live'>En juego</span>" if cls == "live" else _esc(status)
    sub = _esc(league) if show_league else (f"{_local(row['datetime'], tz):%H:%M}" if status == "Final" else "")
    time_html = f"<div class='pd-r-time'><b>{main}</b><span title='{_esc(league)}'>{sub}</span></div>"
    score = pred.final_score if pred is not None else None
    teams = []
    for k, (name, team_id) in enumerate(((row["home"], item.home_id), (row["away"], item.away_id))):
        goals = f"<em>{score[k]}</em>" if score else ""
        win = " win" if score and score[k] > score[1 - k] else ""
        teams.append(f"<div class='pd-r-team{win}'>{_crest(team_id, name)}<span>{_esc(name)}</span>{goals}</div>")
    teams_html = f"<div class='pd-r-teams'>{''.join(teams)}</div>"
    if pred is None:
        return (f"<details class='pd-row' id='m-{item.key}'><summary>{time_html}{teams_html}"
                f"<div class='pd-r-probs pd-note'>{_esc(item.error)}</div><div></div><div></div><div></div></summary>"
                "</details>")
    best = int(pred.p_final.argmax())
    nums = "".join(f"<b>{x * 100:.0f}</b>" if k == best else f"<span>{x * 100:.0f}</span>"
                   for k, x in enumerate(pred.p_final))
    body = (f"{_market_html(pred)}{_lineup_html(pred)}{_detail_html(pred)}")
    return (
        f"<details class='pd-row' id='m-{item.key}'><summary>"
        f"{time_html}{teams_html}"
        f"<div class='pd-r-probs'><div class='pd-r-nums'>{nums}</div>{_bar(pred.p_final)}</div>"
        f"<div class='pd-r-over' title='Probabilidad de más de 2.5 goles'>{_pct(pred.over25)}</div>"
        f"<div class='pd-r-pick'>{_pick_pill(pred)}{_flags_html(pred)}</div>"
        "<div class='pd-chev' aria-hidden='true'>›</div>"
        f"</summary><div class='pd-r-body'>{body}</div></details>"
    )


def _table_html(items: list[DayItem], tz: str, now: pd.Timestamp, sort: str) -> str:
    head = ("<div class='pd-cols'><span>Hora</span><span>Partido</span>"
            "<span class='pd-c-probs'><span>1</span><span>X</span><span>2</span></span>"
            "<span style='text-align:right'>+2.5</span><span style='text-align:right'>Pronóstico</span><span></span></div>")
    rows = []
    if sort == "Liga":
        last = None
        for it in sorted(items, key=lambda it: (ORDERED_CODES.index(it.code), it.row["datetime"])):
            if it.code != last:
                rows.append(f"<div class='pd-group'>{_esc(comps.BY_CODE[it.code].name)}</div>")
                last = it.code
            rows.append(_row_html(it, tz, now, show_league=False))
    else:
        if sort == "Más claros":
            items = sorted(items, key=lambda it: -(it.pred.p_final.max() if it.pred is not None else 0))
        rows = [_row_html(it, tz, now, show_league=True) for it in items]
    return f"<div class='pd-table'>{head}{''.join(rows)}</div>"


def _highlight_items(items: list[DayItem], kind: str) -> list[tuple[DayItem, str, str, str]]:
    """Hasta N_HIGHLIGHTS partidos destacados: (partido, cifra, texto, pie)."""
    preds = [it for it in items if it.pred is not None]
    pending = [it for it in preds if it.pred.final_score is None] or preds
    out = []
    if kind == "Más claros":
        for it in sorted(pending, key=lambda it: -it.pred.p_final.max())[:N_HIGHLIGHTS]:
            k = int(it.pred.p_final.argmax())
            out.append((it, _pct(it.pred.p_final[k]), _outcome_label(it.pred, k), ""))
    elif kind == "Más parejos":
        for it in sorted(pending, key=lambda it: it.pred.p_final.max())[:N_HIGHLIGHTS]:
            p = it.pred.p_final
            out.append((it, _pct(p.max()), "como máximo",
                        " · ".join(f"{s} {_pct(x)}" for s, x in zip(("1", "X", "2"), p))))
    elif kind == "Vs mercado":
        edges = [(it, _value_edge(it.pred)) for it in pending if it.pred.market]
        for it, (k, d) in sorted(edges, key=lambda x: -x[1][1])[:N_HIGHLIGHTS]:
            pm = it.pred.market["p_1x2"][k]
            out.append((it, f"+{d * 100:.0f}", f"pts al {('1', 'X', '2')[k]} ({_outcome_label(it.pred, k)})",
                        f"Modelo {_pct(it.pred.p_final[k])} · mercado {_pct(pm)}"))
    elif kind == "XI confirmados":
        with_xi = [it for it in preds if it.pred.lineups is not None]
        for it in sorted(with_xi, key=lambda it: -abs(it.pred.lineups.shift))[:N_HIGHLIGHTS]:
            lu, p0, p1 = it.pred.lineups, it.pred.p_before_lineups, it.pred.p_final
            k = int(abs(p1 - p0).argmax())
            out.append((it, f"{(p1[k] - p0[k]) * 100:+.0f}", f"pts al {('1', 'X', '2')[k]} por las alineaciones",
                        f"{it.pred.home}: {_rotation_text(lu.home)} · {it.pred.away}: {_rotation_text(lu.away)}"))
    return out


def _highlights_html(entries: list[tuple[DayItem, str, str, str]], tz: str) -> str:
    cards = []
    for it, main, text, foot in entries:
        when = _local(it.row["datetime"], tz)
        teams = "".join(f"<div class='pd-hl-team'>{_crest(tid, name)}<span>{_esc(name)}</span></div>"
                        for name, tid in ((it.row["home"], it.home_id), (it.row["away"], it.away_id)))
        foot_html = f"<div class='pd-hl-foot'>{_esc(foot)}</div>" if foot else ""
        cards.append(
            f"<a class='pd-hl-card' href='#m-{it.key}' title='Ver el partido en la tabla'>"
            f"<div class='pd-hl-top'>{when:%H:%M} · {_esc(comps.BY_CODE[it.code].name)}</div>"
            f"<div class='pd-hl-teams'>{teams}</div>"
            f"<div class='pd-hl-main'><b>{_esc(main)}</b><span>{_esc(text)}</span></div>"
            f"{_bar(it.pred.p_final)}{foot_html}</a>")
    return f"<div class='pd-hl'>{''.join(cards)}</div>"


def _matches_query(row: pd.Series, query: str) -> bool:
    """Cada palabra buscada empieza alguna palabra del local o del visitante: "man u" encuentra
    "Manchester United" y "real" no encuentra "Montréal"."""
    tokens = mm.normalize(query).split()
    for side in ("home", "away"):
        words = mm.normalize(str(row[side])).split()
        if all(any(w.startswith(t) for w in words) for t in tokens):
            return True
    return not tokens


def _long_date(day: date) -> str:
    return f"{WEEKDAYS[day.weekday()].capitalize()} {day.day} de {MONTHS[day.month - 1]}"


def _safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text))


def _recent_table(snap: dict, signal: str) -> pd.DataFrame:
    rows = snap["recent_rows"].iloc[::-1]
    return pd.DataFrame(
        {
            "Fecha": rows["datetime"].dt.strftime("%Y-%m-%d"),
            "Sede": rows["venue"].map({"h": "Local", "a": "Visitante"}),
            "Rival": rows["opp"],
            "Resultado": [f"{int(g)}-{int(c)}" for g, c in zip(rows["gf"], rows["ga"])],
            f"{signal} F": rows["sf"].round(2),
            f"{signal} C": rows["sa"].round(2),
        }
    )


def _validation_table(models: dict[str, mm.LeagueModel]) -> pd.DataFrame:
    rows = []
    for code, m in models.items():
        t = m.trained
        rows.append({
            "Competición": comps.BY_CODE[code].name,
            "Señal": m.data.signal_name,
            "Peso señal": f"{t.signal_weight:.0%}",
            "Peso mercado": f"{t.market_weight:.0%}",
            "ρ (Dixon-Coles)": round(t.rho, 3),
            "Log loss referencia": round(t.ll_baseline, 3),
            "Log loss modelo": round(t.ll_ensemble, 3),
            "Mejora": round(t.ll_baseline - t.ll_ensemble, 3),
            "Partidos": t.n_val,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------


def _date_from_url() -> date | None:
    """Fecha inicial desde `?fecha=AAAA-MM-DD` (enlaces a un día concreto)."""
    try:
        return pd.Timestamp(st.query_params["fecha"]).date()
    except (KeyError, ValueError):
        return None


def _jump_to(day: date) -> None:
    st.session_state["pd_date"] = day


def _shift_day(delta: int) -> None:
    st.session_state["pd_date"] = st.session_state["pd_date"] + timedelta(days=delta)


def _section_title(title: str, note: str = "") -> None:
    note_html = f"<span class='pd-note' style='margin:0'>{_esc(note)}</span>" if note else ""
    st.markdown(f"<div class='pd-section'><h3>{_esc(title)}</h3>{note_html}</div>", unsafe_allow_html=True)


def section_today(tz: str) -> None:
    if "pd_date" not in st.session_state:  # los botones ‹ › y "próxima fecha" la cambian vía session_state
        st.session_state["pd_date"] = _date_from_url() or pd.Timestamp.now(tz=tz).date()
    for region in comps.REGIONS:  # selección por defecto (vía session_state, sin `default` en el widget)
        st.session_state.setdefault(f"pd_codes_{region}", [c for c in comps.DEFAULT_CODES
                                                          if comps.BY_CODE[c].region == region])
    chosen = sum(len(st.session_state[f"pd_codes_{r}"]) for r in comps.REGIONS)
    with st.container(key="pd_controls", horizontal=True, vertical_alignment="center", gap="small", wrap=True):
        st.button("", icon=":material/chevron_left:", key="pd_prev", on_click=_shift_day, args=(-1,),
                  help="Día anterior")
        day = st.date_input("Fecha", key="pd_date", format="DD/MM/YYYY", label_visibility="collapsed", width=140)
        st.button("", icon=":material/chevron_right:", key="pd_next", on_click=_shift_day, args=(1,),
                  help="Día siguiente")
        query = st.text_input("Buscar equipo", key="pd_query", placeholder="Buscar equipo", icon=":material/search:",
                              label_visibility="collapsed", width=230)
        with st.popover(f"Ligas y copas · {chosen}", icon=":material/tune:", key="pd_leagues"):
            for region in comps.REGIONS:
                st.pills(region, [c.code for c in comps.COMPETITIONS if c.region == region],
                         format_func=lambda c: comps.BY_CODE[c].name, selection_mode="multi",
                         key=f"pd_codes_{region}")
    codes = [c for r in comps.REGIONS for c in st.session_state[f"pd_codes_{r}"]]
    if not codes:
        st.markdown("<div class='pd-empty'>Elige al menos una liga o copa.</div>", unsafe_allow_html=True)
        return

    start, end = _day_bounds_utc(day, tz)
    selected = tuple(c for c in ORDERED_CODES if c in codes)
    with st.spinner("Buscando los partidos del día…"):
        fixtures, errors = _day_listing(selected, start, end)
    for code, err in errors.items():
        st.warning(f"{comps.BY_CODE[code].name}: {err}")

    if not fixtures:
        st.markdown(f"<div class='pd-empty'>No hay partidos el {_long_date(day).lower()} en las competiciones "
                    "elegidas.</div>", unsafe_allow_html=True)
        nxt = _next_kickoff(selected, end)
        if nxt is not None:
            nxt_day = _local(nxt, tz).date()
            st.button(f"Ir a la próxima fecha con partidos: {_long_date(nxt_day).lower()}", on_click=_jump_to,
                      args=(nxt_day,), icon=":material/event:")
        return

    models: dict[str, mm.LeagueModel] = {}
    progress = st.progress(0.0, text="Preparando modelos…") if len(fixtures) > 1 else None
    for n, code in enumerate(fixtures, start=1):
        if progress:
            progress.progress(n / len(fixtures), text=f"Modelo de {comps.BY_CODE[code].name} ({n}/{len(fixtures)})…")
        model = _model_or_error(code)
        if model is not None:
            models[code] = model
    if progress:
        progress.empty()
    if not models:
        return

    day_rows = []
    for code, model in models.items():
        rows = comps.fixture_rows_for_model(comps.BY_CODE[code], model.data, fixtures[code])
        day_rows.append(rows.assign(_code=code))
    day_matches = pd.concat(day_rows, ignore_index=True)
    day_matches["_order"] = day_matches["_code"].map(ORDERED_CODES.index)
    day_matches = day_matches.sort_values(["datetime", "_order", "home"])

    items = []
    for fixture in day_matches.to_dict("records"):
        row = pd.Series(fixture)
        code = row["_code"]
        pred, error = None, None
        try:
            pred = mm.predict_match(models[code], row["home"], row["away"], fixture=row,
                                    lineups=_row_lineups(code, row))
        except mm.PredictionError as exc:
            error = str(exc)
        items.append(DayItem(_safe_key(row["id"]), code, row, pred, error, *_team_ids(row)))
    now = mm.utc_now()
    shown = [it for it in items if _matches_query(it.row, query)]

    n_leagues = len({it.code for it in items})
    st.markdown(f"<div class='pd-summary'><b>{_long_date(day)}</b> · {len(items)} partido"
                f"{'' if len(items) == 1 else 's'} en {n_leagues} {'liga' if n_leagues == 1 else 'ligas y copas'} · "
                f"hora de {_esc(tz.split('/')[-1].replace('_', ' '))}</div>", unsafe_allow_html=True)

    if query:
        if not shown:
            st.markdown(f"<div class='pd-empty'>Ningún partido de este día coincide con «{_esc(query)}».</div>",
                        unsafe_allow_html=True)
            return
    else:
        kinds = [k for k in HIGHLIGHTS
                 if (k != "Vs mercado" or any(it.pred is not None and it.pred.market for it in items))
                 and (k != "XI confirmados" or any(it.pred is not None and it.pred.lineups for it in items))]
        _section_title("Destacados")
        kind = st.segmented_control("Destacados", kinds, default=kinds[0], key="pd_hl", required=True,
                                    label_visibility="collapsed")
        entries = _highlight_items(items, kind or kinds[0])
        if entries:
            st.markdown(_highlights_html(entries, tz), unsafe_allow_html=True)

    _section_title(f"Partidos ({len(shown)})" if query else "Todos los partidos",
                   "Toca un partido para ver el análisis")
    sort = st.segmented_control("Ordenar por", SORTS, default=SORTS[0], key="pd_sort", required=True,
                                label_visibility="collapsed")
    st.markdown(_table_html(shown, tz, now, sort or SORTS[0]), unsafe_allow_html=True)

    with st.expander("Cómo funciona y limitaciones"):
        st.markdown(
            "- **Datos:** xG partido a partido de [Understat](https://understat.com) en Premier League, LaLiga, "
            "Bundesliga, Serie A, Ligue 1 y la liga rusa. En el resto, resultados y tiros de ESPN con un **xG "
            "aproximado** (0.23 por tiro a puerta + 0.065 por tiro fuera, calibrado con el xG de Understat); en "
            "Uruguay y Paraguay, sin tiros publicados, solo goles.\n"
            "- **Copas internacionales:** se modelan junto con las ligas de sus participantes, para que la fuerza "
            "de equipos de países distintos sea comparable.\n"
            "- **Poisson (Dixon-Coles):** fuerza de ataque/defensa ajustada por rival (12 meses ponderados por "
            "antigüedad, mezcla de señal y goles con pesos elegidos con datos) → goles esperados λ, con el total "
            "acercado a la media de la competición → matriz de marcadores con la corrección de Dixon-Coles.\n"
            "- **Señal de mercado:** donde hay cuotas de cierre de partidos anteriores (football-data.co.uk en "
            "21 ligas y sus divisiones inferiores; ESPN, desde finales de 2025, en copas y ligas sudamericanas), "
            "se despejan los goles esperados que implican y entran como tercera señal de fuerza: recogen lo que "
            "el mercado sabía (fichajes, lesiones, alineaciones). Nunca se usan las cuotas del propio partido.\n"
            "- **Alineaciones:** desde ~1 h antes del inicio, cuando ESPN publica los titulares, se mide cuánto "
            "rota cada equipo (sus 11 habituales de los 10 partidos anteriores que no salen de titulares, "
            "pesados por sus titularidades) y se ajusta el 1X2; en validación cruzada sobre 5.871 partidos de "
            "17 ligas mejora el log loss en 0.002. Solo en ligas.\n"
            "- **Recién ascendidos:** su historial de la división inferior (ESPN) entra en el cálculo de fuerzas, "
            "con una calibración de su nivel estimada con ascensos anteriores.\n"
            "- **Regresión logística:** reajusta la señal de Poisson con la racha de puntos, la localía y el "
            "descanso. **Ensemble:** el peso de cada modelo minimiza el log loss en el 30% más reciente.\n"
            "- **Mercado del partido:** probabilidades de las cuotas de DraftKings (vía ESPN) sin el margen de la "
            "casa. La etiqueta amarilla marca el resultado al que el modelo da 10 o más puntos más que el mercado; "
            "**no es una recomendación de apuesta**: el mercado suele ser más preciso que cualquier modelo público.\n"
            "- **No incluye** lesiones ni sanciones hasta que se conocen los titulares; el descanso solo cuenta "
            "los partidos de las competiciones descargadas.\n"
            "- En partidos ya jugados, la regresión logística se entrenó con toda la temporada: el "
            "\"acertó/falló\" es orientativo, no un backtest.\n"
            "- Cada modelo se reentrena cada 3 horas con los resultados nuevos."
        )
        st.markdown("**Validación de los modelos de hoy** (log loss 1X2 en el 30% más reciente del histórico; "
                    "menor es mejor; la referencia es predecir siempre las frecuencias de 1/X/2):")
        st.dataframe(_validation_table(models), hide_index=True, width="stretch")


def section_manual(tz: str) -> None:
    with st.container(key="pd_manual_form"):
        code = st.selectbox("Liga o copa", ORDERED_CODES, key="pd_m_code",
                            format_func=lambda c: f"{comps.BY_CODE[c].name} · {comps.BY_CODE[c].region}")
        model = _model_or_error(code)
        if model is None:
            return
        teams = model.data.teams
        c1, c2 = st.columns(2)
        with c1:
            home = st.selectbox("Local", teams, index=0, key=f"pd_m_home_{code}")
        with c2:
            away = st.selectbox("Visitante", teams, index=min(1, len(teams) - 1), key=f"pd_m_away_{code}")
        run = st.button("Correr modelo", type="primary", icon=":material/play_arrow:", width="stretch",
                        disabled=home == away)
        if home == away:
            st.caption("Elige dos equipos distintos.")
    if run:
        st.session_state["pd_manual"] = (code, home, away)

    request = st.session_state.get("pd_manual")
    if not request or request[0] != code:
        st.markdown(f"<div class='pd-note'>Si el partido está en el calendario de los próximos "
                    f"{mm.FIXTURE_HORIZON_DAYS} días se usa su fecha (y sus cuotas); si no, todos los datos "
                    "disponibles hasta hoy.</div>", unsafe_allow_html=True)
        return

    code, home, away = request
    comp = comps.BY_CODE[code]
    try:
        pred = comps.predict_fixture(comp, model, home, away)
    except mm.PredictionError as exc:
        st.warning(str(exc))
        return

    left, right = st.columns([1, 1.2])
    with left:
        _match_card(pred, tz, key="manual", show_date=True, with_detail=False)
    with right:
        st.markdown(_detail_html(pred), unsafe_allow_html=True)
    signal = pred.signal_short
    c1, c2 = st.columns(2)
    for col, name, snap in ((c1, pred.home, pred.home_snap), (c2, pred.away, pred.away_snap)):
        with col:
            st.markdown(f"<div class='pd-label'>{_esc(name)} · últimos {snap['n_recent']} partidos</div>",
                        unsafe_allow_html=True)
            st.dataframe(_recent_table(snap, signal), hide_index=True, width="stretch")


def _backtest_row(code: str, r: backtest.BacktestResult) -> dict:
    return {
        "Competición": comps.BY_CODE[code].name,
        "Partidos": r.n_matched,
        "Modelo": round(r.ll_model, 4),
        "Mercado": round(r.ll_market, 4),
        "Distancia": round(r.gap, 4),
        "Peso del modelo en la mezcla": f"{r.alpha:.0%}",
        "Over/Under modelo": round(r.ll_ou_model, 4) if r.ll_ou_model is not None else None,
        "Over/Under mercado": round(r.ll_ou_market, 4) if r.ll_ou_market is not None else None,
    }


def section_performance() -> None:
    st.markdown(
        "<div class='pd-note'>Cada modelo se valida con el 30% más reciente de su histórico (partidos que la "
        "regresión logística no vio al entrenar) y se compara con las <b>cuotas de cierre</b> de esos mismos "
        "partidos (football-data.co.uk; en copas y ligas que no cubre, las que ESPN guarda desde finales de "
        "2025): la referencia más exigente, porque recogen toda la información del mercado justo antes del "
        "inicio. Log loss: menor es mejor.</div>", unsafe_allow_html=True)
    with st.container(key="pd_perf_form"):
        code = st.selectbox("Liga o copa", ORDERED_CODES, key="pd_perf_code",
                            format_func=lambda c: f"{comps.BY_CODE[c].name} · "
                                                  f"{'cuotas históricas' if fd.has_odds(c) else 'cuotas recientes'}")
    model = _model_or_error(code)
    if model is None:
        return
    t = model.trained
    c1, c2, c3 = st.columns(3)
    c1.metric("Log loss del modelo", f"{t.ll_ensemble:.4f}", help=f"{t.n_val} partidos de validación ({t.val_period})")
    c2.metric("Referencia (frecuencias 1/X/2)", f"{t.ll_baseline:.4f}")
    c3.metric("Mejora sobre la referencia", f"{t.ll_baseline - t.ll_ensemble:+.4f}")
    market = (f"mercado {t.market_weight:.0%} ({model.data.market_coverage:.0%} de partidos con cuotas en 12 "
              f"meses) · " if t.market_weight else "sin señal de mercado · ")
    st.caption(f"Parámetros elegidos con datos: {market}señal {model.data.signal_name} {t.signal_weight:.0%} / goles "
               f"{1 - t.signal_weight:.0%} · Dixon-Coles ρ {t.rho:+.3f} · total de goles {t.total_shrink:.0%} propio "
               f"/ {1 - t.total_shrink:.0%} media · recién llegados ×{t.newcomer[0]:.2f} goles, ×{t.newcomer[1]:.2f} "
               f"rival · ensemble {t.w_poisson:.0%} Poisson / {1 - t.w_poisson:.0%} logística.")

    with st.spinner("Comparando con las cuotas de cierre…"):
        try:
            r = _backtest(code)
        except mm.PredictionError as exc:
            st.warning(str(exc))
            r = None
    if r is not None:
        st.markdown(f"**Contra el mercado** · {r.n_matched} partidos ({r.period}) · cuotas: {r.odds_source}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Modelo", f"{r.ll_model:.4f}")
        m2.metric("Mercado (cierre)", f"{r.ll_market:.4f}")
        m3.metric("Distancia", f"{r.gap:+.4f}", help="Log loss modelo − mercado: negativo = mejor que el cierre")
        m4.metric("Peso en la mezcla", f"{r.alpha:.0%}",
                  help="Mezcla modelo + mercado con menor log loss. ~0% = el modelo no añade información "
                       f"al cierre (log loss de la mezcla, con validación cruzada: {r.ll_blend_cv:.4f}).")
        if r.ll_ou_model is not None:
            st.caption(f"Over/Under 2.5 ({r.n_ou} partidos): modelo {r.ll_ou_model:.4f} · mercado "
                       f"{r.ll_ou_market:.4f}.")
        st.markdown("ROI simulado apostando 1 unidad a la cuota de cierre cuando el modelo da al menos "
                    "el umbral de puntos más que el mercado (orientativo: muestras pequeñas, mucho ruido):")
        roi = r.roi.assign(ROI=r.roi["ROI"].map(lambda x: f"{x * 100:+.1f}%" if pd.notna(x) else "—"))
        st.dataframe(roi, hide_index=True, width="stretch")

    if st.button("Resumen de todas las competiciones", icon=":material/table_chart:"):
        rows, skipped = [], []
        bar = st.progress(0.0)
        for n, c in enumerate(ORDERED_CODES, start=1):
            bar.progress(n / len(ORDERED_CODES), text=f"{comps.BY_CODE[c].name} ({n}/{len(ORDERED_CODES)})…")
            try:
                rows.append(_backtest_row(c, _backtest(c)))
            except Exception as exc:  # noqa: BLE001 — una competición sin datos no corta el resumen
                skipped.append(f"{comps.BY_CODE[c].name} ({exc})")
        bar.empty()
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if skipped:
            st.caption("Sin backtest: " + "; ".join(skipped))


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

browser_tz = _browser_timezone()
tz_options = ([browser_tz] if browser_tz and browser_tz not in TIMEZONES else []) + TIMEZONES
default_tz = browser_tz if browser_tz else "UTC"

st.markdown(
    "<div class='pd-head'><div class='pd-title'>Partidos del día</div>"
    f"<div class='pd-sub'>Pronósticos del modelo para {len(comps.COMPETITIONS)} ligas y copas, comparados con el "
    "mercado</div></div>",
    unsafe_allow_html=True)

with st.sidebar:
    tz = st.selectbox("Zona horaria", tz_options, index=tz_options.index(default_tz), key="pd_tz",
                      help="Por defecto, la de tu navegador. Define qué partidos son \"del día\" y sus horas.")
view = st.segmented_control("Sección", ["Partidos", "Analizar", "Rendimiento"], default="Partidos", key="pd_view",
                            required=True, label_visibility="collapsed")

if view == "Analizar":
    section_manual(tz)
elif view == "Rendimiento":
    section_performance()
else:
    section_today(tz)
