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
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import requests
import streamlit as st

from src import backtest
from src import competitions as comps
from src import football_data_source as fd
from src import match_model as mm

MODEL_TTL_S = 3 * 3600  # igual que la caché del año/temporada en curso: el modelo ve los resultados nuevos
DAY_TTL_S = 10 * 60
# Diferencia modelo − mercado que se señala: la media es ~5 puntos por resultado; 10 o más
# aparece en ~1 de cada 10 partidos (medido en una jornada de 39 partidos con cuotas).
VALUE_THRESHOLD = 0.10
TIMEZONES = [
    "Europe/Madrid", "Europe/London", "America/Mexico_City", "America/Bogota", "America/Lima",
    "America/Santiago", "America/Argentina/Buenos_Aires", "America/New_York", "UTC",
]
ORDERED_CODES = [c.code for region in comps.REGIONS for c in comps.COMPETITIONS if c.region == region]

PAGE_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: var(--fc-bg); }
/* Colores de la barra 1X2: escala divergente local (azul) ↔ visitante (naranja) con empate neutro.
   Los dos polos pasan el validador de paleta (CVD ΔE 24.7, contraste ≥3:1); el gris es el punto medio. */
:root { --pd-home: #2a78d6; --pd-draw: #CBD5E1; --pd-away: #eb6834; }

.pd-head { padding-bottom: 18px; margin-bottom: 18px; border-bottom: 1px solid var(--fc-border); }
.pd-title { font-size: clamp(1.75rem, 3.2vw, 2.35rem); font-weight: 800; letter-spacing: -.03em; line-height: 1.05;
  color: var(--fc-text); text-transform: uppercase; }
.pd-title span { color: var(--fc-blue); }
.pd-sub { color: var(--fc-muted); font-size: .95rem; margin-top: 6px; }
.pd-label { font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--fc-faint); }
.pd-note { font-size: .8rem; color: var(--fc-muted); line-height: 1.5; margin-bottom: 1rem; }
.pd-empty { border: 1px dashed var(--fc-border-strong); border-radius: var(--fc-radius); padding: 18px 20px;
  color: var(--fc-muted); font-size: .9rem; line-height: 1.5; background: var(--fc-surface); margin-bottom: 1rem; }

.st-key-pd_grid { display: grid !important;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr)); gap: 16px; align-items: start; }
[class*="st-key-pd_card_"] { background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius);
  box-shadow: var(--fc-shadow); padding: 14px 16px 4px; gap: 4px; }
[class*="st-key-pd_card_"] [data-testid="stExpander"] details { border: 0; border-top: 1px solid var(--fc-border);
  border-radius: 0; background: transparent; }
[class*="st-key-pd_card_"] [data-testid="stExpander"] summary { padding-left: 2px; padding-right: 2px; }
[class*="st-key-pd_card_"] [data-testid="stExpander"] summary p { font-weight: 600; color: var(--fc-blue); }
[class*="st-key-pd_card_"] [data-testid="stExpanderDetails"] { padding: 2px 2px 12px; }
.st-key-pd_controls, .st-key-pd_manual_form { background: var(--fc-surface); border: 1px solid var(--fc-border);
  border-radius: var(--fc-radius); box-shadow: var(--fc-shadow); padding: 14px 16px 6px; }

.pd-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; }
.pd-time { font-size: .8rem; font-weight: 700; color: var(--fc-text); font-variant-numeric: tabular-nums; }
.pd-pill { display: inline-flex; align-items: center; gap: 4px; font-size: .66rem; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; padding: 3px 9px; border-radius: 999px; white-space: nowrap;
  background: var(--fc-surface-2); color: var(--fc-muted); border: 1px solid var(--fc-border); }
.pd-pill.ok { background: var(--fc-green-soft); color: var(--fc-green-ink); border-color: color-mix(in srgb, var(--fc-green) 25%, transparent); }
.pd-pill.miss { background: var(--fc-red-soft); color: var(--fc-red-ink); border-color: color-mix(in srgb, var(--fc-red) 25%, transparent); }
.pd-pill.pick { background: var(--fc-blue-soft); color: var(--fc-blue-ink); border-color: color-mix(in srgb, var(--fc-blue) 25%, transparent); }

.pd-teams { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); align-items: center; gap: 8px; }
.pd-team { min-width: 0; }
.pd-team.away { text-align: right; }
.pd-team-name { font-size: 1rem; font-weight: 700; color: var(--fc-text); line-height: 1.2; overflow-wrap: anywhere; }
.pd-team-xg { font-size: .74rem; color: var(--fc-muted); margin-top: 2px; font-variant-numeric: tabular-nums; }
.pd-score { font-size: 1.25rem; font-weight: 800; color: var(--fc-text); font-variant-numeric: tabular-nums; white-space: nowrap; }
.pd-vs { font-size: .72rem; font-weight: 700; color: var(--fc-faint); text-transform: uppercase; letter-spacing: .08em; }

/* Barra 1X2: segmentos separados por 2px de superficie, extremos redondeados a 4px. */
.pd-bar { display: flex; gap: 2px; height: 10px; margin: 12px 0 6px; }
.pd-bar i { display: block; height: 100%; min-width: 3px; }
.pd-bar i:first-child { border-radius: 4px 0 0 4px; }
.pd-bar i:last-child { border-radius: 0 4px 4px 0; }
.pd-legend { display: flex; justify-content: space-between; gap: 6px; font-size: .78rem; color: var(--fc-muted);
  font-variant-numeric: tabular-nums; }
.pd-legend span { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.pd-legend b { color: var(--fc-text); font-weight: 700; }
.pd-sw { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }

.pd-markets { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin: 12px 0 8px; }
.pd-markets { margin-bottom: 12px; }
.pd-market { background: var(--fc-surface-2); border-radius: var(--fc-radius-sm); padding: 7px 9px; min-width: 0; }
.pd-market span { display: block; font-size: .72rem; font-weight: 600; color: var(--fc-muted); white-space: nowrap; }
.pd-market b { display: block; font-size: .98rem; color: var(--fc-text); font-variant-numeric: tabular-nums; }

.pd-detail { font-size: .82rem; color: var(--fc-muted); line-height: 1.5; }
.pd-detail table { width: 100%; border-collapse: collapse; margin: 4px 0 10px; font-variant-numeric: tabular-nums; }
.pd-detail th { text-align: left; font-weight: 600; color: var(--fc-faint); font-size: .7rem; text-transform: uppercase;
  letter-spacing: .05em; padding: 4px 4px 4px 0; border-bottom: 1px solid var(--fc-border); }
.pd-detail td { padding: 5px 4px 5px 0; border-bottom: 1px solid var(--fc-border); color: var(--fc-text); }
.pd-detail th:not(:first-child), .pd-detail td:not(:first-child) { text-align: right; }
.pd-ts { padding: 6px 0 8px; border-bottom: 1px solid var(--fc-border); margin-bottom: 8px; }
.pd-ts-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.pd-ts-head b { color: var(--fc-text); font-weight: 700; }
.pd-ts-meta { display: flex; flex-wrap: wrap; gap: 2px 12px; margin-top: 4px; font-size: .78rem;
  font-variant-numeric: tabular-nums; }
.pd-ts-meta span { white-space: nowrap; }
.pd-form { display: inline-flex; gap: 3px; }
.pd-form i { font-style: normal; font-size: .66rem; font-weight: 800; width: 17px; height: 17px; border-radius: 4px;
  display: inline-flex; align-items: center; justify-content: center; }
.pd-form .G { background: var(--fc-green-soft); color: var(--fc-green-ink); }
.pd-form .E { background: var(--fc-surface-2); color: var(--fc-muted); }
.pd-form .P { background: var(--fc-red-soft); color: var(--fc-red-ink); }
.pd-detail a { color: var(--fc-blue); font-weight: 600; text-decoration: none; }

/* Mercado: probabilidades sin margen de las cuotas y la mayor diferencia con el modelo. */
.pd-mkt { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; font-size: .76rem; color: var(--fc-muted);
  font-variant-numeric: tabular-nums; margin: -2px 0 12px; }
.pd-mkt b { color: var(--fc-text); font-weight: 700; }
.pd-mkt .pd-label { margin-right: 2px; }
.pd-pill.val { background: var(--fc-yellow-soft); color: var(--fc-yellow-ink);
  border-color: color-mix(in srgb, var(--fc-yellow) 30%, transparent); margin-left: auto; }
.pd-warn { font-size: .74rem; color: var(--fc-yellow-ink); margin: -4px 0 10px; }
.pd-info { font-size: .74rem; color: var(--fc-muted); margin: -4px 0 10px; }
.st-key-pd_perf_form { background: var(--fc-surface); border: 1px solid var(--fc-border);
  border-radius: var(--fc-radius); box-shadow: var(--fc-shadow); padding: 14px 16px 6px; }
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
            return "<span class='pd-pill' title='Ningún resultado supera el 45%'>Abierto</span>"
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
        f"{_market_html(pred)}{warn}"
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
    return (
        "<div class='pd-detail'>"
        "<table><tr><th>Modelo</th><th>1</th><th>X</th><th>2</th></tr>"
        f"{row('Poisson (DC)', pred.p_poisson)}{row('Logística', pred.p_logistic)}"
        f"{row(f'Ensemble ({t.w_poisson:.0%}/{1 - t.w_poisson:.0%})', pred.p_final)}{market_row}</table>"
        f"{team_block(pred.home, h, att_h, def_h, 'en casa')}"
        f"{team_block(pred.away, a, att_a, def_a, 'fuera')}"
        f"<div><b>Marcadores más probables:</b> {scores}</div>"
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


def section_today(tz: str) -> None:
    if "pd_date" not in st.session_state:  # el botón "próxima fecha" la cambia vía session_state
        st.session_state["pd_date"] = _date_from_url() or pd.Timestamp.now(tz=tz).date()
    for region in comps.REGIONS:  # selección por defecto (vía session_state, sin `default` en el widget)
        st.session_state.setdefault(f"pd_codes_{region}", [c for c in comps.DEFAULT_CODES
                                                          if comps.BY_CODE[c].region == region])
    chosen = sum(len(st.session_state[f"pd_codes_{r}"]) for r in comps.REGIONS)
    with st.container(key="pd_controls"):
        c1, c2 = st.columns([1, 1.6], vertical_alignment="bottom")
        with c1:
            day = st.date_input("Fecha", key="pd_date", format="DD/MM/YYYY")
        with c2:
            with st.popover(f"Ligas y copas · {chosen} elegidas", icon=":material/tune:", width="stretch"):
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
        st.markdown(f"<div class='pd-empty'>No hay partidos el {day:%d/%m/%Y} en las competiciones elegidas.</div>",
                    unsafe_allow_html=True)
        nxt = _next_kickoff(selected, end)
        if nxt is not None:
            nxt_day = _local(nxt, tz).date()
            st.button(f"Ir a la próxima fecha con partidos: {nxt_day:%d/%m/%Y}", on_click=_jump_to,
                      args=(nxt_day,), icon=":material/event:")
        return

    total = sum(len(f) for f in fixtures.values())
    n_comp = f"{len(fixtures)} " + ("competición" if len(fixtures) == 1 else "competiciones")
    st.markdown(f"<div class='pd-note'>{total} partido{'' if total == 1 else 's'} en {n_comp} · horas en {_esc(tz)} · "
                "predicción con los datos anteriores a cada partido.</div>", unsafe_allow_html=True)

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

    with st.container(key="pd_grid"):
        for fixture in day_matches.to_dict("records"):
            row = pd.Series(fixture)
            code = row["_code"]
            try:
                pred = mm.predict_match(models[code], row["home"], row["away"], fixture=row)
            except mm.PredictionError as exc:
                with st.container(key=f"pd_card_{_safe_key(row['id'])}"):
                    st.markdown(f"<div class='pd-top'><span class='pd-time'>{_local(row['datetime'], tz):%H:%M}"
                                f" · {_esc(comps.BY_CODE[code].name)}</span></div>"
                                f"<b>{_esc(row['home'])} vs {_esc(row['away'])}</b>"
                                f"<div class='pd-note'>{_esc(exc)}</div>", unsafe_allow_html=True)
                continue
            _match_card(pred, tz, key=_safe_key(row["id"]))

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
            "- **Recién ascendidos:** su historial de la división inferior (ESPN) entra en el cálculo de fuerzas, "
            "con una calibración de su nivel estimada con ascensos anteriores.\n"
            "- **Regresión logística:** reajusta la señal de Poisson con la racha de puntos, la localía y el "
            "descanso. **Ensemble:** el peso de cada modelo minimiza el log loss en el 30% más reciente.\n"
            "- **Mercado del partido:** probabilidades de las cuotas de DraftKings (vía ESPN) sin el margen de la "
            "casa. La etiqueta amarilla marca el resultado al que el modelo da 10 o más puntos más que el mercado; "
            "**no es una recomendación de apuesta**: el mercado suele ser más preciso que cualquier modelo público.\n"
            "- **No incluye** las lesiones, sanciones ni alineaciones del propio partido (solo, de forma indirecta, "
            "las que ya reflejaban las cuotas de partidos anteriores); el descanso solo cuenta los partidos de las "
            "competiciones descargadas.\n"
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
        pred = mm.predict_match(model, home, away)
        if comp.understat and pred.match_id is not None:  # cuotas de ESPN para el partido de Understat
            fixture = comps.understat_fixture_with_odds(comp, model.data, pred.match_id)
            if fixture is not None:
                pred = mm.predict_match(model, pred.home, pred.away, fixture=fixture)
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
    "<div class='pd-head'><div class='pd-title'>Partidos <span>del día</span></div>"
    f"<div class='pd-sub'>Predicciones Poisson + regresión logística para {len(comps.COMPETITIONS)} ligas y copas: "
    "xG real (Understat) o xG aproximado con tiros (ESPN), comparadas con el mercado.</div></div>",
    unsafe_allow_html=True)

with st.sidebar:
    tz = st.selectbox("Zona horaria", tz_options, index=tz_options.index(default_tz), key="pd_tz",
                      help="Por defecto, la de tu navegador. Define qué partidos son \"del día\" y sus horas.")
view = st.segmented_control("Sección", ["Partidos del día", "Analizar un partido", "Rendimiento del modelo"],
                            default="Partidos del día", key="pd_view", label_visibility="collapsed")

if view == "Analizar un partido":
    section_manual(tz)
elif view == "Rendimiento del modelo":
    section_performance()
else:
    section_today(tz)
