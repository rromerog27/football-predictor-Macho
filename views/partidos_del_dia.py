"""Partidos del día — predicciones automáticas con xG de Understat.

Página registrada en `app.py`. Lista los partidos de la fecha elegida (en la
zona horaria del navegador) de las ligas que cubre Understat y muestra para
cada uno la predicción del ensemble Poisson + regresión logística de
`src/understat_model.py`. La sección "Analizar un partido" corre el mismo
modelo para cualquier cruce de una liga.

Cada liga se entrena una sola vez y queda en caché unas horas (la primera
visita tarda unos segundos por liga); predecir un partido es instantáneo.
Este archivo solo presenta los datos: el modelo vive en `src/understat_model.py`.
"""

from __future__ import annotations

import html
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

from src import understat_model as um

MODEL_TTL_S = um.CURRENT_SEASON_CACHE_TTL_S  # el modelo se reentrena cuando caduca la temporada en caché
FIXTURES_TTL_S = 30 * 60
TIMEZONES = [
    "Europe/Madrid", "Europe/London", "America/Mexico_City", "America/Bogota", "America/Lima",
    "America/Santiago", "America/Argentina/Buenos_Aires", "America/New_York", "UTC",
]
OUTCOME_NAMES = ("Local", "Empate", "Visitante")

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
</style>
"""

st.markdown(PAGE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Datos (con caché)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=FIXTURES_TTL_S, show_spinner=False)
def _season_matches(league: str) -> pd.DataFrame:
    """Calendario de la temporada en curso (una descarga ligera por liga)."""
    season = um.current_season_for(um.utc_now())
    dates, _ = um.fetch_season(requests.Session(), league, season, season)
    return um.matches_frame({season: dates})


@st.cache_resource(ttl=MODEL_TTL_S, show_spinner=False)
def _league_model(league: str) -> um.LeagueModel:
    """Descarga las temporadas de la liga y entrena Poisson + logística (lo caro: una vez cada pocas horas)."""
    return um.train_league(um.load_league(league))


def _model_or_error(league: str) -> um.LeagueModel | None:
    with st.spinner(f"Preparando el modelo de {um.league_name(league)} (la primera vez tarda unos segundos)…"):
        try:
            return _league_model(league)
        except um.PredictionError as exc:
            st.warning(f"{um.league_name(league)}: {exc}")
        except Exception as exc:  # noqa: BLE001 — una liga caída no debe tumbar la página
            st.warning(f"{um.league_name(league)}: no se pudo preparar el modelo ({exc}).")
    return None


def _browser_timezone() -> str | None:
    try:
        return st.context.timezone
    except AttributeError:
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
    letters = um.form_string(rows).split()
    return "<span class='pd-form'>" + "".join(f"<i class='{c}'>{c}</i>" for c in letters) + "</span>"


def _pick_pill(pred: um.MatchPrediction) -> str:
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


def _bar_html(pred: um.MatchPrediction) -> str:
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


def _card_summary_html(pred: um.MatchPrediction, tz: str, show_date: bool = False) -> str:
    if pred.kickoff is None:
        when = f"Datos al {_local(pred.cutoff, tz):%d/%m}"
    else:
        when = _local(pred.kickoff, tz).strftime("%d/%m %H:%M" if show_date else "%H:%M")
    when = f"{when} · {um.league_name(pred.league)}"
    middle = (f"<div class='pd-score'>{pred.final_score[0]} - {pred.final_score[1]}</div>"
              if pred.final_score else "<div class='pd-vs'>vs</div>")
    i, j, p = pred.top_scores[0]
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
    )


def _detail_html(pred: um.MatchPrediction) -> str:
    t, h, a = pred.trained, pred.home_snap, pred.away_snap
    att_h, def_h = pred.attack_defense("home")
    att_a, def_a = pred.attack_defense("away")

    def row(name: str, probs) -> str:
        return f"<tr><td>{name}</td>" + "".join(f"<td>{p * 100:.1f}%</td>" for p in probs) + "</tr>"

    def team_block(name: str, snap: dict, att: float, dfn: float, venue: str) -> str:
        recent = snap["recent_rows"]
        return (f"<div class='pd-ts'><div class='pd-ts-head'><b>{_esc(name)}</b>{_form_html(snap['form_rows'])}</div>"
                f"<div class='pd-ts-meta'><span title='Puntos en los últimos 5 partidos {venue}'>"
                f"{int(snap['venue_rows']['pts'].sum())} pts {venue}</span>"
                f"<span title='xG a favor / en contra, media de los últimos {len(recent)} partidos'>"
                f"xG {recent['xgf'].mean():.2f} / {recent['xga'].mean():.2f}</span>"
                f"<span title='Ataque / defensa ajustados por rival (1.00 = media de la liga)'>"
                f"Atq {att:.2f} · Def {dfn:.2f}</span>"
                f"<span title='Días desde su último partido de liga'>{snap['rest_days']:.0f} d de descanso</span>"
                "</div></div>")

    scores = " · ".join(f"{i}-{j} ({p * 100:.1f}%)" for i, j, p in pred.top_scores)
    link = (f"<a href='{pred.understat_url}' target='_blank' rel='noopener'>Ver en Understat ↗</a>"
            if pred.understat_url else "")
    return (
        "<div class='pd-detail'>"
        "<table><tr><th>Modelo</th><th>1</th><th>X</th><th>2</th></tr>"
        f"{row('Poisson', pred.p_poisson)}{row('Logística', pred.p_logistic)}"
        f"{row(f'Ensemble ({t.w_poisson:.0%}/{1 - t.w_poisson:.0%})', pred.p_final)}</table>"
        f"{team_block(pred.home, h, att_h, def_h, 'en casa')}"
        f"{team_block(pred.away, a, att_a, def_a, 'fuera')}"
        f"<div><b>Marcadores más probables:</b> {scores}</div>"
        f"<div><b>Poisson puro:</b> Over 2.5 {pred.over25_poisson * 100:.1f}% · "
        f"Ambos anotan {pred.btts_poisson * 100:.1f}%</div>"
        f"<div style='margin-top:6px'>{link}</div>"
        "</div>"
    )


def _match_card(pred: um.MatchPrediction, tz: str, key: str, show_date: bool = False,
                with_detail: bool = True) -> None:
    with st.container(key=f"pd_card_{key}"):
        st.markdown(_card_summary_html(pred, tz, show_date), unsafe_allow_html=True)
        if with_detail:
            with st.expander("Ver análisis"):
                st.markdown(_detail_html(pred), unsafe_allow_html=True)


def _recent_table(snap: dict) -> pd.DataFrame:
    rows = snap["recent_rows"].iloc[::-1]
    return pd.DataFrame(
        {
            "Fecha": rows["datetime"].dt.strftime("%Y-%m-%d"),
            "Sede": rows["venue"].map({"h": "Local", "a": "Visitante"}),
            "Rival": rows["opp"],
            "Resultado": [f"{int(g)}-{int(c)}" for g, c in zip(rows["gf"], rows["ga"])],
            "xG F": rows["xgf"].round(2),
            "xG C": rows["xga"].round(2),
        }
    )


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------


def _jump_to(day: date) -> None:
    st.session_state["pd_date"] = day


def section_today(tz: str) -> None:
    if "pd_date" not in st.session_state:  # el botón "próxima fecha" la cambia vía session_state
        st.session_state["pd_date"] = pd.Timestamp.now(tz=tz).date()
    with st.container(key="pd_controls"):
        day = st.date_input("Fecha", key="pd_date", format="DD/MM/YYYY", width=220)
        leagues = st.pills("Ligas", options=list(um.LEAGUES), format_func=um.league_name,
                           selection_mode="multi", default=list(um.LEAGUES), key="pd_leagues")
    if not leagues:
        st.markdown("<div class='pd-empty'>Elige al menos una liga.</div>", unsafe_allow_html=True)
        return

    start, end = _day_bounds_utc(day, tz)
    fixtures: dict[str, pd.DataFrame] = {}
    for league in leagues:
        try:
            matches = _season_matches(league)
        except um.PredictionError as exc:
            st.warning(f"{um.league_name(league)}: {exc}")
            continue
        day_matches = matches[(matches["datetime"] >= start) & (matches["datetime"] < end)]
        if len(day_matches):
            fixtures[league] = day_matches

    if not fixtures:
        upcoming = []
        for league in leagues:
            try:
                m = _season_matches(league)
            except um.PredictionError:
                continue
            nxt = m[m["datetime"] >= end]
            if len(nxt):
                upcoming.append(_local(nxt["datetime"].iloc[0], tz).date())
        st.markdown(f"<div class='pd-empty'>No hay partidos el {day:%d/%m/%Y} en las ligas elegidas.</div>",
                    unsafe_allow_html=True)
        if upcoming:
            nxt_day = min(upcoming)
            st.button(f"Ir a la próxima fecha con partidos: {nxt_day:%d/%m/%Y}", on_click=_jump_to,
                      args=(nxt_day,), icon=":material/event:")
        return

    models = {league: m for league in fixtures if (m := _model_or_error(league)) is not None}
    if not models:
        return
    day_matches = pd.concat([fixtures[league].assign(league=league) for league in models])
    day_matches = day_matches.sort_values(["datetime", "league", "home"])
    st.markdown(f"<div class='pd-note'>{len(day_matches)} partidos · horas en {_esc(tz)} · predicción con los "
                "datos anteriores a cada partido.</div>", unsafe_allow_html=True)

    with st.container(key="pd_grid"):
        for fixture in day_matches.to_dict("records"):
            row = pd.Series(fixture)
            try:
                pred = um.predict_match(models[row["league"]], row["home"], row["away"], fixture=row)
            except um.PredictionError as exc:
                with st.container(key=f"pd_card_{row['id']}"):
                    st.markdown(f"<div class='pd-top'><span class='pd-time'>{_local(row['datetime'], tz):%H:%M}"
                                f" · {_esc(um.league_name(row['league']))}</span></div>"
                                f"<b>{_esc(row['home'])} vs {_esc(row['away'])}</b>"
                                f"<div class='pd-note'>{_esc(exc)}</div>", unsafe_allow_html=True)
                continue
            _match_card(pred, tz, key=str(row["id"]))

    with st.expander("Cómo funciona y limitaciones"):
        st.markdown(
            "- **Datos:** xG partido a partido de [Understat](https://understat.com) (temporada en curso y 5 "
            "anteriores). Solo cubre Premier League, LaLiga, Bundesliga, Serie A, Ligue 1 y la liga rusa.\n"
            "- **Poisson:** fuerza de ataque/defensa ajustada por rival (xG de 12 meses, mezclado con los "
            "últimos 10 partidos) → goles esperados λ → matriz de marcadores.\n"
            "- **Regresión logística:** reajusta la señal de Poisson con la racha de puntos, la localía y el "
            "descanso.\n"
            "- **Ensemble:** el peso de cada modelo minimiza el log loss en las dos últimas temporadas.\n"
            "- **No incluye** lesiones, sanciones ni alineaciones; el descanso solo cuenta partidos de liga.\n"
            "- En partidos ya jugados, el pronóstico usa los datos previos al partido, pero la regresión "
            "logística se entrenó con toda la temporada: tómalo como referencia, no como un backtest.\n"
            "- El modelo de cada liga se reentrena cada 3 horas con los resultados nuevos."
        )
        st.markdown("**Validación de cada liga** (log loss 1X2 en las dos últimas temporadas; menor es mejor, "
                    "1.099 equivale a repartir 33/33/33):")
        st.dataframe(
            pd.DataFrame(
                {
                    "Liga": [um.league_name(league) for league in models],
                    "Peso Poisson": [f"{m.trained.w_poisson:.0%}" for m in models.values()],
                    "Log loss Poisson": [round(m.trained.ll_poisson, 3) for m in models.values()],
                    "Log loss logística": [round(m.trained.ll_logistic, 3) for m in models.values()],
                    "Log loss ensemble": [round(m.trained.ll_ensemble, 3) for m in models.values()],
                    "Partidos": [m.trained.n_val for m in models.values()],
                }
            ),
            hide_index=True, width="stretch",
        )


def section_manual(tz: str) -> None:
    with st.container(key="pd_manual_form"):
        league = st.selectbox("Liga", options=list(um.LEAGUES), format_func=um.league_name, key="pd_m_league")
        try:
            matches = _season_matches(league)
        except um.PredictionError as exc:
            st.warning(str(exc))
            return
        teams = sorted(set(matches["home"]) | set(matches["away"]))
        c1, c2 = st.columns(2)
        with c1:
            home = st.selectbox("Local", teams, index=0, key=f"pd_m_home_{league}")
        with c2:
            away = st.selectbox("Visitante", teams, index=min(1, len(teams) - 1), key=f"pd_m_away_{league}")
        run = st.button("Correr modelo", type="primary", icon=":material/play_arrow:", width="stretch",
                        disabled=home == away)
        if home == away:
            st.caption("Elige dos equipos distintos.")
    if run:
        st.session_state["pd_manual"] = (league, home, away)

    request = st.session_state.get("pd_manual")
    if not request:
        st.markdown("<div class='pd-note'>Si el partido está en el calendario se usa su fecha; si no, todos los "
                    "datos disponibles hasta hoy.</div>", unsafe_allow_html=True)
        return

    league, home, away = request
    model = _model_or_error(league)
    if model is None:
        return
    try:
        pred = um.predict_match(model, home, away)
    except um.PredictionError as exc:
        st.warning(str(exc))
        return

    left, right = st.columns([1, 1.2])
    with left:
        _match_card(pred, tz, key="manual", show_date=True, with_detail=False)
    with right:
        st.markdown(_detail_html(pred), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, name, snap in ((c1, pred.home, pred.home_snap), (c2, pred.away, pred.away_snap)):
        with col:
            st.markdown(f"<div class='pd-label'>{_esc(name)} · últimos {snap['n_recent']} partidos de liga</div>",
                        unsafe_allow_html=True)
            st.dataframe(_recent_table(snap), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

browser_tz = _browser_timezone()
tz_options = ([browser_tz] if browser_tz and browser_tz not in TIMEZONES else []) + TIMEZONES
default_tz = browser_tz if browser_tz else "UTC"

st.markdown(
    "<div class='pd-head'><div class='pd-title'>Partidos <span>del día</span></div>"
    "<div class='pd-sub'>Predicciones Poisson + regresión logística con el xG real de Understat.</div></div>",
    unsafe_allow_html=True)

with st.sidebar:
    tz = st.selectbox("Zona horaria", tz_options, index=tz_options.index(default_tz), key="pd_tz",
                      help="Por defecto, la de tu navegador. Define qué partidos son \"del día\" y sus horas.")
view = st.segmented_control("Sección", ["Partidos del día", "Analizar un partido"], default="Partidos del día",
                            key="pd_view", label_visibility="collapsed")

if view == "Analizar un partido":
    section_manual(tz)
else:
    section_today(tz)
