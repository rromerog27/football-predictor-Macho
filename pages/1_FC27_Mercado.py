"""FC 27 Mercado — señales de trading para EA SPORTS FC 27 Ultimate Team.

Página independiente del predictor de partidos. Al abrirla descarga datos en
vivo de FUT.GG (con caché de unos minutos), guarda una instantánea en el
historial local y calcula Market Score, Risk Score, señales y alertas.

Si FUT.GG no responde, muestra la última instantánea guardada y avisa de su
antigüedad.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import fc27_history, fc27_market, fc27_signals, ui_theme

CACHE_TTL_S = 570          # algo menos que el auto-refresco, para que cada refresco traiga datos nuevos
AUTO_REFRESH_S = 600
MIN_CALIBRATION = 20       # señales evaluadas necesarias para mostrar una tasa de acierto
FUTGG = fc27_market.BASE_URL

st.set_page_config(page_title="FC 27 Mercado", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown(ui_theme.inject_global_css(), unsafe_allow_html=True)
st.markdown(
    f"""
    <style>
    .fc-head {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-end; gap:8px 24px; margin-bottom:6px; }}
    .fc-title {{ font-size:2rem; font-weight:800; color:{ui_theme.INK_900}; line-height:1.1; }}
    .fc-sub {{ color:{ui_theme.INK_600}; font-size:.92rem; }}
    .fc-stamp {{ color:{ui_theme.INK_600}; font-size:.85rem; text-align:right; }}
    .fc-pill {{ display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
               padding:4px 9px; border-radius:999px; }}
    .fc-buy {{ background:#DCFCE7; color:#15803D; }}
    .fc-watch {{ background:#FEF3C7; color:#92400E; }}
    .fc-risk {{ background:#FEE2E2; color:#B91C1C; }}
    .fc-chip {{ display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
               padding:2px 7px; border-radius:5px; background:{ui_theme.GRAY_100}; color:{ui_theme.INK_600}; margin-left:6px; }}
    .fc-up {{ color:#1D4ED8; font-weight:600; }}
    .fc-down {{ color:#B91C1C; font-weight:600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def _fetch_live() -> fc27_market.MarketSnapshot:
    return fc27_market.fetch_market_snapshot()


@st.cache_resource
def _db():
    return fc27_history.connect()


def _fmt(n) -> str:
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    return f"{int(n):,}".replace(",", ".")


def _pct_html(p) -> str:
    if p is None or pd.isna(p):
        return "<span style='color:#94A3B8'>—</span>"
    cls = "fc-up" if p > 0 else ("fc-down" if p < 0 else "")
    return f"<span class='{cls}'>{p:+.1f}%</span>".replace(".", ",")


def _age_text(ts: datetime) -> str:
    mins = int((datetime.now(timezone.utc) - ts).total_seconds() // 60)
    if mins < 1:
        return "hace menos de 1 min"
    if mins < 60:
        return f"hace {mins} min"
    return f"hace {mins // 60} h {mins % 60} min"


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------

PILL = {"COMPRAR": ("fc-buy", "▲ Comprar"), "VIGILAR": ("fc-watch", "● Vigilar"), "RIESGO": ("fc-risk", "▼ Riesgo de caída")}


def _calibration_text(evaluated: pd.DataFrame, kind: str) -> str:
    sub = evaluated[evaluated["kind"] == kind] if not evaluated.empty else evaluated
    n = len(sub)
    if n < MIN_CALIBRATION:
        return f"Sin calibrar todavía: {n}/{MIN_CALIBRATION} señales evaluadas."
    rate = sub["hit"].mean() * 100
    return f"Acierto histórico de esta señal: {rate:.0f}% ({n} evaluadas).".replace(".", ",", 1)


def _signal_card(row: pd.Series) -> None:
    with st.container(border=True):
        is_fodder = str(row["key"]).startswith("fodder")
        title = html.escape(str(row["name"]) if is_fodder else f"{row['name']} {row['overall']}")
        st.markdown(
            f"**{title}**  \n<span style='color:#475569;font-size:.85rem'>{html.escape(str(row['rarity']))} · "
            f"{html.escape(row['label'])} · Horizonte {row['horizon']}</span>",
            unsafe_allow_html=True,
        )
        changes = " · ".join(
            f"{lbl} {_pct_html(row[col])}" for col, lbl in
            [("pct_1h", "1h"), ("pct_6h", "6h"), ("pct_24h", "24h"), ("pct_168h", "7d")]
        )
        st.markdown(f"<span style='font-size:1.15rem;font-weight:700'>{_fmt(row['price'])}</span> monedas  \n{changes}",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.progress(int(row["market_score"]) / 100, text=f"Market Score {int(row['market_score'])}")
        c2.progress(int(row["risk_score"]) / 100, text=f"Risk Score {int(row['risk_score'])}")
        plan = fc27_signals.trade_plan(row)
        st.markdown(
            f"- **Precio interesante:** {plan['zona']}\n- **Objetivo:** {plan['objetivo']}\n"
            f"- **Invalidación:** {plan['invalidacion']}"
        )
        if row["reasons"]:
            st.markdown("**Por qué:** " + " ".join(row["reasons"]))
        if row["risks"]:
            st.markdown("**Riesgos:** " + " ".join(row["risks"]))
        with st.expander("Desglose del Market Score"):
            bd = pd.DataFrame(
                [(label, int(row[col]), mx) for col, label, mx in fc27_signals.SCORE_COMPONENTS],
                columns=["Componente", "Puntos", "Máximo"],
            )
            st.dataframe(bd, hide_index=True, width="stretch")
            st.caption("News vale 5 (neutro) porque las noticias no se leen automáticamente.")
        if row.get("url"):
            st.markdown(f"[Ver en FUT.GG]({FUTGG}{row['url']})")


def render_signals(signals: pd.DataFrame, evaluated: pd.DataFrame, analyst: dict, now: datetime) -> None:
    st.caption(
        "Las señales salen de reglas fijas (ver Metodología). El Market Score es un indicador comparativo, "
        "no una probabilidad. La tasa de acierto real aparece cuando haya suficientes señales evaluadas en tu historial."
    )
    cols = st.columns(3)
    for col, kind in zip(cols, ["COMPRAR", "VIGILAR", "RIESGO"]):
        top = fc27_signals.top_by_signal(signals, kind)
        css, label = PILL[kind]
        with col:
            st.markdown(f"<span class='fc-pill {css}'>{label}</span> <span style='color:#94A3B8'>{len(top)}</span>",
                        unsafe_allow_html=True)
            if kind != "VIGILAR":
                st.caption(_calibration_text(evaluated, kind))
            if top.empty:
                st.info("Ninguna carta cumple las condiciones ahora mismo.")
            for _, row in top.iterrows():
                _signal_card(row)

    notes = fc27_signals.active_notes(analyst, now)
    if notes:
        st.subheader("Notas del analista")
        st.caption(f"Tesis manuales de data/fc27_analyst.json (actualizado {analyst.get('updated', '—')}). Desaparecen al caducar.")
        for n in notes:
            css, label = PILL.get(n.get("signal", ""), ("fc-watch", n.get("signal", "")))
            with st.container(border=True):
                st.markdown(f"<span class='fc-pill {css}'>{label}</span> **{html.escape(n['player'])}**",
                            unsafe_allow_html=True)
                st.markdown(f"{n['thesis']}  \n**Invalidación:** {n['invalidation']}")


def render_fodder(snap: fc27_market.MarketSnapshot) -> None:
    fodder = fc27_market.fodder_table(snap.cheapest)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Coste del fodder por punto de Item Score")
        if fodder.empty:
            st.info("No hay datos de las cartas más baratas por rating.")
        else:
            colors = ["#B8860B" if b else "#93C5FD" for b in fodder["is_best"]]
            fig = go.Figure(go.Bar(
                x=fodder["overall"].astype(str), y=fodder["coins_per_point"], marker_color=colors,
                customdata=fodder[["name", "price", "item_score"]],
                hovertemplate="Rating %{x} · más barato: %{customdata[0]}<br>Referencia %{customdata[1]:,} monedas · %{customdata[2]:,} pts"
                              "<br>%{y:.2f} monedas por punto<extra></extra>",
            ))
            fig.update_layout(template="plotly_white", height=320, margin=dict(t=10, b=40, l=40, r=10),
                              xaxis_title="Rating", yaxis_title="Monedas por punto", separators=",.")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                fodder.rename(columns={"overall": "Rating", "name": "Más barato", "min_price": "Mínimo",
                                       "price": "Referencia (mediana 5)",
                                       "item_score": "Item Score", "coins_per_point": "Monedas/punto",
                                       "score_verified": "Puntos verificados", "is_best": "Mejor"}),
                hide_index=True, width="stretch",
            )
            floor = fc27_market.fodder_floor_price(snap.cheapest)
            st.caption(f"Precio mínimo observado en 81-84: {_fmt(floor)} (probable suelo de EA). "
                       "Item Score de 83 y 89 tomado de tablas de terceros.")
    with right:
        st.subheader("SBC activos")
        if snap.sbcs.empty:
            st.info("No hay datos de SBC.")
        else:
            sb = snap.sbcs.copy()
            sb["link"] = FUTGG + sb["url"].fillna("")
            sb["termina"] = pd.to_datetime(sb["end_time"], utc=True, errors="coerce")
            sb = sb[sb["termina"] < pd.Timestamp("2030-01-01", tz="UTC")]  # fuera los permanentes
            sb = sb.sort_values("cost", ascending=False)
            for col in ("score_requirement", "cost", "cost_pc", "award_overall"):
                sb[col] = pd.to_numeric(sb[col], errors="coerce")
            st.dataframe(
                sb[["name", "link", "category", "score_requirement", "cost", "cost_pc", "termina", "award_name", "award_overall"]],
                hide_index=True, width="stretch",
                column_config={
                    "name": "SBC", "link": st.column_config.LinkColumn("Enlace", display_text="Abrir"),
                    "category": "Tipo", "score_requirement": st.column_config.NumberColumn("Puntos", format="localized"),
                    "cost": st.column_config.NumberColumn("Coste consola", format="localized"),
                    "cost_pc": st.column_config.NumberColumn("Coste PC", format="localized"),
                    "termina": st.column_config.DatetimeColumn("Termina (UTC)", format="ddd D MMM HH:mm"),
                    "award_name": "Premio", "award_overall": st.column_config.NumberColumn("Media", format="%d"),
                },
            )
            st.caption("Coste estimado por FUT.GG con la ruta más barata. No incluye los SBC permanentes.")


BUDGETS = {
    "Todos": (0, float("inf")), "Bajo (<20k)": (0, 20_000), "Medio (20k-100k)": (20_000, 100_000),
    "Alto (100k-500k)": (100_000, 500_000), "Premium (500k+)": (500_000, float("inf")),
}


def render_movers(signals: pd.DataFrame) -> None:
    cards = signals[~signals["key"].str.startswith("fodder")].copy()
    if cards.empty:
        st.info("No hay datos de momentum.")
        return
    f1, f2, f3, f4 = st.columns([2, 2, 1.4, 1.6])
    rarities = sorted(cards["rarity"].dropna().unique())
    sel_rar = f1.multiselect("Rareza", rarities, default=[], placeholder="Todas", key="mv_rar")
    budget = f2.selectbox("Presupuesto", list(BUDGETS), key="mv_budget")
    direction = f3.selectbox("Dirección", ["Todas", "Suben", "Bajan"], key="mv_dir")
    query = f4.text_input("Buscar jugador", key="mv_q")
    lo, hi = BUDGETS[budget]
    view = cards[(cards["price"] >= lo) & (cards["price"] < hi)]
    if sel_rar:
        view = view[view["rarity"].isin(sel_rar)]
    if direction != "Todas":
        view = view[view["pct_24h"] > 0] if direction == "Suben" else view[view["pct_24h"] < 0]
    if query:
        view = view[view["name"].str.contains(query, case=False, na=False)]
    view = view.assign(link=FUTGG + view["url"].fillna(""), signal=view["signal"].fillna("—"))
    for col in ("pct_1h", "pct_6h", "pct_24h", "pct_72h", "pct_168h"):
        view[col] = pd.to_numeric(view[col], errors="coerce")  # None -> celda vacía
    history_cols = [c for c in ("pct_1h", "pct_6h", "pct_72h", "pct_168h") if view[c].notna().any()]
    missing = {"pct_1h": "1h", "pct_6h": "6h", "pct_72h": "3d", "pct_168h": "7d"}
    hidden = [lbl for c, lbl in missing.items() if c not in history_cols]
    note = f" Columnas {', '.join(hidden)} ocultas hasta que tu historial local las cubra." if hidden else ""
    st.caption(f"{len(view)} de {len(cards)} cartas.{note}")
    ordered = ["pct_1h", "pct_6h", "pct_24h", "pct_72h", "pct_168h"]
    pct_cols = [c for c in ordered if c == "pct_24h" or c in history_cols]
    st.dataframe(
        view[["name", "overall", "rarity", "price", *pct_cols,
              "market_score", "risk_score", "signal", "label", "link"]],
        hide_index=True, width="stretch", height=560,
        column_config={
            "name": "Carta", "link": st.column_config.LinkColumn("FUT.GG", display_text="Abrir"),
            "overall": st.column_config.NumberColumn("OVR", format="%d"),
            "rarity": "Rareza",
            "price": st.column_config.NumberColumn("Precio", format="localized"),
            "pct_1h": st.column_config.NumberColumn("1h %", format="%+.1f"),
            "pct_6h": st.column_config.NumberColumn("6h %", format="%+.1f"),
            "pct_24h": st.column_config.NumberColumn("24h %", format="%+.1f"),
            "pct_72h": st.column_config.NumberColumn("3d %", format="%+.1f"),
            "pct_168h": st.column_config.NumberColumn("7d %", format="%+.1f"),
            "market_score": st.column_config.ProgressColumn("Market Score", min_value=0, max_value=100, format="%d"),
            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%d"),
            "signal": "Señal", "label": "Clasificación",
        },
    )


LEVEL_ICON = {"crítica": "🚨", "aviso": "⚠️", "info": "ℹ️"}


def render_alerts(alerts: list[fc27_signals.Alert], analyst: dict, now: datetime) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Alertas")
        if not alerts:
            st.info("Sin alertas ahora mismo.")
        for a in alerts:
            st.markdown(
                f"{LEVEL_ICON.get(a.level, '•')} **{html.escape(a.title)}** "
                f"<span class='fc-chip'>{html.escape(a.status)}</span>  \n"
                f"<span style='color:#475569;font-size:.88rem'>{html.escape(a.detail)}</span>",
                unsafe_allow_html=True,
            )
        st.caption("Las alertas de 1h necesitan dos instantáneas separadas al menos una hora.")
    with right:
        st.subheader("Calendario")
        events = fc27_signals.upcoming_events(analyst, now)
        if events.empty:
            st.info("No hay eventos en data/fc27_analyst.json para los próximos 30 días.")
        else:
            events = events.assign(cuando=events["when"].map(fc27_signals.format_when))
            st.dataframe(
                events[["cuando", "label", "status", "source"]].rename(
                    columns={"cuando": "Cuándo", "label": "Evento", "status": "Estado", "source": "Fuente"}),
                hide_index=True, width="stretch",
            )
            st.caption("Calendario mantenido a mano en data/fc27_analyst.json.")


def render_history(conn, signals: pd.DataFrame, evaluated: pd.DataFrame) -> None:
    st.subheader("Precio de una carta en tu historial")
    options = signals.drop_duplicates("ea_id")
    labels = {f"{r['name']} · {r['rarity']} {r['overall']}": int(r["ea_id"]) for _, r in options.iterrows()}
    choice = st.selectbox("Carta", list(labels), key="hist_card")
    hist = fc27_history.price_history(conn, labels[choice]) if choice else pd.DataFrame()
    if len(hist) < 2:
        st.info("Hace falta al menos dos instantáneas de esta carta. El historial crece cada vez que abres "
                "la página (máximo una instantánea cada 10 min) o ejecutas scripts/fc27_snapshot.py.")
    else:
        fig = go.Figure(go.Scatter(x=hist["fetched_at"], y=hist["price"], mode="lines+markers",
                                   line=dict(color="#1D4ED8", width=2), marker=dict(size=6),
                                   hovertemplate="%{x|%d %b %H:%M} UTC<br>%{y:,} monedas<extra></extra>"))
        fig.update_layout(template="plotly_white", height=320, margin=dict(t=10, b=40, l=60, r=10),
                          yaxis_title="Monedas", separators=",.")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Calibración de señales")
    if evaluated.empty:
        st.info("Todavía no hay señales con el horizonte cumplido. Cada señal COMPRAR o RIESGO se guarda y se "
                "evalúa sola cuando pasa su horizonte (72h o 7 días).")
        return
    c1, c2, c3 = st.columns(3)
    for col, kind in zip((c1, c2), ("COMPRAR", "RIESGO")):
        sub = evaluated[evaluated["kind"] == kind]
        col.metric(f"Acierto {kind}", f"{sub['hit'].mean() * 100:.0f}%" if len(sub) else "—", f"{len(sub)} evaluadas",
                   delta_color="off")
    buys = evaluated[evaluated["kind"] == "COMPRAR"]
    c3.metric("Retorno medio neto (COMPRAR)", f"{buys['net_return_pct'].mean():+.1f}%" if len(buys) else "—",
              "tras el 5% de EA", delta_color="off")
    st.dataframe(evaluated.sort_values("created_at", ascending=False), hide_index=True, width="stretch")


def render_method() -> None:
    st.markdown(
        """
**Fuente.** FUT.GG, precios de su plataforma por defecto (presumiblemente consola). Se leen tres páginas:
momentum (las ~280 cartas con más movimiento en 24h), más baratas por rating y SBC activos.
FUTBIN no se usa: bloquea las peticiones automáticas.

**Historial.** Cada visita guarda una instantánea en `data/fc27_market.sqlite` (como mucho una cada 10 minutos).
Con él se calculan las variaciones de 1h, 6h, 3 días y 7 días. Para que crezca aunque no abras la app,
programa `python scripts/fc27_snapshot.py` (por ejemplo cada 30 minutos con cron o el Programador de tareas).

**Market Score (0-100).** Momentum 20 + Supply 15 + Demand 15 + Upcoming Content 15 + SBC/Evo Utility 15 +
News 10 + Risk/Downside 10. Reglas principales:
- *Momentum:* premia subidas moderadas (+2% a +10%); una caída en carta recién salida es normal (puntúa bajo);
  una caída en carta fuera de packs puede ser sobrerreacción.
- *Supply:* carta nueva en packs = mucha oferta; fuera de packs = oferta congelada; Icons/Heroes siempre en packs.
- *Demand:* tono de su rareza en 24h y si su precio está cerca de su valor como fodder.
- *Upcoming Content:* baja si sale una promo en menos de 48h (según el calendario del analista).
- *SBC/Evo Utility:* valor como fodder frente al precio; 0 si un SBC regala la misma carta más barata.
- *News:* 5 fijo (sin noticias automáticas).
- *Risk/Downside:* 10 − Risk Score / 10.

**Risk Score (0-100).** Parte de 25 y suma por subida vertical (≥20% en 24h), sustituto por SBC (+45, basta para RIESGO), precio alto
(≥1M, ≥5M), carta nueva en packs, promo próxima y caída fuerte en 1h; resta si el precio está cerca del suelo de fodder.

**Señales.** COMPRAR: Market Score ≥70 y Risk ≤45. RIESGO: Risk ≥70 o Market Score <40. VIGILAR: Market Score ≥55.
Objetivo de compra: +15% bruto (≈ +9% neto). Invalidación: −10% desde la entrada.

**Límites.** Ninguna señal es una certeza. Las reglas son del analista y se ajustarán con la calibración.
Las noticias, filtraciones y rumores se añaden a mano en `data/fc27_analyst.json`.
        """
    )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------


def main() -> None:
    with st.sidebar:
        st.markdown("### FC 27 Mercado")
        refresh = st.button("🔄 Actualizar ahora", type="primary", width="stretch")
        auto = st.toggle("Actualizar sola cada 10 min", value=False,
                         help="Mientras la página esté abierta, vuelve a descargar FUT.GG cada 10 minutos.")
        st.caption("Los datos se descargan al abrir la página y se reutilizan unos minutos para no saturar FUT.GG.")
    if refresh:
        _fetch_live.clear()

    @st.fragment(run_every=AUTO_REFRESH_S if auto else None)
    def body() -> None:
        conn = _db()
        with st.spinner("Descargando datos de FUT.GG…"):
            snap = _fetch_live()
        live = not snap.is_empty
        if not live:
            _fetch_live.clear()  # no guardar el fallo en caché: la próxima visita lo vuelve a intentar
            saved_snap = fc27_history.load_latest_snapshot(conn)
            if saved_snap is None:
                st.error("No se pudo descargar FUT.GG y no hay instantáneas guardadas. Revisa tu conexión y pulsa "
                         "**Actualizar ahora**. Detalle: " + "; ".join(snap.errors[:2]))
                st.stop()
            st.warning(f"FUT.GG no respondió. Se muestra la última instantánea guardada ({_age_text(saved_snap.fetched_at)}).")
            snap = saved_snap
        elif snap.errors:
            st.warning("Algunas páginas de FUT.GG fallaron; los datos pueden estar incompletos. "
                       + "; ".join(snap.errors[:2]))

        saved = fc27_history.save_snapshot(conn, snap) if live else False
        now = snap.fetched_at
        changes = fc27_history.price_changes(conn, now)
        analyst = fc27_signals.load_analyst_file()
        signals = fc27_signals.build_signals(snap, changes, analyst, now)
        if saved:
            fc27_history.record_signals(conn, signals, now)
        evaluated = fc27_history.evaluate_signals(conn, datetime.now(timezone.utc))
        alerts = fc27_signals.build_alerts(snap, signals, changes, analyst, now)

        st.markdown(
            f"""<div class="fc-head"><div><div class="fc-title">FC 27 Mercado</div>
            <div class="fc-sub">Señales de compra, vigilancia y riesgo para Ultimate Team · datos de FUT.GG</div></div>
            <div class="fc-stamp">Datos de {fc27_signals.format_when(now)}<br>{_age_text(now)}
            {' · en vivo' if live else ' · sin conexión'}</div></div>""",
            unsafe_allow_html=True,
        )
        events = fc27_signals.upcoming_events(analyst, now)
        nxt = events[events["when"] >= now].head(1)
        counts = signals["signal"].value_counts()
        tone = fc27_signals.overall_tone(snap.movers)
        st.markdown(ui_theme.stat_row([
            {"icon": "📈" if tone == "Alcista" else ("📉" if tone == "Bajista" else "➖"), "label": "Mercado (24h)",
             "value": tone, "accent": "blue"},
            {"icon": "▲", "label": "Comprar", "value": str(counts.get("COMPRAR", 0)), "accent": "green"},
            {"icon": "●", "label": "Vigilar", "value": str(counts.get("VIGILAR", 0)), "accent": "amber"},
            {"icon": "▼", "label": "Riesgo de caída", "value": str(counts.get("RIESGO", 0)), "accent": "gray"},
            {"icon": "🚨", "label": "Alertas críticas", "value": str(sum(a.level == "crítica" for a in alerts)),
             "accent": "gray"},
            {"icon": "🗂", "label": "Instantáneas guardadas", "value": str(fc27_history.snapshot_count(conn)),
             "accent": "blue"},
        ]), unsafe_allow_html=True)
        if not nxt.empty:
            e = nxt.iloc[0]
            st.markdown(f"**Próximo evento:** {html.escape(e['label'])} · {fc27_signals.format_when(e['when'])} "
                        f"<span class='fc-chip'>{html.escape(e['status'])}</span>", unsafe_allow_html=True)

        breadth = fc27_signals.market_breadth(snap.movers)
        if not breadth.empty:
            st.caption("Por rareza (24h): " + " · ".join(
                f"{r.rarity}: {r.tone.lower()} (mediana {r.median_24h:+.1f}%, {r.up}↑ {r.down}↓)".replace(".", ",")
                for r in breadth.itertuples(index=False)))

        tabs = st.tabs(["🎯 Señales", "🧱 Fodder y SBC", "📊 Movimientos", "🚨 Alertas y calendario",
                        "📈 Historial y calibración", "📘 Metodología"])
        with tabs[0]:
            render_signals(signals, evaluated, analyst, now)
        with tabs[1]:
            render_fodder(snap)
        with tabs[2]:
            render_movers(signals)
        with tabs[3]:
            render_alerts(alerts, analyst, now)
        with tabs[4]:
            render_history(conn, signals, evaluated)
        with tabs[5]:
            render_method()

    body()


main()
