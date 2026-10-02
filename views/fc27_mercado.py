"""Mercado FC 27 — señales de trading para EA SPORTS FC 27 Ultimate Team.

Página registrada en `app.py`. Al abrirla descarga datos en vivo de FUT.GG
(con caché de unos minutos), guarda una instantánea en el historial local y
calcula Market Score, Risk Score, señales y alertas.

Orden de la página (de lo más urgente a lo más detallado):
1. Resumen de hoy: mejor compra, mayor riesgo y próximo evento.
2. Alertas críticas, si las hay.
3. Pestañas: Señales, Mi lista, Mercado (con ficha de carta), Fodder y SBC,
   Alertas y Cómo funciona.

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

BUDGETS = {
    "Cualquier presupuesto": (0, float("inf")),
    "Bajo (hasta 20k)": (0, 20_000),
    "Medio (20k-100k)": (20_000, 100_000),
    "Alto (100k-500k)": (100_000, 500_000),
    "Premium (500k+)": (500_000, float("inf")),
}
PILL = {"COMPRAR": ("fc-buy", "▲ Comprar"), "VIGILAR": ("fc-watch", "● Vigilar"), "RIESGO": ("fc-risk", "▼ Riesgo")}
LEVEL_ICON = {"crítica": "🚨", "aviso": "⚠️", "info": "ℹ️"}

st.markdown(
    f"""
    <style>
    .fc-title {{ font-size:1.9rem; font-weight:800; color:var(--text-color, {ui_theme.INK_900}); line-height:1.1; margin:0; }}
    .fc-sub {{ color:{ui_theme.INK_600}; font-size:.92rem; margin:.2rem 0 1rem 0; }}
    .fc-pill {{ display:inline-block; font-size:.7rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
               padding:3px 9px; border-radius:999px; white-space:nowrap; }}
    .fc-buy {{ background:#DCFCE7; color:#15803D; }}
    .fc-watch {{ background:#FEF3C7; color:#92400E; }}
    .fc-risk {{ background:#FEE2E2; color:#B91C1C; }}
    .fc-chip {{ display:inline-block; font-size:.66rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
               padding:2px 7px; border-radius:5px; background:{ui_theme.GRAY_100}; color:{ui_theme.INK_600}; margin-left:6px; }}
    .fc-up {{ color:#1D4ED8; font-weight:600; }}
    .fc-down {{ color:#B91C1C; font-weight:600; }}
    .fc-hero {{ border-radius:14px; padding:14px 16px; border:1px solid {ui_theme.GRAY_200}; background:{ui_theme.WHITE};
               height:100%; color:{ui_theme.INK_900}; }}
    .fc-hero.buy {{ border-top:4px solid #16A34A; }}
    .fc-hero.risk {{ border-top:4px solid #DC2626; }}
    .fc-hero.event {{ border-top:4px solid {ui_theme.BLUE}; }}
    .fc-hero .k {{ font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:{ui_theme.INK_600}; }}
    .fc-hero .n {{ font-size:1.25rem; font-weight:800; margin:.25rem 0 .1rem 0; line-height:1.2; }}
    .fc-hero .p {{ font-size:1.05rem; font-weight:700; }}
    .fc-hero .d {{ font-size:.86rem; color:{ui_theme.INK_600}; margin-top:.35rem; line-height:1.45; }}
    .fc-strip {{ display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; font-size:.88rem; color:{ui_theme.INK_600};
                margin:.8rem 0 .2rem 0; }}
    .fc-strip b {{ color:{ui_theme.INK_900}; }}
    .fc-card-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }}
    .fc-card-name {{ font-weight:700; font-size:1.02rem; line-height:1.25; }}
    .fc-card-meta {{ color:{ui_theme.INK_600}; font-size:.8rem; }}
    .fc-plan {{ font-size:.86rem; line-height:1.55; }}
    .fc-plan span {{ color:{ui_theme.INK_600}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Datos y formato
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


def _in_budget(df: pd.DataFrame, budget: str) -> pd.DataFrame:
    lo, hi = BUDGETS[budget]
    return df[(df["price"] >= lo) & (df["price"] < hi)]


def _is_fodder(row) -> bool:
    return str(row["key"]).startswith("fodder")


def _card_title(row) -> str:
    return str(row["name"]) if _is_fodder(row) else f"{row['name']} {row['overall']}"


def _follow_button(conn, row, followed: set[int], key: str) -> None:
    """Botón para añadir o quitar una carta de Mi lista."""
    if _is_fodder(row) or pd.isna(row["ea_id"]):
        return
    ea_id = int(row["ea_id"])
    if ea_id in followed:
        if st.button("★ En mi lista", key=key, help="Quitar de Mi lista", width="stretch"):
            fc27_history.remove_from_watchlist(conn, ea_id)
            st.rerun()
    elif st.button("☆ Seguir", key=key, help="Añadir a Mi lista para seguir su precio", width="stretch"):
        fc27_history.add_to_watchlist(conn, row.to_dict(), datetime.now(timezone.utc))
        st.toast(f"{_card_title(row)} añadida a Mi lista", icon="⭐")
        st.rerun()


def _price_chart(conn, ea_id: int, height: int = 260) -> None:
    hist = fc27_history.price_history(conn, ea_id)
    if len(hist) < 2:
        st.caption("Aún no hay historial de esta carta: se necesitan al menos dos instantáneas.")
        return
    fig = go.Figure(go.Scatter(x=hist["fetched_at"], y=hist["price"], mode="lines+markers",
                               line=dict(color=ui_theme.BLUE, width=2), marker=dict(size=6),
                               hovertemplate="%{x|%d %b %H:%M} UTC<br>%{y:,} monedas<extra></extra>"))
    fig.update_layout(template="plotly_white", height=height, margin=dict(t=10, b=30, l=60, r=10),
                      yaxis_title="Monedas", separators=",.")
    st.plotly_chart(fig, width="stretch")


def _calibration_text(evaluated: pd.DataFrame, kind: str) -> str:
    sub = evaluated[evaluated["kind"] == kind] if not evaluated.empty else evaluated
    n = len(sub)
    if n < MIN_CALIBRATION:
        return f"Acierto real: sin calibrar ({n}/{MIN_CALIBRATION} señales evaluadas)."
    return f"Acierto real: {sub['hit'].mean() * 100:.0f}% en {n} señales evaluadas."


# ---------------------------------------------------------------------------
# Resumen de hoy
# ---------------------------------------------------------------------------


def render_today(signals: pd.DataFrame, events: pd.DataFrame, now: datetime) -> None:
    buy = fc27_signals.top_by_signal(signals, "COMPRAR", 1)
    risk = fc27_signals.top_by_signal(signals, "RIESGO", 1)
    nxt = events[events["when"] >= now].head(1)

    c1, c2, c3 = st.columns(3)
    with c1:
        if buy.empty:
            body = "<div class='n'>Nada claro hoy</div><div class='d'>Ninguna carta de tu presupuesto cumple las " \
                   "condiciones de compra. Esperar también es una decisión.</div>"
        else:
            r = buy.iloc[0]
            plan = fc27_signals.trade_plan(r)
            body = (f"<div class='n'>{html.escape(_card_title(r))}</div><div class='p'>{_fmt(r['price'])} monedas <span style='font-size:.75rem;font-weight:500'>(consola)</span></div>"
                    f"<div class='d'>Comprar: {html.escape(plan['zona'])}<br>Objetivo: {html.escape(plan['objetivo'])}"
                    f"<br>Market Score {int(r['market_score'])} · Riesgo {int(r['risk_score'])}</div>")
        st.markdown(f"<div class='fc-hero buy'><div class='k'>Mejor compra</div>{body}</div>", unsafe_allow_html=True)
    with c2:
        if risk.empty:
            body = "<div class='n'>Sin riesgos fuertes</div><div class='d'>Ninguna carta de tu presupuesto tiene " \
                   "señal de caída ahora mismo.</div>"
        else:
            r = risk.iloc[0]
            why = (r["risks"] or r["reasons"] or ["Risk Score alto."])[0]
            body = (f"<div class='n'>{html.escape(_card_title(r))}</div><div class='p'>{_fmt(r['price'])} monedas (consola) "
                    f"{_pct_html(r['pct_24h'])}</div><div class='d'>{html.escape(why)}</div>")
        st.markdown(f"<div class='fc-hero risk'><div class='k'>Mayor riesgo</div>{body}</div>", unsafe_allow_html=True)
    with c3:
        if nxt.empty:
            body = "<div class='n'>Sin eventos</div><div class='d'>Añade fechas en data/fc27_analyst.json.</div>"
        else:
            e = nxt.iloc[0]
            hours = (e["when"] - now).total_seconds() / 3600
            left = f"en {hours:.0f} h" if hours < 48 else f"en {hours / 24:.0f} días"
            body = (f"<div class='n'>{html.escape(e['label'])}</div><div class='p'>{left}</div>"
                    f"<div class='d'>{fc27_signals.format_when(e['when'])} · {html.escape(e['status'])}</div>")
        st.markdown(f"<div class='fc-hero event'><div class='k'>Próximo evento</div>{body}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------


def _signal_card(conn, row: pd.Series, followed: set[int], kind: str) -> None:
    with st.container(border=True):
        css, label = PILL[kind]
        meta = "" if _is_fodder(row) else f"{html.escape(str(row['rarity']))} · "
        st.markdown(
            f"<div class='fc-card-head'><div><div class='fc-card-name'>{html.escape(_card_title(row))}</div>"
            f"<div class='fc-card-meta'>{meta}Horizonte {row['horizon']}</div></div>"
            f"<span class='fc-pill {css}'>{label}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{_fmt(row['price'])}** monedas (consola) &nbsp; 24h {_pct_html(row['pct_24h'])}",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.progress(int(row["market_score"]) / 100, text=f"Market {int(row['market_score'])}")
        c2.progress(int(row["risk_score"]) / 100, text=f"Riesgo {int(row['risk_score'])}")
        plan = fc27_signals.trade_plan(row)
        st.markdown(
            f"<div class='fc-plan'><span>Entrada:</span> {html.escape(plan['zona'])}<br>"
            f"<span>Objetivo:</span> {html.escape(plan['objetivo'])}<br>"
            f"<span>Invalidación:</span> {html.escape(plan['invalidacion'])}</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Ver análisis"):
            if row["reasons"]:
                st.markdown("**Por qué**\n" + "\n".join(f"- {r}" for r in row["reasons"]))
            if row["risks"]:
                st.markdown("**Riesgos**\n" + "\n".join(f"- {r}" for r in row["risks"]))
            changes = " · ".join(f"{lbl} {_pct_html(row[col])}" for col, lbl in
                                 [("pct_1h", "1h"), ("pct_6h", "6h"), ("pct_24h", "24h"), ("pct_168h", "7d")])
            st.markdown(f"Variación: {changes}", unsafe_allow_html=True)
            bd = pd.DataFrame([(label, int(row[col]), mx) for col, label, mx in fc27_signals.SCORE_COMPONENTS],
                              columns=["Componente", "Puntos", "Máximo"])
            st.dataframe(bd, hide_index=True, width="stretch")
            if row.get("url"):
                st.markdown(f"[Abrir en FUT.GG ↗]({FUTGG}{row['url']})")
        _follow_button(conn, row, followed, key=f"follow_{kind}_{row['key']}")


def render_signals(conn, signals: pd.DataFrame, evaluated: pd.DataFrame, analyst: dict, now: datetime,
                   followed: set[int]) -> None:
    cols = st.columns(3)
    for col, kind in zip(cols, ["COMPRAR", "VIGILAR", "RIESGO"]):
        top = fc27_signals.top_by_signal(signals, kind)
        css, label = PILL[kind]
        with col:
            st.markdown(f"<span class='fc-pill {css}'>{label}</span> <span style='color:#94A3B8'>"
                        f"top {len(top)} de {int((signals['signal'] == kind).sum())}</span>", unsafe_allow_html=True)
            if kind != "VIGILAR":
                st.caption(_calibration_text(evaluated, kind))
            else:
                st.caption("Cartas con algo de interés que aún no cumplen todas las condiciones.")
            if top.empty:
                st.info("Ninguna carta de tu presupuesto cumple las condiciones ahora mismo.")
            for _, row in top.iterrows():
                _signal_card(conn, row, followed, kind)

    notes = fc27_signals.active_notes(analyst, now)
    if notes:
        with st.expander(f"📝 Notas del analista ({len(notes)})"):
            st.caption(f"Tesis manuales de data/fc27_analyst.json (actualizado {analyst.get('updated', '—')}). "
                       "Desaparecen solas al caducar.")
            for n in notes:
                css, label = PILL.get(n.get("signal", ""), ("fc-watch", n.get("signal", "")))
                st.markdown(f"<span class='fc-pill {css}'>{label}</span> **{html.escape(n['player'])}**  \n"
                            f"{html.escape(n['thesis'])}  \n*Invalidación:* {html.escape(n['invalidation'])}",
                            unsafe_allow_html=True)


def render_watchlist(conn, signals: pd.DataFrame) -> None:
    wl = fc27_history.watchlist(conn)
    if wl.empty:
        st.info("Tu lista está vacía. Pulsa **☆ Seguir** en una señal, o selecciona una carta en la pestaña "
                "**Mercado** y añádela. Aquí apuntas tus precios de PC y la app te calcula el precio para no perder, "
                "el objetivo y el beneficio real tras el 5% de EA.")
        return
    current = signals.drop_duplicates("ea_id").set_index("ea_id")
    wl["signal"] = wl["ea_id"].map(current["signal"]).fillna("—")
    wl["link"] = FUTGG + wl["url"].fillna("")

    st.markdown("##### Tus precios de PC")
    st.caption("FUT.GG no publica precios de cartas en PC de forma abierta. Apunta aquí a cuánto compraste cada carta "
               "en PC y cuánto vale ahora (lo ves en el juego o en FUT.GG); la app calcula el resto con el 5% de EA.")
    labels = {f"{r['name']} {'' if pd.isna(r['overall']) else int(r['overall'])}".strip(): r for _, r in wl.iterrows()}
    with st.form("pc_prices_form", border=True):
        c1, c2, c3, c4 = st.columns([2.2, 1.3, 1.3, 1])
        choice = c1.selectbox("Carta", list(labels), key="pc_card")
        row = labels[choice]
        buy = c2.number_input("Compré a (PC)", min_value=0, step=500, key=f"pc_buy_{row['ea_id']}",
                              value=None if pd.isna(row["buy_price_pc"]) else int(row["buy_price_pc"]),
                              placeholder="Ej. 150000")
        now_price = c3.number_input("Vale ahora (PC)", min_value=0, step=500, key=f"pc_now_{row['ea_id']}",
                                    value=None if pd.isna(row["price_pc"]) else int(row["price_pc"]),
                                    placeholder="Ej. 171000")
        c4.write("")
        if c4.form_submit_button("Guardar", type="primary", width="stretch"):
            fc27_history.set_pc_prices(conn, int(row["ea_id"]), buy, now_price, datetime.now(timezone.utc))
            st.toast(f"Precios de PC guardados para {choice}", icon="💾")
            st.rerun()

    st.dataframe(
        wl[["name", "overall", "buy_price_pc", "price_pc", "pc_net_now", "pc_net_now_pct", "pc_break_even",
            "pc_target", "pc_stop", "signal"]],
        hide_index=True, width="stretch",
        column_config={
            "name": "Carta", "overall": st.column_config.NumberColumn("OVR", format="%d"),
            "buy_price_pc": st.column_config.NumberColumn("Compré a (PC)", format="localized"),
            "price_pc": st.column_config.NumberColumn("Vale ahora (PC)", format="localized"),
            "pc_net_now": st.column_config.NumberColumn("Si vendes ya (neto)", format="localized",
                                                        help="Precio actual × 0,95 − precio de compra"),
            "pc_net_now_pct": st.column_config.NumberColumn("Neto %", format="%+.1f"),
            "pc_break_even": st.column_config.NumberColumn("Vender a ≥ (sin perder)", format="localized",
                                                           help="Compra / 0,95"),
            "pc_target": st.column_config.NumberColumn("Objetivo (+10% neto)", format="localized"),
            "pc_stop": st.column_config.NumberColumn("Invalidación (−10%)", format="localized"),
            "signal": "Señal actual",
        },
    )

    with st.expander("Precios de consola (FUT.GG) desde que añadiste cada carta"):
        st.caption("Precio de la última instantánea en la que apareció cada carta. Si una carta deja de estar entre "
                   "las que más se mueven en FUT.GG, su precio deja de actualizarse.")
        st.dataframe(
            wl[["name", "added_price", "last_price", "pct_since_added", "net_if_sold_pct", "link"]],
            hide_index=True, width="stretch",
            column_config={
                "name": "Carta",
                "added_price": st.column_config.NumberColumn("Al añadir (consola)", format="localized"),
                "last_price": st.column_config.NumberColumn("Último (consola)", format="localized"),
                "pct_since_added": st.column_config.NumberColumn("Cambio %", format="%+.1f"),
                "net_if_sold_pct": st.column_config.NumberColumn("Neto si vendes %", format="%+.1f"),
                "link": st.column_config.LinkColumn("FUT.GG", display_text="Abrir"),
            },
        )

    names = {f"{r['name']} {'' if pd.isna(r['overall']) else int(r['overall'])}".strip(): int(r["ea_id"])
             for _, r in wl.iterrows()}
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Ver gráfico (precio de consola) de", list(names), key="wl_chart")
    c2.write("")
    if c2.button("Quitar de Mi lista", key="wl_remove", width="stretch"):
        fc27_history.remove_from_watchlist(conn, names[choice])
        st.rerun()
    _price_chart(conn, names[choice])


OTHER_CARD = "✏️ Otra carta (escribir nombre)"


def render_trades(conn, signals: pd.DataFrame) -> None:
    df = fc27_history.trades(conn)
    s = fc27_history.trade_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beneficio neto", f"{_fmt(s['net_profit'])}", f"{s['closed']} cerradas", delta_color="off")
    c2.metric("ROI", "—" if s["roi_pct"] is None else f"{s['roi_pct']:+.1f}%".replace(".", ","),
              help="Beneficio neto / dinero invertido en operaciones cerradas")
    c3.metric("Operaciones ganadoras", "—" if s["win_rate_pct"] is None else f"{s['win_rate_pct']:.0f}%")
    c4.metric("Invertido en abiertas", _fmt(s["capital_in_open"]), f"{s['open']} abiertas", delta_color="off")
    st.caption(f"Precios de PC por unidad, como en el juego. Impuesto de EA pagado en ventas: {_fmt(s['taxes_paid'])}.")

    left, right = st.columns(2)
    with left, st.form("trade_buy", border=True, clear_on_submit=True):
        st.markdown("**Registrar compra**")
        cards = signals[~signals["key"].str.startswith("fodder")].drop_duplicates("ea_id")
        wl = fc27_history.watchlist(conn)
        options = {f"{r['name']} {'' if pd.isna(r['overall']) else int(r['overall'])}".strip(): r for _, r in wl.iterrows()}
        for _, r in cards.iterrows():
            options.setdefault(f"{r['name']} {r['overall']}", r)
        choice = st.selectbox("Carta", [OTHER_CARD] + list(options), index=None, placeholder="Busca una carta…")
        custom = st.text_input("Nombre (si elegiste Otra carta)", placeholder="Ej. Fodder 84")
        b1, b2 = st.columns(2)
        price = b1.number_input("Precio de compra (PC, por unidad)", min_value=0, step=50, value=None)
        qty = b2.number_input("Unidades", min_value=1, step=1, value=1)
        note = st.text_input("Nota (opcional)", placeholder="Ej. compra para SBC POTM")
        if st.form_submit_button("Guardar compra", type="primary", width="stretch"):
            row = options.get(choice) if choice and choice != OTHER_CARD else None
            name = custom.strip() if (choice in (None, OTHER_CARD)) else choice
            sig = None
            if row is not None and not pd.isna(row["ea_id"]):
                match = signals[signals["ea_id"] == row["ea_id"]]
                sig = match.iloc[0]["signal"] if not match.empty and isinstance(match.iloc[0]["signal"], str) else None
            try:
                fc27_history.open_trade(conn, name, price, datetime.now(timezone.utc), int(qty),
                                        None if row is None else row["ea_id"], sig, note or None)
                st.toast(f"Compra guardada: {name}", icon="📒")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    open_df = df[df["status"] == "Abierta"]
    with right, st.form("trade_sell", border=True, clear_on_submit=True):
        st.markdown("**Registrar venta**")
        if open_df.empty:
            st.caption("No tienes operaciones abiertas. Registra primero una compra.")
        labels = {f"#{r.id} · {r.card_name} · {r.quantity}× a {_fmt(r.buy_price)}": int(r.id)
                  for r in open_df.itertuples(index=False)}
        pick = st.selectbox("Operación abierta", list(labels), index=None, placeholder="Elige una operación…")
        sell = st.number_input("Precio de venta (PC, por unidad)", min_value=0, step=50, value=None)
        if pick:
            r = open_df[open_df["id"] == labels[pick]].iloc[0]
            st.caption(f"Para no perder vende a ≥ {_fmt(r['break_even'])}; objetivo (+10% neto): {_fmt(r['target'])}.")
        if st.form_submit_button("Guardar venta", type="primary", width="stretch", disabled=open_df.empty):
            try:
                if not pick:
                    raise ValueError("Elige la operación que vendiste.")
                fc27_history.close_trade(conn, labels[pick], sell, datetime.now(timezone.utc))
                st.toast("Venta guardada", icon="💰")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    if df.empty:
        st.info("Todavía no hay operaciones. Cada vez que compres o vendas en el juego, apúntalo aquí: verás tu "
                "beneficio real después del 5% de EA y qué señales te hacen ganar dinero.")
        return
    st.markdown("##### Historial")
    st.dataframe(
        df[["id", "status", "card_name", "quantity", "buy_price", "sell_price", "net_profit", "roi_pct",
            "break_even", "target", "signal_at_buy", "buy_at", "sell_at", "note"]].assign(
            buy_at=pd.to_datetime(df["buy_at"], utc=True), sell_at=pd.to_datetime(df["sell_at"], utc=True)),
        hide_index=True, width="stretch",
        column_config={
            "id": "#", "status": "Estado", "card_name": "Carta", "quantity": "Uds.",
            "buy_price": st.column_config.NumberColumn("Compra", format="localized"),
            "sell_price": st.column_config.NumberColumn("Venta", format="localized"),
            "net_profit": st.column_config.NumberColumn("Beneficio neto", format="localized"),
            "roi_pct": st.column_config.NumberColumn("ROI %", format="%+.1f"),
            "break_even": st.column_config.NumberColumn("No perder ≥", format="localized"),
            "target": st.column_config.NumberColumn("Objetivo", format="localized"),
            "signal_at_buy": "Señal al comprar",
            "buy_at": st.column_config.DatetimeColumn("Comprada", format="D MMM HH:mm"),
            "sell_at": st.column_config.DatetimeColumn("Vendida", format="D MMM HH:mm"),
            "note": "Nota",
        },
    )
    by_signal = fc27_history.trade_results_by_signal(df)
    if not by_signal.empty:
        st.markdown("##### ¿Qué señales te hacen ganar dinero?")
        st.dataframe(by_signal, hide_index=True, width="stretch", column_config={
            "signal_at_buy": "Señal al comprar", "operations": "Operaciones",
            "win_rate_pct": st.column_config.NumberColumn("Ganadoras %", format="%.0f"),
            "net_profit": st.column_config.NumberColumn("Beneficio neto", format="localized"),
            "roi_pct": st.column_config.NumberColumn("ROI %", format="%+.1f")})
    with st.expander("Borrar una operación"):
        ids = {f"#{r.id} · {r.card_name} ({r.status.lower()})": int(r.id) for r in df.itertuples(index=False)}
        victim = st.selectbox("Operación", list(ids), index=None, placeholder="Elige la operación a borrar…",
                              key="trade_delete_pick")
        if st.button("Borrar", disabled=victim is None, key="trade_delete"):
            fc27_history.delete_trade(conn, ids[victim])
            st.rerun()


def render_market(conn, signals: pd.DataFrame, followed: set[int]) -> None:
    cards = signals[~signals["key"].str.startswith("fodder")].copy()
    if cards.empty:
        st.info("No hay cartas en tu presupuesto. Cambia el presupuesto en la barra lateral.")
        return
    f1, f2, f3 = st.columns([2, 1.2, 1.8])
    rarities = sorted(cards["rarity"].dropna().unique())
    sel_rar = f1.multiselect("Rareza", rarities, default=[], placeholder="Todas", key="mv_rar")
    direction = f2.segmented_control("Dirección 24h", ["Todas", "Suben", "Bajan"], default="Todas", key="mv_dir")
    query = f3.text_input("Buscar jugador", key="mv_q", placeholder="Ej. Mbappé")
    view = cards
    if sel_rar:
        view = view[view["rarity"].isin(sel_rar)]
    if direction in ("Suben", "Bajan"):
        view = view[view["pct_24h"] > 0] if direction == "Suben" else view[view["pct_24h"] < 0]
    if query:
        view = view[view["name"].str.contains(query, case=False, na=False)]
    view = view.reset_index(drop=True)
    view = view.assign(link=FUTGG + view["url"].fillna(""), signal=view["signal"].fillna("—"),
                       followed=view["ea_id"].isin(followed).map({True: "⭐", False: ""}))
    for col in ("pct_1h", "pct_6h", "pct_24h", "pct_72h", "pct_168h"):
        view[col] = pd.to_numeric(view[col], errors="coerce")
    history_cols = [c for c in ("pct_1h", "pct_6h", "pct_72h", "pct_168h") if view[c].notna().any()]
    pct_cols = [c for c in ("pct_1h", "pct_6h", "pct_24h", "pct_72h", "pct_168h") if c == "pct_24h" or c in history_cols]
    hidden = 4 - len(history_cols)
    st.caption(f"{len(view)} cartas. Selecciona una fila para ver su ficha."
               + (" Las columnas de 1h, 6h, 3d y 7d aparecen cuando tu historial las cubre." if hidden else ""))
    event = st.dataframe(
        view[["followed", "name", "overall", "rarity", "price", *pct_cols, "market_score", "risk_score", "signal"]],
        hide_index=True, width="stretch", height=480, on_select="rerun", selection_mode="single-row", key="mv_table",
        column_config={
            "followed": st.column_config.TextColumn("", help="⭐ = está en Mi lista", width="small"),
            "name": "Carta", "overall": st.column_config.NumberColumn("OVR", format="%d"), "rarity": "Rareza",
            "price": st.column_config.NumberColumn("Precio consola", format="localized"),
            "pct_1h": st.column_config.NumberColumn("1h %", format="%+.1f"),
            "pct_6h": st.column_config.NumberColumn("6h %", format="%+.1f"),
            "pct_24h": st.column_config.NumberColumn("24h %", format="%+.1f"),
            "pct_72h": st.column_config.NumberColumn("3d %", format="%+.1f"),
            "pct_168h": st.column_config.NumberColumn("7d %", format="%+.1f"),
            "market_score": st.column_config.ProgressColumn("Market", min_value=0, max_value=100, format="%d"),
            "risk_score": st.column_config.ProgressColumn("Riesgo", min_value=0, max_value=100, format="%d"),
            "signal": "Señal",
        },
    )
    rows = event.selection.rows if event and event.selection else []
    if not rows:
        return
    row = view.iloc[rows[0]]
    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            kind = row["signal"] if row["signal"] in PILL else None
            pill = f"<span class='fc-pill {PILL[kind][0]}'>{PILL[kind][1]}</span>" if kind else \
                   "<span class='fc-chip'>Sin señal</span>"
            st.markdown(f"#### {html.escape(_card_title(row))}  \n{html.escape(str(row['rarity']))} · {pill}",
                        unsafe_allow_html=True)
            st.markdown(f"**{_fmt(row['price'])}** monedas (consola) · 24h {_pct_html(row['pct_24h'])}  \n"
                        f"Market Score **{int(row['market_score'])}** ({row['label']}) · Riesgo **{int(row['risk_score'])}**",
                        unsafe_allow_html=True)
            plan = fc27_signals.trade_plan(row)
            st.markdown(f"<div class='fc-plan'><span>Entrada:</span> {html.escape(plan['zona'])}<br>"
                        f"<span>Objetivo:</span> {html.escape(plan['objetivo'])}<br>"
                        f"<span>Invalidación:</span> {html.escape(plan['invalidacion'])}</div>", unsafe_allow_html=True)
            for r in row["reasons"]:
                st.markdown(f"✅ {r}")
            for r in row["risks"]:
                st.markdown(f"⚠️ {r}")
            _follow_button(conn, row, followed, key=f"follow_market_{row['key']}")
            st.markdown(f"[Abrir en FUT.GG ↗]({row['link']})")
        with right:
            _price_chart(conn, int(row["ea_id"]), height=300)


def render_fodder(snap: fc27_market.MarketSnapshot) -> None:
    fodder = fc27_market.fodder_table(snap.cheapest)
    st.caption("Los SBC piden puntos de Item Score. Cuanto menos cueste cada punto, mejor fodder. "
               "El precio de referencia es la mediana de las 5 cartas más baratas de cada rating.")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Monedas por punto de Item Score")
        if fodder.empty:
            st.info("No hay datos de las cartas más baratas por rating.")
        else:
            colors = ["#B8860B" if b else "#93C5FD" for b in fodder["is_best"]]
            fig = go.Figure(go.Bar(
                x=fodder["overall"].astype(str), y=fodder["coins_per_point"], marker_color=colors,
                customdata=fodder[["name", "price", "item_score"]],
                hovertemplate="Rating %{x} · más barato: %{customdata[0]}<br>Referencia %{customdata[1]:,} monedas · "
                              "%{customdata[2]:,} pts<br>%{y:.2f} monedas por punto<extra></extra>",
            ))
            fig.update_layout(template="plotly_white", height=300, margin=dict(t=10, b=40, l=40, r=10),
                              xaxis_title="Rating", yaxis_title="Monedas por punto", separators=",.")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                fodder.drop(columns=["is_best", "score_verified"]), hide_index=True, width="stretch",
                column_config={
                    "overall": "Rating", "name": "Más barato",
                    "min_price": st.column_config.NumberColumn("Mínimo", format="localized"),
                    "price": st.column_config.NumberColumn("Referencia", format="localized"),
                    "item_score": st.column_config.NumberColumn("Item Score", format="localized"),
                    "coins_per_point": st.column_config.NumberColumn("Monedas/punto", format="%.2f"),
                },
            )
            floor = fc27_market.fodder_floor_price(snap.cheapest)
            st.caption(f"Precio mínimo observado en 81-84: {_fmt(floor)} (probable suelo de EA). "
                       "Item Score de 83 y 89 tomado de tablas de terceros.")
    with right:
        st.markdown("##### SBC activos")
        if snap.sbcs.empty:
            st.info("No hay datos de SBC.")
        else:
            sb = snap.sbcs.copy()
            sb["link"] = FUTGG + sb["url"].fillna("")
            sb["termina"] = pd.to_datetime(sb["end_time"], utc=True, errors="coerce")
            sb = sb[sb["termina"] < pd.Timestamp("2030-01-01", tz="UTC")]  # fuera los permanentes
            for col in ("score_requirement", "cost", "cost_pc", "award_overall"):
                sb[col] = pd.to_numeric(sb[col], errors="coerce")
            sb = sb.sort_values("cost_pc", ascending=False)
            st.dataframe(
                sb[["name", "score_requirement", "cost_pc", "cost", "termina", "award_name", "link"]],
                hide_index=True, width="stretch",
                column_config={
                    "name": "SBC", "score_requirement": st.column_config.NumberColumn("Puntos", format="localized"),
                    "cost_pc": st.column_config.NumberColumn("Coste PC", format="localized"),
                    "cost": st.column_config.NumberColumn("Consola", format="localized"),
                    "termina": st.column_config.DatetimeColumn("Termina (UTC)", format="ddd D MMM HH:mm"),
                    "award_name": "Premio", "link": st.column_config.LinkColumn("FUT.GG", display_text="Abrir"),
                },
            )
            st.caption("Coste estimado por FUT.GG con la ruta más barata. No incluye los SBC permanentes.")


def render_alerts(alerts: list[fc27_signals.Alert], analyst: dict, now: datetime) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Alertas")
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
        st.markdown("##### Calendario")
        events = fc27_signals.upcoming_events(analyst, now)
        if events.empty:
            st.info("No hay eventos en data/fc27_analyst.json para los próximos 30 días.")
        else:
            events = events.assign(cuando=events["when"].map(fc27_signals.format_when))
            st.dataframe(
                events[["cuando", "label", "status", "source"]],
                hide_index=True, width="stretch",
                column_config={"cuando": "Cuándo", "label": "Evento", "status": "Estado", "source": "Fuente"},
            )
            st.caption("Calendario mantenido a mano en data/fc27_analyst.json.")


def render_help(evaluated: pd.DataFrame, snap: fc27_market.MarketSnapshot) -> None:
    st.markdown("##### Estado del mercado por rareza (24h)")
    breadth = fc27_signals.market_breadth(snap.movers)
    if not breadth.empty:
        st.dataframe(breadth, hide_index=True, width="stretch", column_config={
            "rarity": "Rareza", "cards": "Cartas", "median_24h": st.column_config.NumberColumn("Mediana 24h %", format="%+.1f"),
            "up": "Suben", "down": "Bajan", "tone": "Tono"})

    st.markdown("##### ¿Aciertan las señales?")
    if evaluated.empty:
        st.info("Todavía no hay señales con el horizonte cumplido. Cada señal COMPRAR o RIESGO se guarda y se "
                "evalúa sola cuando pasa su horizonte (72h o 7 días).")
    else:
        c1, c2, c3 = st.columns(3)
        for col, kind in zip((c1, c2), ("COMPRAR", "RIESGO")):
            sub = evaluated[evaluated["kind"] == kind]
            col.metric(f"Acierto {kind}", f"{sub['hit'].mean() * 100:.0f}%" if len(sub) else "—",
                       f"{len(sub)} evaluadas", delta_color="off")
        buys = evaluated[evaluated["kind"] == "COMPRAR"]
        c3.metric("Retorno medio neto (COMPRAR)", f"{buys['net_return_pct'].mean():+.1f}%" if len(buys) else "—",
                  "tras el 5% de EA", delta_color="off")
        with st.expander("Ver señales evaluadas"):
            st.dataframe(evaluated.sort_values("created_at", ascending=False), hide_index=True, width="stretch")

    st.markdown("##### Cómo funciona")
    st.markdown(
        """
**Fuente.** FUT.GG, precios de su plataforma por defecto (presumiblemente consola). Se leen tres páginas:
las ~280 cartas con más movimiento en 24h, las más baratas por rating y los SBC activos. FUTBIN no se usa:
bloquea las peticiones automáticas.

**Historial.** Cada visita guarda una instantánea en `data/fc27_market.sqlite` (como mucho una cada 10 minutos).
Con él se calculan las variaciones de 1h, 6h, 3 días y 7 días, Mi lista y el acierto de las señales. Para que
crezca aunque no abras la app, programa `python scripts/fc27_snapshot.py` (por ejemplo cada 30 minutos).

**Market Score (0-100).** Momentum 20 + Supply 15 + Demand 15 + Upcoming Content 15 + SBC/Evo Utility 15 +
News 10 + Risk/Downside 10. Es un indicador comparativo, no una probabilidad.
- *Momentum:* premia subidas moderadas (+2% a +10%); una caída en carta recién salida es normal;
  una caída en carta fuera de packs puede ser sobrerreacción.
- *Supply:* carta nueva en packs = mucha oferta; fuera de packs = oferta congelada; Icons/Heroes siempre en packs.
- *Demand:* tono de su rareza en 24h y si su precio está cerca de su valor como fodder.
- *Upcoming Content:* baja si sale una promo en menos de 48h (según el calendario del analista).
- *SBC/Evo Utility:* valor como fodder frente al precio; 0 si un SBC regala la misma carta más barata.
- *News:* 5 fijo (sin noticias automáticas).
- *Risk/Downside:* 10 − Risk Score / 10.

**Risk Score (0-100).** Parte de 25 y suma por subida vertical (≥20% en 24h), sustituto por SBC (+45, basta para
RIESGO), precio alto (≥1M, ≥5M), carta nueva en packs, promo próxima y caída fuerte en 1h; resta si el precio está
cerca del suelo de fodder.

**Señales.** COMPRAR: Market Score ≥70 y Riesgo ≤45. RIESGO: Riesgo ≥70 o Market Score <40. VIGILAR: Market Score ≥55.
Objetivo de compra: +15% bruto (≈ +9% neto tras el 5% de EA). Invalidación: −10% desde la entrada.

**Límites.** Ninguna señal es una certeza. Las noticias, filtraciones y rumores se añaden a mano en
`data/fc27_analyst.json`.
        """
    )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------


def main() -> None:
    with st.sidebar:
        st.markdown("##### Datos de mercado")
        refresh = st.button("🔄 Actualizar ahora", type="primary", width="stretch")
        auto = st.toggle("Actualizar sola cada 10 min", value=False,
                         help="Mientras la página esté abierta, vuelve a descargar FUT.GG cada 10 minutos.")
        st.markdown("##### Tu presupuesto")
        budget = st.selectbox("Precio máximo por carta", list(BUDGETS), key="fc_budget", label_visibility="collapsed",
                              help="Filtra el resumen, las señales y la tabla de mercado (con precios de consola).")
        st.caption("Plataforma: PC. FUT.GG solo publica abiertamente los costes de SBC en PC; los precios de cartas "
                   "son de consola. Tus precios de PC van en ⭐ Mi lista y 📒 Operaciones.")
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
                st.error("No se pudo descargar FUT.GG y no hay datos guardados. Revisa tu conexión y pulsa "
                         "**Actualizar ahora**. Detalle: " + "; ".join(snap.errors[:2]))
                st.stop()
            snap = saved_snap

        saved = fc27_history.save_snapshot(conn, snap) if live else False
        now = snap.fetched_at
        changes = fc27_history.price_changes(conn, now)
        analyst = fc27_signals.load_analyst_file()
        all_signals = fc27_signals.build_signals(snap, changes, analyst, now)
        if saved:
            fc27_history.record_signals(conn, all_signals, now)
        evaluated = fc27_history.evaluate_signals(conn, datetime.now(timezone.utc))
        alerts = fc27_signals.build_alerts(snap, all_signals, changes, analyst, now)
        events = fc27_signals.upcoming_events(analyst, now)
        followed = fc27_history.watchlist_ids(conn)

        is_fodder = all_signals["key"].str.startswith("fodder")
        signals = pd.concat([all_signals[is_fodder], _in_budget(all_signals[~is_fodder], budget)])

        badge = "🟢 En vivo" if live else "🟠 Sin conexión"
        st.markdown("<div class='fc-title'>Mercado FC 27</div>"
                    "<div class='fc-sub'>Qué comprar, qué vigilar y qué vender hoy en Ultimate Team<br>"
                    f"<span style='font-size:.82rem'>{badge} · FUT.GG {fc27_signals.format_when(now)} "
                    f"({_age_text(now)}) · {fc27_history.snapshot_count(conn)} instantáneas guardadas</span></div>",
                    unsafe_allow_html=True)
        if not live:
            st.warning(f"FUT.GG no respondió. Estás viendo los datos guardados {_age_text(now)}.")
        elif snap.errors:
            st.warning("Algunas páginas de FUT.GG fallaron; los datos pueden estar incompletos.")

        render_today(signals, events, now)

        counts = signals["signal"].value_counts()
        st.markdown(
            f"<div class='fc-strip'><span>Mercado 24h: <b>{fc27_signals.overall_tone(snap.movers)}</b></span>"
            f"<span>Señales en tu presupuesto: <b>{counts.get('COMPRAR', 0)}</b> comprar · "
            f"<b>{counts.get('VIGILAR', 0)}</b> vigilar · <b>{counts.get('RIESGO', 0)}</b> riesgo</span></div>",
            unsafe_allow_html=True,
        )
        for a in fc27_signals.headline_alerts(alerts, followed)[:3]:
            show = st.error if a.level == "crítica" else st.warning
            show(f"**{a.title}** · {a.detail}", icon="🚨" if a.level == "crítica" else "⚠️")

        # Etiquetas fijas (sin contadores): si cambian, Streamlit vuelve a la primera pestaña
        # después de guardar algo, y el usuario pierde dónde estaba.
        tabs = st.tabs(["🎯 Señales", "⭐ Mi lista", "📒 Operaciones", "📊 Mercado",
                        "🧱 Fodder y SBC", "🚨 Alertas", "📘 Cómo funciona"])
        with tabs[0]:
            render_signals(conn, signals, evaluated, analyst, now, followed)
        with tabs[1]:
            render_watchlist(conn, all_signals)
        with tabs[2]:
            render_trades(conn, all_signals)
        with tabs[3]:
            render_market(conn, signals, followed)
        with tabs[4]:
            render_fodder(snap)
        with tabs[5]:
            render_alerts(alerts, analyst, now)
        with tabs[6]:
            render_help(evaluated, snap)

    body()


main()
