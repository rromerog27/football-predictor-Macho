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
from datetime import datetime, timedelta, timezone

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
    .fc-up {{ color:#15803D; font-weight:600; }}
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
    .fc-badges {{ display:flex; flex-wrap:wrap; gap:6px; margin:.35rem 0 1rem 0; }}
    .fc-badge {{ display:inline-flex; align-items:center; gap:5px; font-size:.78rem; font-weight:600;
                padding:3px 10px; border-radius:999px; background:{ui_theme.GRAY_100}; color:{ui_theme.INK_600}; }}
    .fc-badge.live {{ background:#DCFCE7; color:#15803D; }}
    .fc-badge.off {{ background:#FFEDD5; color:#9A3412; }}
    .fc-banner {{ display:flex; gap:10px; align-items:flex-start; padding:9px 14px; border-radius:10px;
                 font-size:.88rem; line-height:1.4; margin:.35rem 0; border:1px solid; }}
    .fc-banner.crit {{ background:#FEF2F2; border-color:#FECACA; color:#991B1B; }}
    .fc-banner.warn {{ background:#FFFBEB; border-color:#FDE68A; color:#92400E; }}
    .fc-banner b {{ font-weight:700; }}
    .fc-sig {{ display:flex; flex-direction:column; gap:10px; }}
    .fc-sig-top {{ display:flex; gap:12px; align-items:center; }}
    .fc-ovr {{ flex:0 0 auto; width:46px; height:52px; border-radius:8px; display:flex; flex-direction:column;
              align-items:center; justify-content:center; font-weight:800; line-height:1;
              background:linear-gradient(160deg,#F7E08A 0%,#D4AF37 55%,#B8860B 100%); color:#3B2F0B;
              box-shadow:inset 0 0 0 1px rgba(0,0,0,.08); }}
    .fc-ovr .r {{ font-size:1.25rem; }}
    .fc-ovr .t {{ font-size:.55rem; font-weight:700; letter-spacing:.05em; margin-top:3px; text-transform:uppercase; }}
    .fc-ovr.special {{ background:linear-gradient(160deg,#1F2937 0%,#111827 100%); color:#F7E08A; }}
    .fc-ovr.fodder {{ background:linear-gradient(160deg,#E2E8F0 0%,#CBD5E1 100%); color:#334155; }}
    .fc-sig-id {{ flex:1 1 auto; min-width:0; }}
    .fc-sig-name {{ font-weight:700; font-size:1rem; line-height:1.25; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .fc-sig-meta {{ color:{ui_theme.INK_600}; font-size:.78rem; }}
    .fc-stats {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:6px; }}
    .fc-stat {{ background:{ui_theme.GRAY_50}; border:1px solid {ui_theme.GRAY_100}; border-radius:8px; padding:6px 8px; min-width:0; }}
    .fc-stat .k {{ font-size:.66rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:{ui_theme.INK_400}; }}
    .fc-stat .v {{ font-size:.92rem; font-weight:700; color:{ui_theme.INK_900}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .fc-meter {{ height:4px; border-radius:2px; background:{ui_theme.GRAY_200}; margin-top:4px; overflow:hidden; }}
    .fc-meter i {{ display:block; height:100%; border-radius:2px; }}
    .fc-spark {{ display:flex; align-items:center; justify-content:space-between; gap:8px; font-size:.75rem; color:{ui_theme.INK_400}; }}
    .fc-planline {{ font-size:.84rem; line-height:1.5; color:{ui_theme.INK_900}; }}
    .fc-planline span {{ color:{ui_theme.INK_600}; }}
    .fc-since {{ background:#EFF6FF; border:1px solid #BFDBFE; color:#1E3A8A; border-radius:10px; padding:8px 14px;
                 font-size:.88rem; margin:.2rem 0 .6rem 0; }}
    .fc-empty {{ border:1px dashed {ui_theme.GRAY_200}; border-radius:12px; padding:14px 16px; color:{ui_theme.INK_600};
                font-size:.88rem; background:{ui_theme.GRAY_50}; }}
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


def _fmt_short(n) -> str:
    """Precio corto para espacios pequeños: 1,70M · 995k · 14.000."""
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M".replace(".", ",")
    if n >= 100_000:
        return f"{n / 1000:.0f}k"
    return _fmt(n)


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


RANGES = {"24h": timedelta(hours=24), "7 días": timedelta(days=7), "Todo": None}
EVENT_COLORS = {"promo": "#7C3AED", "sbc": "#0891B2", "totw": "#B45309", "season": "#475569", "other": "#64748B"}


def _all_events(analyst: dict) -> pd.DataFrame:
    """Todos los eventos del calendario del analista (pasados y futuros), con fecha UTC."""
    rows = []
    for e in analyst.get("events", []):
        try:
            when = pd.Timestamp(e["when"]).tz_convert("UTC") if pd.Timestamp(e["when"]).tzinfo else \
                pd.Timestamp(e["when"]).tz_localize("UTC")
        except (KeyError, ValueError):
            continue
        rows.append({"when": when, "type": e.get("type", "other"), "label": e.get("label", "")})
    return pd.DataFrame(rows, columns=["when", "type", "label"])


def _price_chart(conn, ea_id: int, key: str, height: int = 320, row=None, analyst: dict | None = None) -> None:
    """Gráfico de precio (consola) de una carta con selector de rango, niveles del
    plan, eventos del calendario y tus compras y ventas de 📒 Operaciones."""
    hist = fc27_history.price_history(conn, ea_id)
    if len(hist) < 2:
        st.caption("Aún no hay historial de esta carta: se necesitan al menos dos instantáneas. Activa las "
                   "instantáneas automáticas para que crezca solo.")
        return
    rng = st.segmented_control("Rango", list(RANGES), default="7 días", key=f"rng_{key}",
                               label_visibility="collapsed") or "7 días"
    end = hist["fetched_at"].max()
    start = end - RANGES[rng] if RANGES[rng] is not None else hist["fetched_at"].min()
    view = hist[hist["fetched_at"] >= start]
    if len(view) < 2:
        view = hist.tail(2)
        start = view["fetched_at"].min()

    fig = go.Figure(go.Scatter(
        x=view["fetched_at"], y=view["price"], mode="lines+markers", name="Precio (consola)",
        line=dict(color=ui_theme.BLUE, width=2), marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(29,78,216,0.06)",
        hovertemplate="%{x|%a %d %b %H:%M} UTC<br>%{y:,} monedas<extra></extra>",
    ))
    lo, hi = float(view["price"].min()), float(view["price"].max())

    if row is not None:
        level_colors = {"Objetivo": "#15803D", "Stop": "#DC2626", "Entrada": "#B45309",
                        "Caída posible": "#DC2626", "Invalidación": "#475569"}
        for name, value in fc27_signals.plan_levels(row).items():
            fig.add_hline(y=value, line=dict(color=level_colors.get(name, "#64748B"), width=1, dash="dot"),
                          annotation_text=f"{name} {_fmt_short(value)}", annotation_position="top left",
                          annotation_font=dict(size=11, color=level_colors.get(name, "#64748B")))
            lo, hi = min(lo, value), max(hi, value)

    events_at = None
    if analyst:
        ev = _all_events(analyst)
        ev = ev[(ev["when"] >= start) & (ev["when"] <= end + timedelta(days=2))]
        if not ev.empty:
            # Un marcador por momento (varios eventos a la misma hora se juntan) con el texto al pasar el ratón:
            # así las etiquetas no se pisan entre sí ni con las líneas del plan.
            events_at = ev.groupby("when").agg(label=("label", " · ".join), type=("type", "first")).reset_index()
            for e in events_at.itertuples(index=False):
                fig.add_vline(x=e.when, line=dict(color=EVENT_COLORS.get(e.type, "#64748B"), width=1, dash="dash"))

    trades = fc27_history.trades(conn)
    if not trades.empty:
        mine = trades[trades["ea_id"] == ea_id]
        for tr in mine.itertuples(index=False):
            for when, price, label, color in ((tr.buy_at, tr.buy_price, "Compra", "#15803D"),
                                              (tr.sell_at, tr.sell_price, "Venta", "#DC2626")):
                if when is None or pd.isna(when):
                    continue
                ts = pd.Timestamp(when).tz_convert("UTC")
                if ts < start:
                    continue
                fig.add_vline(x=ts, line=dict(color=color, width=2))
                fig.add_annotation(x=ts, y=0, yref="paper", text=f"{label} PC {_fmt_short(price)}", showarrow=False,
                                   xanchor="left", yanchor="bottom", bgcolor="rgba(255,255,255,0.85)",
                                   font=dict(size=10, color=color))

    pad = (hi - lo) * 0.12 or hi * 0.05
    if events_at is not None:
        fig.add_trace(go.Scatter(
            x=events_at["when"], y=[hi + pad * 0.6] * len(events_at), mode="markers", name="Eventos",
            marker=dict(symbol="triangle-down", size=11,
                        color=[EVENT_COLORS.get(x, "#64748B") for x in events_at["type"]]),
            text=events_at["label"], hovertemplate="📅 %{text}<br>%{x|%a %d %b %H:%M} UTC<extra></extra>",
        ))
    fig.update_layout(template="plotly_white", height=height, margin=dict(t=10, b=30, l=60, r=10),
                      yaxis=dict(title="Monedas (consola)", range=[max(0, lo - pad), hi + pad], gridcolor="#EEF2F7"),
                      xaxis=dict(gridcolor="#EEF2F7"), separators=",.", showlegend=False, hovermode="x unified")
    st.plotly_chart(fig, width="stretch", key=f"chart_{key}")
    st.caption("Línea azul: precio de consola en tus instantáneas. Líneas punteadas: niveles del plan. "
               "▼ y líneas discontinuas: eventos del calendario (pasa el ratón para verlos). Verticales verdes/rojas: tus "
               "compras y ventas (precio PC).")


def _calibration_text(evaluated: pd.DataFrame, kind: str) -> str:
    sub = evaluated[evaluated["kind"] == kind] if not evaluated.empty else evaluated
    n = len(sub)
    if n < MIN_CALIBRATION:
        return f"Acierto real: sin calibrar ({n}/{MIN_CALIBRATION} señales evaluadas)."
    return f"Acierto real: {sub['hit'].mean() * 100:.0f}% en {n} señales evaluadas."


# ---------------------------------------------------------------------------
# Resumen de hoy
# ---------------------------------------------------------------------------


def render_since_last_visit(conn, snap: fc27_market.MarketSnapshot, alerts: list, previous: datetime | None) -> None:
    """Resumen de lo que pasó mientras no estabas: SBC nuevos, señales nuevas y avisos de precio."""
    if previous is None:
        return
    new_sbcs = []
    if not snap.sbcs.empty:
        created = pd.to_datetime(snap.sbcs["created_at"], utc=True, errors="coerce")
        new_sbcs = list(snap.sbcs.loc[created >= previous, "name"])
    new_sig = fc27_history.signals_since(conn, previous)
    buys = list(new_sig.loc[new_sig["kind"] == "COMPRAR", "name"]) if not new_sig.empty else []
    risks = list(new_sig.loc[new_sig["kind"] == "RIESGO", "name"]) if not new_sig.empty else []
    price_hits = [a.title for a in alerts if a.category == "precio"]
    when = _age_text(previous)
    parts = []
    if new_sbcs:
        parts.append(f"<b>{len(new_sbcs)}</b> SBC nuevos")
    if buys:
        parts.append(f"<b>{len(buys)}</b> nuevas compras")
    if risks:
        parts.append(f"<b>{len(risks)}</b> nuevos riesgos")
    if price_hits:
        parts.append(f"<b>{len(price_hits)}</b> avisos de precio")
    if not parts:
        st.markdown(f"<div class='fc-since'>🆕 Desde tu última visita ({when}): sin novedades importantes.</div>",
                    unsafe_allow_html=True)
        return
    st.markdown(f"<div class='fc-since'>🆕 <b>Desde tu última visita</b> ({when}): {' · '.join(parts)}</div>",
                unsafe_allow_html=True)
    with st.expander("Ver novedades"):
        for title, items in (("SBC nuevos", new_sbcs), ("Nuevas señales de compra", buys),
                             ("Nuevas señales de riesgo", risks), ("Avisos de precio", price_hits)):
            if items:
                st.markdown(f"**{title}:** " + ", ".join(html.escape(str(i)) for i in items[:15])
                            + (f" y {len(items) - 15} más" if len(items) > 15 else ""))


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


def _ovr_badge(row) -> str:
    if _is_fodder(row):
        return f"<div class='fc-ovr fodder'><span class='r'>{row['overall']}</span><span class='t'>Fodder</span></div>"
    rarity = str(row["rarity"] or "")
    short = {"Team of the week": "TOTW", "Destined for Glory": "DFG", "Base Icon": "Icon", "Base Hero": "Hero"}
    special = rarity not in ("", "Oro rara (fodder)")
    tag = short.get(rarity, rarity[:6])
    return (f"<div class='fc-ovr {'special' if special else ''}'><span class='r'>{row['overall']}</span>"
            f"<span class='t'>{html.escape(tag)}</span></div>")


def _meter(value: int, color: str) -> str:
    return f"<div class='fc-meter'><i style='width:{max(0, min(100, int(value)))}%;background:{color}'></i></div>"


@st.dialog("Ficha de la carta", width="large")
def card_dialog(conn, row: pd.Series, followed: set[int], analyst: dict) -> None:
    """Ficha completa en una ventana emergente: cifras, gráfico, plan, motivos y desglose."""
    kind = row["signal"] if row["signal"] in PILL else None
    pill = (f"<span class='fc-pill {PILL[kind][0]}'>{PILL[kind][1]}</span>" if kind
            else "<span class='fc-chip'>Sin señal</span>")
    st.markdown(f"<div class='fc-sig-top'>{_ovr_badge(row)}<div class='fc-sig-id'>"
                f"<div class='fc-sig-name' style='font-size:1.2rem'>{html.escape(_card_title(row))}</div>"
                f"<div class='fc-sig-meta'>{html.escape(str(row['rarity']))} · {row['label']} · horizonte "
                f"{row['horizon']}</div></div>{pill}</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio (consola)", _fmt(row["price"]))
    m2.metric("24h", "—" if pd.isna(row["pct_24h"]) else f"{row['pct_24h']:+.1f}%")
    m3.metric("Market Score", int(row["market_score"]))
    m4.metric("Riesgo", int(row["risk_score"]))
    if not _is_fodder(row) and not pd.isna(row["ea_id"]):
        _price_chart(conn, int(row["ea_id"]), key=f"dlg_{row['key']}", row=row, analyst=analyst, height=300)
    plan = fc27_signals.trade_plan(row)
    st.markdown(f"<div class='fc-plan'><span>Entrada:</span> {html.escape(plan['zona'])}<br>"
                f"<span>Objetivo:</span> {html.escape(plan['objetivo'])}<br>"
                f"<span>Invalidación:</span> {html.escape(plan['invalidacion'])}</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Por qué**")
        for r in row["reasons"]:
            st.markdown(f"✅ {r}")
        if not row["reasons"]:
            st.caption("Sin motivos destacados.")
        st.markdown("**Riesgos**")
        for r in row["risks"]:
            st.markdown(f"⚠️ {r}")
        if not row["risks"]:
            st.caption("Sin riesgos destacados.")
    with right:
        bd = pd.DataFrame([(label, int(row[col]), mx) for col, label, mx in fc27_signals.SCORE_COMPONENTS],
                          columns=["Componente", "Puntos", "Máximo"])
        st.dataframe(bd, hide_index=True, width="stretch",
                     column_config={"Puntos": st.column_config.ProgressColumn("Puntos", min_value=0, max_value=20,
                                                                              format="%d")})
    b1, b2 = st.columns(2)
    with b1:
        _follow_button(conn, row, followed, key=f"dlg_follow_{row['key']}")
    if row.get("url"):
        b2.link_button("Abrir en FUT.GG ↗", f"{FUTGG}{row['url']}", width="stretch")


def _signal_card(conn, row: pd.Series, followed: set[int], kind: str, series: dict[int, list[int]],
                 analyst: dict) -> None:
    with st.container(border=True):
        css, label = PILL[kind]
        meta = "Mejor fodder para SBC" if _is_fodder(row) else html.escape(str(row["rarity"]))
        name = html.escape(str(row["name"]) if _is_fodder(row) else str(row["name"]))
        plan = fc27_signals.trade_plan(row)
        pts = series.get(int(row["ea_id"]), []) if not pd.isna(row["ea_id"]) else []
        spark = ui_theme.sparkline_svg(pts, width=150, height=30)
        spark_html = (f"<div class='fc-spark'><span>Tu historial ({len(pts)} puntos)</span>{spark}</div>" if spark
                      else "<div class='fc-spark'><span>Sin historial todavía</span></div>")
        ms, rk = int(row["market_score"]), int(row["risk_score"])
        st.markdown(
            f"""<div class='fc-sig'>
              <div class='fc-sig-top'>{_ovr_badge(row)}
                <div class='fc-sig-id'><div class='fc-sig-name' title='{name}'>{name}</div>
                  <div class='fc-sig-meta'>{meta} · {row['horizon']}</div></div>
                <span class='fc-pill {css}'>{label}</span></div>
              <div class='fc-stats'>
                <div class='fc-stat'><div class='k'>Precio</div><div class='v' title='{_fmt(row['price'])} monedas (consola)'>{_fmt_short(row['price'])}</div></div>
                <div class='fc-stat'><div class='k'>24h</div><div class='v'>{_pct_html(row['pct_24h'])}</div></div>
                <div class='fc-stat'><div class='k'>Market</div><div class='v'>{ms}</div>{_meter(ms, ui_theme.BLUE)}</div>
                <div class='fc-stat'><div class='k'>Riesgo</div><div class='v'>{rk}</div>{_meter(rk, '#DC2626' if rk >= 70 else '#F59E0B' if rk >= 45 else '#16A34A')}</div>
              </div>
              {spark_html}
              <div class='fc-planline'><span>Entrada</span> {html.escape(plan['zona'])}<br>
                <span>Objetivo</span> {html.escape(plan['objetivo'])} · <span>Stop</span> {html.escape(plan['invalidacion'])}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([3, 2]) if not _is_fodder(row) else (st.container(), None)
        with c1:
            if st.button("🔍 Ver ficha", key=f"card_{kind}_{row['key']}", width="stretch"):
                card_dialog(conn, row, followed, analyst)
        if c2 is not None:
            with c2:
                _follow_button(conn, row, followed, key=f"follow_{kind}_{row['key']}")


def render_signals(conn, signals: pd.DataFrame, evaluated: pd.DataFrame, analyst: dict, now: datetime,
                   followed: set[int]) -> None:
    tops = {k: fc27_signals.top_by_signal(signals, k) for k in ("COMPRAR", "VIGILAR", "RIESGO")}
    ids = pd.concat(list(tops.values()))["ea_id"] if any(not v.empty for v in tops.values()) else []
    series = fc27_history.price_series(conn, ids, now - timedelta(days=3))
    cols = st.columns(3)
    for col, kind in zip(cols, ["COMPRAR", "VIGILAR", "RIESGO"]):
        top = tops[kind]
        css, label = PILL[kind]
        with col:
            st.markdown(f"<span class='fc-pill {css}'>{label}</span> <span style='color:#94A3B8'>"
                        f"top {len(top)} de {int((signals['signal'] == kind).sum())}</span>", unsafe_allow_html=True)
            if kind != "VIGILAR":
                st.caption(_calibration_text(evaluated, kind))
            else:
                st.caption("Cartas con algo de interés que aún no cumplen todas las condiciones.")
            if top.empty:
                st.markdown("<div class='fc-empty'>Ninguna carta de tu presupuesto cumple las condiciones ahora "
                            "mismo.</div>", unsafe_allow_html=True)
            for _, row in top.iterrows():
                _signal_card(conn, row, followed, kind, series, analyst)

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


def render_watchlist(conn, signals: pd.DataFrame, analyst: dict) -> None:
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

    trend = fc27_history.price_series(conn, wl["ea_id"], datetime.now(timezone.utc) - timedelta(days=7))
    wl["trend"] = wl["ea_id"].map(lambda i: trend.get(int(i)) if len(trend.get(int(i), [])) >= 2 else None)
    st.dataframe(
        wl[["name", "overall", "buy_price_pc", "price_pc", "pc_net_now", "pc_net_now_pct", "pc_break_even",
            "pc_target", "pc_stop", "trend", "signal"]],
        hide_index=True, width="stretch",
        column_config={
            "name": "Carta", "overall": st.column_config.NumberColumn("OVR", format="%d"),
            "trend": st.column_config.LineChartColumn("Tendencia", width="small", help="Precio de consola en tus instantáneas de los últimos 7 días"),
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

    st.markdown("##### 🔔 Avisos de precio")
    st.caption("Te avisa arriba de la página cuando el precio de consola de FUT.GG cruza tu límite. Se comprueba "
               "con cada instantánea; déjalo vacío para no avisar.")
    alert_labels = {f"{r['name']} {'' if pd.isna(r['overall']) else int(r['overall'])}".strip(): r
                    for _, r in wl.iterrows()}
    with st.form("price_alerts_form", border=True):
        a1, a2, a3, a4 = st.columns([2.2, 1.3, 1.3, 1])
        pick = a1.selectbox("Carta", list(alert_labels), key="alert_card")
        r = alert_labels[pick]
        below = a2.number_input("Avisar si baja de", min_value=0, step=500, key=f"al_below_{r['ea_id']}",
                                value=None if pd.isna(r["alert_below"]) else int(r["alert_below"]),
                                placeholder=f"Ahora {_fmt_short(r['last_price'])}")
        above = a3.number_input("Avisar si sube de", min_value=0, step=500, key=f"al_above_{r['ea_id']}",
                                value=None if pd.isna(r["alert_above"]) else int(r["alert_above"]),
                                placeholder=f"Ahora {_fmt_short(r['last_price'])}")
        a4.write("")
        if a4.form_submit_button("Guardar aviso", type="primary", width="stretch"):
            fc27_history.set_price_alerts(conn, int(r["ea_id"]), below or None, above or None)
            st.toast(f"Aviso guardado para {pick}", icon="🔔")
            st.rerun()
    active = wl[wl["alert_below"].notna() | wl["alert_above"].notna()]
    if not active.empty:
        st.dataframe(active[["name", "last_price", "alert_below", "alert_above"]], hide_index=True, width="stretch",
                     column_config={"name": "Carta",
                                    "last_price": st.column_config.NumberColumn("Último (consola)", format="localized"),
                                    "alert_below": st.column_config.NumberColumn("Baja de", format="localized"),
                                    "alert_above": st.column_config.NumberColumn("Sube de", format="localized")})

    names = {f"{r['name']} {'' if pd.isna(r['overall']) else int(r['overall'])}".strip(): int(r["ea_id"])
             for _, r in wl.iterrows()}
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Ver gráfico (precio de consola) de", list(names), key="wl_chart")
    c2.write("")
    if c2.button("Quitar de Mi lista", key="wl_remove", width="stretch"):
        fc27_history.remove_from_watchlist(conn, names[choice])
        st.rerun()
    _price_chart(conn, names[choice], key="wl", analyst=analyst,
                 row=signals[signals["ea_id"] == names[choice]].iloc[0] if (signals["ea_id"] == names[choice]).any() else None)


OTHER_CARD = "✏️ Otra carta (escribir nombre)"


def render_equity_curve(df: pd.DataFrame) -> None:
    """Beneficio neto acumulado (PC) a lo largo de tus ventas, con la caída desde el máximo."""
    curve = fc27_history.equity_curve(df)
    if curve.empty:
        return
    # Punto de partida en 0 el día de la primera compra: la curva se lee aunque haya una sola venta.
    first_buy = pd.to_datetime(df.loc[df["status"] == "Cerrada", "buy_at"], utc=True).min()
    origin = pd.DataFrame([{"sell_at": min(first_buy, curve["sell_at"].min()), "card_name": "Inicio",
                            "net_profit": 0, "cumulative": 0, "peak": 0, "drawdown": 0}])
    curve = pd.concat([origin, curve], ignore_index=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["sell_at"], y=curve["cumulative"], mode="lines+markers", name="Beneficio acumulado",
        line=dict(color=ui_theme.BLUE, width=2, shape="hv"), marker=dict(size=7),
        customdata=curve[["card_name", "net_profit"]],
        hovertemplate="%{x|%d %b %H:%M}<br>%{customdata[0]}: %{customdata[1]:+,.0f}<br>"
                      "Acumulado %{y:+,.0f}<extra></extra>",
    ))
    if (curve["drawdown"] < 0).any():
        fig.add_trace(go.Scatter(
            x=curve["sell_at"], y=curve["drawdown"], mode="lines", name="Caída desde el máximo",
            line=dict(color="#DC2626", width=1, shape="hv"), fill="tozeroy", fillcolor="rgba(220,38,38,0.10)",
            hovertemplate="%{x|%d %b}<br>Caída %{y:,.0f}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color="#94A3B8", width=1))
    fig.update_layout(template="plotly_white", height=260, margin=dict(t=10, b=30, l=60, r=10), separators=",.",
                      yaxis=dict(title="Monedas (PC, neto)", gridcolor="#EEF2F7"), xaxis=dict(gridcolor="#EEF2F7"),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    st.markdown("##### 💰 Curva de beneficio")
    st.plotly_chart(fig, width="stretch", key="equity_curve")


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
    render_equity_curve(df)

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


HEAT_SCALE = [(0.0, "#B91C1C"), (0.35, "#FCA5A5"), (0.5, "#E5E7EB"), (0.65, "#86EFAC"), (1.0, "#15803D")]
HEAT_RANGE = 15  # ±15% satura el color: así los movimientos pequeños siguen viéndose


def render_heatmap(cards: pd.DataFrame) -> None:
    """Mapa de calor del mercado: bloques por rareza; tamaño según precio (escala
    logarítmica, para que los Icons millonarios no tapen al resto) y color según
    la variación de 24h (rojo baja, gris plano, verde sube)."""
    import numpy as np

    data = cards.dropna(subset=["pct_24h"]).copy()
    if data.empty:
        return
    data["size"] = np.log10(data["price"].clip(lower=1000)) - 2.5
    data["label"] = data["name"] + " " + data["overall"].astype(str)
    data["rarity_label"] = data["rarity"].replace({"Team of the week": "TOTW", "Destined for Glory": "DFG",
                                                    "Base Icon": "Icons", "Base Hero": "Heroes"})
    data["price_txt"] = data["price"].map(_fmt)
    import plotly.express as px

    fig = px.treemap(
        data, path=[px.Constant("Mercado"), "rarity_label", "label"], values="size", color="pct_24h",
        color_continuous_scale=HEAT_SCALE, range_color=(-HEAT_RANGE, HEAT_RANGE), color_continuous_midpoint=0,
        custom_data=["price_txt", "pct_24h", "signal"],
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[1]:+.1f}%", textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{customdata[0]} monedas (consola)<br>24h: %{customdata[1]:+.1f}%"
                      "<br>Señal: %{customdata[2]}<extra></extra>",
        marker=dict(line=dict(width=1, color="#FFFFFF")), root_color="#F8FAFC",
    )
    fig.update_layout(height=430, margin=dict(t=10, b=10, l=0, r=0), separators=",.",
                      coloraxis_colorbar=dict(title="24h %", ticksuffix="%", len=0.7, thickness=12))
    st.markdown("##### Mapa del mercado (24h)")
    st.plotly_chart(fig, width="stretch", key="heatmap")
    st.caption(f"Cada bloque es una carta, agrupada por rareza. Tamaño: precio (escala logarítmica). Color: variación "
               f"de 24h (el color satura en ±{HEAT_RANGE}%). Haz clic en una rareza para ampliarla; clic en el "
               "título para volver.")


def render_market(conn, signals: pd.DataFrame, followed: set[int], analyst: dict) -> None:
    cards = signals[~signals["key"].str.startswith("fodder")].copy()
    if cards.empty:
        st.info("No hay cartas en tu presupuesto. Cambia el presupuesto en la barra lateral.")
        return
    render_heatmap(cards)
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
    trend = fc27_history.price_series(conn, view["ea_id"], datetime.now(timezone.utc) - timedelta(days=3))
    view["trend"] = view["ea_id"].map(lambda i: trend.get(int(i)) if len(trend.get(int(i), [])) >= 2 else None)
    event = st.dataframe(
        view[["followed", "name", "overall", "rarity", "price", "trend", *pct_cols, "market_score", "risk_score",
              "signal"]],
        hide_index=True, width="stretch", height=min(480, 38 + 35 * max(len(view), 1)), on_select="rerun", selection_mode="single-row", key="mv_table",
        column_config={
            "followed": st.column_config.TextColumn("", help="⭐ = está en Mi lista", width="small"),
            "name": "Carta", "overall": st.column_config.NumberColumn("OVR", format="%d"), "rarity": "Rareza",
            "price": st.column_config.NumberColumn("Precio consola", format="localized"),
            "trend": st.column_config.LineChartColumn("Tendencia", width="small",
                                                      help="Precio de consola en tus instantáneas de los últimos 3 días"),
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
            _price_chart(conn, int(row["ea_id"]), key="market", row=row, analyst=analyst)


FODDER_COLORS = {84: "#2A78D6", 85: "#EB6834", 86: "#1BAF7A"}


def render_fodder_index(conn) -> None:
    """Evolución del precio de referencia del fodder 84/85/86 en tus instantáneas."""
    idx = fc27_history.fodder_index(conn, ratings=tuple(FODDER_COLORS))
    st.markdown("##### Índice de fodder (84 · 85 · 86)")
    if idx.empty or idx["fetched_at"].nunique() < 2:
        st.caption("Se necesitan al menos dos instantáneas para dibujar el índice.")
        return
    rng = st.segmented_control("Rango del índice", list(RANGES), default="Todo", key="rng_fodder",
                               label_visibility="collapsed") or "Todo"
    if RANGES[rng] is not None:
        idx = idx[idx["fetched_at"] >= idx["fetched_at"].max() - RANGES[rng]]
    fig = go.Figure()
    summary = []
    for ovr, color in FODDER_COLORS.items():
        s = idx[idx["overall"] == ovr]
        if s.empty:
            continue
        first, last = int(s["price"].iloc[0]), int(s["price"].iloc[-1])
        change = (last / first - 1) * 100 if first else 0
        summary.append(f"**{ovr}**: {_fmt(first)} → {_fmt(last)} ({change:+.1f}%)")
        fig.add_trace(go.Scatter(
            x=s["fetched_at"], y=s["price"] / first * 100, mode="lines+markers", name=f"Rating {ovr}",
            line=dict(color=color, width=2), marker=dict(size=5), customdata=s["price"],
            hovertemplate=f"Rating {ovr}<br>%{{x|%a %d %b %H:%M}} UTC<br>%{{customdata:,}} monedas"
                          "<br>Índice %{y:.0f}<extra></extra>",
        ))
        fig.add_annotation(x=s["fetched_at"].iloc[-1], y=last / first * 100, text=f"{ovr}", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=color, size=12))
    fig.add_hline(y=100, line=dict(color="#94A3B8", width=1, dash="dot"))
    fig.update_layout(template="plotly_white", height=300, margin=dict(t=10, b=30, l=50, r=30), separators=",.",
                      yaxis=dict(title="Índice (inicio = 100)", gridcolor="#EEF2F7"), xaxis=dict(gridcolor="#EEF2F7"),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    st.plotly_chart(fig, width="stretch", key="fodder_index")
    st.markdown(" · ".join(summary))
    st.caption("Precio de referencia (mediana de las 5 más baratas) de cada rating, normalizado a 100 al inicio del "
               "rango para comparar ratings con precios distintos. Si sube, crece la demanda de fodder para SBC.")


def render_fodder(snap: fc27_market.MarketSnapshot, conn) -> None:
    fodder = fc27_market.fodder_table(snap.cheapest)
    st.caption("Los SBC piden puntos de Item Score. Cuanto menos cueste cada punto, mejor fodder. "
               "El precio de referencia es la mediana de las 5 cartas más baratas de cada rating.")
    render_fodder_index(conn)
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
        st.caption("Precios de cartas: consola (FUT.GG). Tus precios de PC: ⭐ Mi lista y 📒 Operaciones.")
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
        alerts = fc27_signals.price_alert_hits(fc27_history.watchlist(conn)) + alerts
        if "fc_prev_visit" not in st.session_state:  # una vez por sesión del navegador
            st.session_state["fc_prev_visit"] = fc27_history.start_visit(conn, datetime.now(timezone.utc))

        is_fodder = all_signals["key"].str.startswith("fodder")
        signals = pd.concat([all_signals[is_fodder], _in_budget(all_signals[~is_fodder], budget)])

        live_badge = ("<span class='fc-badge live'>● En vivo</span>" if live
                      else "<span class='fc-badge off'>● Sin conexión</span>")
        st.markdown(
            "<div class='fc-title'>Mercado FC 27</div>"
            "<div class='fc-sub' style='margin-bottom:0'>Qué comprar, qué vigilar y qué vender hoy en Ultimate Team</div>"
            f"<div class='fc-badges'>{live_badge}"
            f"<span class='fc-badge'>🕒 {fc27_signals.format_when(now)} · {_age_text(now)}</span>"
            f"<span class='fc-badge'>🗂 {fc27_history.snapshot_count(conn)} instantáneas</span>"
            "<span class='fc-badge'>💻 PC · cartas a precio de consola</span></div>",
            unsafe_allow_html=True)
        if not live:
            st.warning(f"FUT.GG no respondió. Estás viendo los datos guardados {_age_text(now)}.")
        elif snap.errors:
            st.warning("Algunas páginas de FUT.GG fallaron; los datos pueden estar incompletos.")

        render_since_last_visit(conn, snap, alerts, st.session_state.get("fc_prev_visit"))
        render_today(signals, events, now)

        counts = signals["signal"].value_counts()
        st.markdown(
            f"<div class='fc-strip'><span>Mercado 24h: <b>{fc27_signals.overall_tone(snap.movers)}</b></span>"
            f"<span>Señales en tu presupuesto: <b>{counts.get('COMPRAR', 0)}</b> comprar · "
            f"<b>{counts.get('VIGILAR', 0)}</b> vigilar · <b>{counts.get('RIESGO', 0)}</b> riesgo</span></div>",
            unsafe_allow_html=True,
        )
        for a in fc27_signals.headline_alerts(alerts, followed)[:3]:
            cls, icon = ("crit", "🚨") if a.level == "crítica" else ("warn", "⚠️")
            if a.category == "precio":
                icon = "🔔"
            st.markdown(f"<div class='fc-banner {cls}'><span>{icon}</span><span><b>{html.escape(a.title)}</b> · "
                        f"{html.escape(a.detail)}</span></div>", unsafe_allow_html=True)

        # Etiquetas fijas (sin contadores): si cambian, Streamlit vuelve a la primera pestaña
        # después de guardar algo, y el usuario pierde dónde estaba.
        tabs = st.tabs(["🎯 Señales", "⭐ Mi lista", "📒 Operaciones", "📊 Mercado",
                        "🧱 Fodder y SBC", "🚨 Alertas", "📘 Cómo funciona"])
        with tabs[0]:
            render_signals(conn, signals, evaluated, analyst, now, followed)
        with tabs[1]:
            render_watchlist(conn, all_signals, analyst)
        with tabs[2]:
            render_trades(conn, all_signals)
        with tabs[3]:
            render_market(conn, signals, followed, analyst)
        with tabs[4]:
            render_fodder(snap, conn)
        with tabs[5]:
            render_alerts(alerts, analyst, now)
        with tabs[6]:
            render_help(evaluated, snap)

    body()


main()
