"""Mercado FC 27 — inteligencia de mercado para EA SPORTS FC 27 Ultimate Team.

Página registrada en `app.py`. Al abrirla descarga datos en vivo de FUT.GG
(con caché de unos minutos), guarda una instantánea en el historial local y
calcula Market Score, Risk Score, señales y alertas.

Orden de la página (de lo más urgente a lo más detallado):
1. Cabecera: mercado en vivo (o datos guardados) y última actualización.
2. Resumen de hoy: mejor compra, mayor riesgo y próximo evento.
3. Barra de estado del mercado y alertas importantes, agrupadas.
4. Secciones: Señales, Mercado (mapa, tabla y ficha de la carta), Mi lista,
   Operaciones y, dentro de "Más", Fodder & SBC, Alertas y Cómo funciona.
   Solo se dibuja la sección abierta.

Este archivo solo presenta los datos: las fórmulas y reglas viven en
`src/fc27_*`. Si FUT.GG no responde, muestra la última instantánea guardada y
avisa de su antigüedad.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import fc27_history, fc27_market, fc27_signals, ui_theme
from src.ui_theme import TOKENS

CACHE_TTL_S = 570          # algo menos que el auto-refresco, para que cada refresco traiga datos nuevos
AUTO_REFRESH_S = 600
MIN_CALIBRATION = 20       # señales evaluadas necesarias para mostrar una tasa de acierto
SIGNAL_PAGE = 9            # tarjetas de señales por tanda (3 filas de 3 en escritorio)
FUTGG = fc27_market.BASE_URL

BUDGETS = {
    "Cualquier presupuesto": (0, float("inf")),
    "Bajo (hasta 20k)": (0, 20_000),
    "Medio (20k-100k)": (20_000, 100_000),
    "Alto (100k-500k)": (100_000, 500_000),
    "Premium (500k+)": (500_000, float("inf")),
}
KINDS = ("COMPRAR", "VIGILAR", "RIESGO")
# Clase CSS, flecha y texto de cada señal.
SIGNAL_UI = {"COMPRAR": ("buy", "▲", "Comprar"), "VIGILAR": ("watch", "●", "Vigilar"),
             "RIESGO": ("risk", "▼", "Riesgo")}
# Nombre de cada paso del plan según la señal (los textos los da fc27_signals.trade_plan).
PLAN_LABELS = {"COMPRAR": ("Entrada", "Objetivo", "Stop"), "VIGILAR": ("Entrada", "Objetivo", "Invalidación"),
               "RIESGO": ("Acción", "Escenario", "Invalidación")}
RARITY_SHORT = {"Team of the week": "TOTW", "Destined for Glory": "DFG", "Base Icon": "Icon", "Base Hero": "Hero"}
LEVEL_CLASS = {"crítica": "crit", "aviso": "warn", "info": "info"}

# Navegación: cuatro secciones principales y el resto dentro de "Más".
NAV = {"signals": ":material/insights: Señales", "market": ":material/monitoring: Mercado",
       "watchlist": ":material/star: Mi lista", "trades": ":material/receipt_long: Operaciones",
       "more": ":material/more_horiz: Más"}
NAV_MORE = {"fodder": ":material/layers: Fodder & SBC", "alerts": ":material/notifications: Alertas",
            "help": ":material/info: Cómo funciona"}

# Mapa de calor: rojo (baja) → gris (plano) → verde (sube); ±15% satura el color, así los
# movimientos pequeños siguen viéndose.
HEAT_SCALE = [(0.0, "#D2504A"), (0.33, "#F2BDB6"), (0.5, "#E8ECF1"), (0.67, "#B0DEC1"), (1.0, "#2F9A5C")]
HEAT_RANGE = 15
TOP_OPTIONS = {"Top 50": 50, "Top 100": 100, "Todas": None}

PAGE_CSS = """
<style>
/* ===== Lienzo y bloques ===== */
/* Streamlit resta 1rem bajo cada bloque de markdown; los bloques propios lo devuelven (margin-bottom: 1rem). */
[data-testid="stAppViewContainer"] { background: var(--fc-bg); }
[data-testid="stMainBlockContainer"] div[data-testid="stMetric"] { min-height: 108px; }
[data-testid="stMainBlockContainer"] [data-testid="stExpander"] details {
  background: var(--fc-surface); border-color: var(--fc-border); border-radius: var(--fc-radius-sm); }
[data-testid="stMainBlockContainer"] [data-testid="stForm"] {
  background: var(--fc-surface); border-color: var(--fc-border); border-radius: var(--fc-radius); box-shadow: var(--fc-shadow); }
[class*="st-key-fc_box_"] {
  background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius);
  box-shadow: var(--fc-shadow); padding: 18px 20px 14px; }
@media (max-width: 640px) { [class*="st-key-fc_box_"] { padding: 14px 12px 10px; } }

/* ===== Tonos semánticos =====
   Cada modificador fija su color (--tone) y sus variantes; pills, avisos, puntos, contadores, barras y
   tarjetas los usan. Verde = comprar/bien, amarillo = vigilar/precaución, rojo = riesgo, azul = información. */
.buy, .ok { --tone: var(--fc-green); --tone-soft: var(--fc-green-soft); --tone-ink: var(--fc-green-ink); }
.watch, .warn { --tone: var(--fc-yellow); --tone-soft: var(--fc-yellow-soft); --tone-ink: var(--fc-yellow-ink); }
.risk, .crit { --tone: var(--fc-red); --tone-soft: var(--fc-red-soft); --tone-ink: var(--fc-red-ink); }
.info, .event { --tone: var(--fc-blue); --tone-soft: var(--fc-blue-soft); --tone-ink: var(--fc-blue-ink); }

/* ===== Primitivas ===== */
.fc-label { font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--fc-faint); }
.fc-up { color: var(--fc-green); font-weight: 600; }
.fc-down { color: var(--fc-red); font-weight: 600; }
.fc-flat { color: var(--fc-faint); }
.fc-pill { display: inline-flex; align-items: center; gap: 4px; font-size: .68rem; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; padding: 3px 9px; border-radius: 999px; white-space: nowrap; background: var(--tone-soft);
  color: var(--tone-ink); border: 1px solid color-mix(in srgb, var(--tone) 25%, transparent); }
.fc-pill.none { background: var(--fc-surface-2); color: var(--fc-muted); border-color: var(--fc-border); }
.fc-chip { display: inline-block; font-size: .64rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
  padding: 2px 7px; border-radius: 6px; background: var(--fc-surface-2); color: var(--fc-muted); border: 1px solid var(--fc-border);
  vertical-align: 1px; }
.fc-meter { height: 6px; border-radius: 999px; background: var(--fc-border); overflow: hidden; }
.fc-meter i { display: block; height: 100%; border-radius: 999px; background: var(--tone); }
.fc-empty { border: 1px dashed var(--fc-border-strong); border-radius: var(--fc-radius); padding: 18px 20px; color: var(--fc-muted);
  font-size: .9rem; line-height: 1.5; background: var(--fc-surface); margin-bottom: 1rem; }
.fc-section { margin: 2px 0 10px; }
.fc-section-title { font-size: 1.02rem; font-weight: 700; color: var(--fc-text); letter-spacing: -.01em; }
.fc-section-sub { font-size: .82rem; color: var(--fc-muted); margin-top: 2px; line-height: 1.45; }
.fc-note { font-size: .8rem; color: var(--fc-muted); line-height: 1.5; margin-bottom: 1rem; }

/* ===== Cabecera ===== */
.fc-head { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 14px 32px;
  padding-bottom: 20px; margin-bottom: 20px; border-bottom: 1px solid var(--fc-border); }
.fc-title { font-size: clamp(1.75rem, 3.2vw, 2.35rem); font-weight: 800; letter-spacing: -.03em; line-height: 1.05;
  color: var(--fc-text); text-transform: uppercase; }
.fc-title span { color: var(--fc-blue); }
.fc-sub { color: var(--fc-muted); font-size: .95rem; margin-top: 6px; }
.fc-live { display: flex; align-items: center; flex-wrap: wrap; gap: 6px 14px; margin-top: 14px; font-size: .8rem; color: var(--fc-muted); }
.fc-live-tag { display: inline-flex; align-items: center; gap: 7px; font-size: .7rem; font-weight: 800; letter-spacing: .09em;
  text-transform: uppercase; color: var(--fc-green-ink); }
.fc-live-tag i { width: 8px; height: 8px; border-radius: 50%; background: var(--fc-green);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--fc-green) 18%, transparent); }
.fc-live-tag.off { color: var(--fc-yellow-ink); }
.fc-live-tag.off i { background: var(--fc-yellow); box-shadow: 0 0 0 3px color-mix(in srgb, var(--fc-yellow) 18%, transparent); }
.fc-meta { display: flex; flex-wrap: wrap; gap: 6px; }
.fc-meta span { font-size: .74rem; color: var(--fc-muted); background: var(--fc-surface); border: 1px solid var(--fc-border);
  border-radius: 999px; padding: 3px 10px; white-space: nowrap; }

/* ===== Resumen de hoy: tres tarjetas de igual altura (una sola rejilla) ===== */
.fc-cq { container-type: inline-size; }
.fc-heroes { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
@container (max-width: 760px) { .fc-heroes { grid-template-columns: 1fr; } }
.fc-hero { position: relative; display: flex; flex-direction: column; min-width: 0; overflow: hidden;
  background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius); box-shadow: var(--fc-shadow);
  padding: 16px 18px 14px; transition: border-color var(--fc-ease), box-shadow var(--fc-ease), transform var(--fc-ease); }
.fc-hero::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 3px; background: var(--tone); }
.fc-hero:hover { border-color: var(--fc-border-strong); box-shadow: var(--fc-shadow-hover); transform: translateY(-1px); }
.fc-hero-k { font-size: .68rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; color: var(--tone); }
.fc-hero-body { flex: 1 1 auto; display: flex; flex-direction: column; gap: 10px; margin-top: 12px; min-width: 0; }
.fc-hero-id { display: flex; align-items: center; gap: 12px; min-width: 0; }
.fc-hero-n { font-size: 1.02rem; font-weight: 700; color: var(--fc-text); line-height: 1.25; display: -webkit-box;
  -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.fc-hero-m { font-size: .78rem; color: var(--fc-muted); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fc-hero-v { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 10px; font-size: 2rem; font-weight: 800;
  letter-spacing: -.03em; line-height: 1; color: var(--fc-text); font-variant-numeric: tabular-nums; }
.fc-hero-v small { font-size: .9rem; font-weight: 700; letter-spacing: 0; }
.fc-hero-why { font-size: .8rem; color: var(--fc-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fc-hero-foot { display: grid; gap: 7px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--fc-border); }
.fc-hero-row { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; font-size: .82rem; color: var(--fc-muted); }
.fc-hero-row b { color: var(--fc-text); font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }

/* ===== Barra de estado del mercado ===== */
.fc-strip { display: flex; align-items: center; flex-wrap: wrap; gap: 8px 18px; margin: 16px 0 0; padding: 10px 16px;
  background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius-sm); box-shadow: var(--fc-shadow);
  font-size: .84rem; color: var(--fc-muted); }
.fc-strip-item { display: inline-flex; align-items: center; flex-wrap: wrap; gap: 6px 10px; }
.fc-strip-sep { width: 1px; height: 18px; background: var(--fc-border); }
.fc-tone { display: inline-flex; align-items: center; gap: 6px; font-size: .8rem; font-weight: 800; letter-spacing: .05em;
  text-transform: uppercase; color: var(--fc-text); }
.fc-tone i { width: 8px; height: 8px; border-radius: 50%; background: var(--fc-faint); }
.fc-tone.up i { background: var(--fc-green); }
.fc-tone.down i { background: var(--fc-red); }
.fc-count { display: inline-flex; align-items: baseline; gap: 5px; white-space: nowrap; }
.fc-count b { font-size: .98rem; font-weight: 800; color: var(--fc-text); font-variant-numeric: tabular-nums; }
.fc-count i { font-style: normal; font-size: .72rem; color: var(--tone); }
@media (max-width: 640px) { .fc-strip-sep { display: none; } }

/* ===== Avisos (un aviso suelto o varios agrupados en un desplegable) ===== */
.fc-banner { display: flex; align-items: flex-start; gap: 12px; margin: 10px 0 0; padding: 11px 16px;
  border-radius: var(--fc-radius-sm); font-size: .87rem; line-height: 1.45; background: var(--tone-soft); color: var(--tone-ink);
  border: 1px solid color-mix(in srgb, var(--tone) 25%, transparent); }
.fc-ico { flex: 0 0 auto; width: 20px; height: 20px; border-radius: 50%; display: inline-flex; align-items: center;
  justify-content: center; font-size: .72rem; font-weight: 800; color: #FFFFFF; background: var(--tone); }
details.fc-banner { display: block; padding: 0; }
details.fc-banner summary { display: flex; align-items: center; gap: 12px; padding: 10px 16px; cursor: pointer; list-style: none; }
details.fc-banner summary::-webkit-details-marker { display: none; }
details.fc-banner .more { margin-left: auto; padding-left: 12px; font-size: .8rem; font-weight: 700; white-space: nowrap; }
details.fc-banner .more::after { content: " ▾"; }
details.fc-banner[open] .more::after { content: " ▴"; }
details.fc-banner ul { list-style: none; margin: 0; padding: 0 16px 10px 48px; }
details.fc-banner li { padding: 7px 0; border-top: 1px solid color-mix(in srgb, currentColor 14%, transparent); }
details.fc-banner li span { display: block; font-size: .8rem; opacity: .85; }

/* ===== Navegación ===== */
.st-key-fc_nav { margin-top: 22px; gap: 8px; container-type: inline-size; }
.st-key-fc_nav button p { font-weight: 600; }
.st-key-fc_nav_more { margin-top: -2px; }
@container (max-width: 640px) {  /* si la barra es estrecha, las cinco secciones caben en una fila: sin iconos */
  .st-key-fc_nav button span:has(> [data-testid="stIconMaterial"]) { display: none; }
  .st-key-fc_nav button { padding-left: 4px; padding-right: 4px; }
  .st-key-fc_nav button p { font-size: .78rem; }
}

/* ===== Tarjetas de señal (rejilla flexible: 3, 2 o 1 por fila según el ancho) ===== */
.st-key-fc_grid { display: grid !important; grid-template-columns: repeat(auto-fill, minmax(min(100%, 280px), 1fr));
  gap: 16px; align-items: start; }
[class*="st-key-fc_card_"] { background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius);
  box-shadow: var(--fc-shadow); padding: 16px 16px 4px; gap: 6px;
  transition: border-color var(--fc-ease), box-shadow var(--fc-ease), transform var(--fc-ease); }
[class*="st-key-fc_card_"]:hover { border-color: var(--fc-border-strong); box-shadow: var(--fc-shadow-hover); transform: translateY(-1px); }
[class*="st-key-fc_card_"] [data-testid="stExpander"] details { border: 0; border-top: 1px solid var(--fc-border);
  border-radius: 0; background: transparent; }
[class*="st-key-fc_card_"] [data-testid="stExpander"] summary { padding-left: 2px; padding-right: 2px; }
[class*="st-key-fc_card_"] [data-testid="stExpander"] summary p { font-weight: 600; color: var(--fc-blue); }
[class*="st-key-fc_card_"] [data-testid="stExpanderDetails"] { padding: 2px 2px 12px; }
.fc-sig { display: flex; flex-direction: column; gap: 14px; margin-bottom: 1rem; }
.fc-sig-top { display: flex; align-items: flex-start; gap: 12px; }
.fc-sig-id { flex: 1 1 auto; min-width: 0; padding-top: 2px; }
.fc-sig-name { font-size: 1.02rem; font-weight: 700; line-height: 1.25; color: var(--fc-text); overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.fc-sig-meta { font-size: .78rem; color: var(--fc-muted); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fc-star { color: var(--fc-yellow); font-size: .9rem; margin-left: 4px; }
.fc-sig-price { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
.fc-price { font-size: 1.65rem; font-weight: 800; letter-spacing: -.03em; line-height: 1; color: var(--fc-text);
  font-variant-numeric: tabular-nums; }
.fc-price small { font-size: .7rem; font-weight: 600; letter-spacing: 0; color: var(--fc-faint); margin-left: 6px; }
.fc-chg { font-size: .92rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
.fc-chg small { font-size: .7rem; font-weight: 600; color: var(--fc-faint); margin-left: 3px; }
.fc-scores { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.fc-score .row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.fc-score b { font-size: .98rem; font-weight: 800; color: var(--fc-text); font-variant-numeric: tabular-nums; }
.fc-score b.ok, .fc-score b.warn, .fc-score b.crit { color: var(--tone); }
.fc-strat { font-size: .84rem; line-height: 1.45; color: var(--fc-text); background: var(--fc-surface-2);
  border: 1px solid var(--fc-border); border-radius: 10px; padding: 9px 11px; min-height: 4.9rem; }
.fc-strat b { font-weight: 600; }
.fc-strat span { display: block; font-size: .76rem; color: var(--fc-muted); margin-top: 2px; }
.fc-spark { display: flex; flex-direction: column; gap: 4px; min-height: 52px; }
.fc-spark svg { width: 100%; height: auto; display: block; }
.fc-spark .none { font-size: .78rem; color: var(--fc-faint); padding-top: 6px; }

/* ===== Miniatura de carta (imagen de FUT.GG o insignia con la media) ===== */
.fc-thumb { flex: 0 0 auto; width: 52px; }
.fc-thumb.sm { width: 46px; }
.fc-thumb.lg { width: 104px; }
.fc-thumb img { display: block; width: 100%; height: auto; filter: drop-shadow(0 2px 3px rgba(15, 23, 42, .18)); }
.fc-ovr { width: 100%; aspect-ratio: .72; border-radius: 9px 9px 15px 15px; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 3px; line-height: 1; font-weight: 800; color: #3B2F0B;
  background: linear-gradient(165deg, #F6E7A8 0%, #E3C56B 55%, #C9A443 100%);
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .08), 0 2px 4px rgba(15, 23, 42, .12); }
.fc-ovr .r { font-size: 1.3rem; letter-spacing: -.02em; }
.fc-ovr .t { font-size: .5rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; opacity: .85; }
.fc-ovr.special { background: linear-gradient(165deg, #24324D 0%, #0F172A 100%); color: #F4D77A; }
.fc-ovr.fodder { background: linear-gradient(165deg, #EEF2F6 0%, #D3DAE3 100%); color: #334155; }
.fc-thumb.lg .fc-ovr .r { font-size: 2.3rem; }
.fc-thumb.lg .fc-ovr .t { font-size: .72rem; }

/* ===== Ficha de mercado de una carta ===== */
.fc-sheet { margin-bottom: 1rem; }
.fc-sheet-head { display: flex; align-items: center; gap: 16px; }
.fc-sheet-name { font-size: 1.45rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.15; color: var(--fc-text); }
.fc-sheet-meta { font-size: .82rem; color: var(--fc-muted); margin: 4px 0 8px; }
.fc-kv { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }
.fc-kv > div { background: var(--fc-surface-2); border: 1px solid var(--fc-border); border-radius: 10px; padding: 10px 12px; min-width: 0; }
.fc-kv .v { font-size: 1.25rem; font-weight: 800; letter-spacing: -.02em; color: var(--fc-text); font-variant-numeric: tabular-nums;
  margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fc-sheet .fc-scores { margin-top: 14px; }
.fc-plan { margin-top: 14px; border: 1px solid var(--fc-border); border-radius: 10px; overflow: hidden; background: var(--fc-surface); }
.fc-plan .row { display: flex; justify-content: space-between; align-items: baseline; gap: 14px; padding: 8px 12px; font-size: .84rem; }
.fc-plan .row + .row { border-top: 1px solid var(--fc-border); }
.fc-plan .row span { flex: 0 0 auto; color: var(--fc-muted); }
.fc-plan .row b { font-weight: 600; text-align: right; color: var(--fc-text); }
.fc-whys { margin-bottom: 1rem; }
.fc-why { margin-top: 14px; }
.fc-why ul { list-style: none; margin: 6px 0 0; padding: 0; }
.fc-why li { position: relative; padding: 3px 0 3px 16px; font-size: .84rem; line-height: 1.45; color: var(--fc-text); }
.fc-why li::before { content: ""; position: absolute; left: 1px; top: 10px; width: 7px; height: 7px; border-radius: 50%;
  background: var(--tone); }
.fc-why li.none { padding-left: 0; color: var(--fc-faint); }
.fc-why li.none::before { display: none; }

/* ===== Mapa del mercado ===== */
.fc-legend { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px 28px; margin: 2px 0 8px; }
.fc-legend-scale { flex: 0 1 380px; min-width: 240px; }
.fc-legend-bar { height: 8px; border-radius: 999px; margin-top: 6px; background: linear-gradient(90deg, __HEAT_GRADIENT__); }
.fc-legend-ticks { display: flex; justify-content: space-between; margin-top: 4px; font-size: .68rem; color: var(--fc-faint);
  font-variant-numeric: tabular-nums; }
.fc-legend-keys { display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: .8rem; color: var(--fc-muted); }
.fc-legend-keys b { color: var(--fc-text); font-weight: 600; }

/* ===== Listas (alertas y calendario) ===== */
.fc-list { background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius);
  box-shadow: var(--fc-shadow); overflow: hidden; margin-bottom: 1rem; }
.fc-list-row { display: flex; align-items: flex-start; gap: 12px; padding: 12px 16px; }
.fc-list-row + .fc-list-row { border-top: 1px solid var(--fc-border); }
.fc-dot { flex: 0 0 auto; width: 8px; height: 8px; border-radius: 50%; margin-top: 7px; background: var(--tone, var(--fc-faint)); }
.fc-list-t { font-size: .9rem; font-weight: 600; color: var(--fc-text); line-height: 1.4; }
.fc-list-d { font-size: .8rem; color: var(--fc-muted); margin-top: 2px; line-height: 1.45; }
.fc-when { flex: 0 0 92px; font-size: .8rem; font-weight: 700; color: var(--fc-text); line-height: 1.35; }
.fc-when span { display: block; font-weight: 500; color: var(--fc-muted); }

/* ===== Barra lateral ===== */
.fc-side-label { font-size: .68rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--fc-faint);
  margin: 4px 0 2px; }
.fc-side-foot { font-size: .74rem; color: var(--fc-faint); line-height: 1.6; border-top: 1px solid var(--fc-border);
  padding-top: 12px; margin-top: 6px; }
</style>
"""


def _heat_gradient() -> str:
    """Los colores del mapa de calor como degradado CSS, para la leyenda (mismos que Plotly)."""
    return ", ".join(f"{color} {stop * 100:.0f}%" for stop, color in HEAT_SCALE)


st.markdown(PAGE_CSS.replace("__HEAT_GRADIENT__", _heat_gradient()), unsafe_allow_html=True)


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
    """Precio corto para leer de un vistazo: 1,70M · 995k · 23,5k · 8.300."""
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M".replace(".", ",")
    if n >= 100_000:
        return f"{n / 1000:.0f}k"
    if n >= 10_000:
        return f"{n / 1000:.1f}k".replace(".", ",")
    return _fmt(n)


_LONG_NUMBER = re.compile(r"\d{1,3}(?:\.\d{3})+")


def _short_numbers(text) -> str:
    """Acorta los precios largos de un texto del plan ('hacia 1.445.000' → 'hacia 1,45M')."""
    return _LONG_NUMBER.sub(lambda m: _fmt_short(int(m.group().replace(".", ""))), str(text))


def _pct_txt(p) -> str:
    if p is None or pd.isna(p):
        return "—"
    return f"{p:+.1f}%".replace(".", ",")


def _pct_html(p) -> str:
    if p is None or pd.isna(p):
        return "<span class='fc-flat'>—</span>"
    cls = "fc-up" if p > 0 else ("fc-down" if p < 0 else "fc-flat")
    return f"<span class='{cls}'>{_pct_txt(p)}</span>"


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


def _safe_key(value) -> str:
    """Texto apto para la key de un contenedor (Streamlit la convierte en clase CSS)."""
    return re.sub(r"\W+", "_", str(value))


def _by_card(movers: pd.DataFrame, column: str) -> dict[int, str]:
    """{ea_id: valor} de una columna solo para mostrar (imagen o nombre corto de la carta).
    Vacío con datos guardados: el historial no guarda esas columnas."""
    if column not in movers:
        return {}
    return {int(e): v for e, v in zip(movers["ea_id"], movers[column]) if isinstance(v, str) and v}


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


# ---------------------------------------------------------------------------
# Piezas visuales reutilizables
# ---------------------------------------------------------------------------


def _box(name: str):
    """Tarjeta blanca: un contenedor cuya key empieza por fc_box_ (el CSS la reconoce por ese prefijo)."""
    return st.container(key=f"fc_box_{name}")


def _section(title: str, sub: str | None = None) -> None:
    st.markdown(f"<div class='fc-section'><div class='fc-section-title'>{html.escape(title)}</div>"
                + (f"<div class='fc-section-sub'>{html.escape(sub)}</div>" if sub else "") + "</div>",
                unsafe_allow_html=True)


def _empty(body_html: str) -> None:
    st.markdown(f"<div class='fc-empty'>{body_html}</div>", unsafe_allow_html=True)


def _banner(level: str, body_html: str, icon: str = "!") -> None:
    st.markdown(f"<div class='fc-banner {level}'><span class='fc-ico'>{icon}</span><span>{body_html}</span></div>",
                unsafe_allow_html=True)


def _pill(kind) -> str:
    if kind not in SIGNAL_UI:
        return "<span class='fc-pill none'>Sin señal</span>"
    cls, arrow, text = SIGNAL_UI[kind]
    return f"<span class='fc-pill {cls}'>{arrow} {text}</span>"


def _signal_label(kind) -> str:
    return f"{SIGNAL_UI[kind][1]} {SIGNAL_UI[kind][2]}" if kind in SIGNAL_UI else "—"


def _risk_level(risk: float) -> str:
    """Tono del Risk Score: verde (ok) <35, amarillo (warn) 35-64, rojo (crit) ≥65."""
    return "ok" if risk < 35 else ("warn" if risk < 65 else "crit")


RISK_COLORS = {"ok": TOKENS["green"], "warn": TOKENS["yellow"], "crit": TOKENS["red"]}   # para la tabla (Styler)


def _meter(value, tone: str) -> str:
    return f"<div class='fc-meter'><i class='{tone}' style='width:{max(0, min(100, int(value)))}%'></i></div>"


def _scores_html(row) -> str:
    ms, rk = int(row["market_score"]), int(row["risk_score"])
    level = _risk_level(rk)
    return ("<div class='fc-scores'>"
            f"<div class='fc-score'><div class='row'><span class='fc-label'>Market</span><b>{ms}</b></div>"
            f"{_meter(ms, 'info')}</div>"
            f"<div class='fc-score'><div class='row'><span class='fc-label'>Riesgo</span>"
            f"<b class='{level}'>{rk}</b></div>{_meter(rk, level)}</div></div>")


def _ovr_badge(row) -> str:
    if _is_fodder(row):
        cls, tag = "fodder", "Fodder"
    else:
        rarity = str(row["rarity"] or "")
        cls = "special" if rarity not in ("", "Oro rara (fodder)") else ""
        tag = RARITY_SHORT.get(rarity, rarity[:6])
    return (f"<div class='fc-ovr {cls}'><span class='r'>{row['overall']}</span>"
            f"<span class='t'>{html.escape(tag)}</span></div>")


def _thumb(row, size: str = "md") -> str:
    """Miniatura de la carta: la imagen de FUT.GG si la tenemos; si no, una insignia con la media."""
    image = row.get("image")
    if not _is_fodder(row) and isinstance(image, str) and image:
        return (f"<div class='fc-thumb {size}'><img src='{html.escape(image)}' alt='' loading='lazy' "
                "referrerpolicy='no-referrer'></div>")
    return f"<div class='fc-thumb {size}'>{_ovr_badge(row)}</div>"


def _plan_html(row, short: bool) -> str:
    """Plan de la señal (textos de fc27_signals.trade_plan) con el horizonte."""
    plan = fc27_signals.trade_plan(row)
    names = PLAN_LABELS.get(row["signal"], ("Entrada", "Objetivo", "Invalidación"))
    rows = "".join(
        f"<div class='row'><span>{name}</span><b>{html.escape(_short_numbers(plan[k]) if short else plan[k])}</b></div>"
        for name, k in zip(names, ("zona", "objetivo", "invalidacion")))
    return f"<div class='fc-plan'>{rows}<div class='row'><span>Horizonte</span><b>{row['horizon']}</b></div></div>"


def _why_html(row) -> str:
    """Razones y riesgos de la señal, como listas cortas."""
    out = []
    for cls, title, items, empty in (("ok", "Razones", row["reasons"], "Sin motivos destacados."),
                                     ("warn", "Riesgos", row["risks"], "Sin riesgos destacados.")):
        lis = "".join(f"<li>{html.escape(str(i))}</li>" for i in items) or f"<li class='none'>{empty}</li>"
        out.append(f"<div class='fc-why {cls}'><div class='fc-label'>{title}</div><ul>{lis}</ul></div>")
    return f"<div class='fc-whys'>{''.join(out)}</div>"


def _style_fig(fig: go.Figure, height: int, **layout) -> go.Figure:
    """Estilo común de los gráficos: tipografía, colores suaves y tooltip blanco."""
    fig.update_layout(
        template="plotly_white", height=height, separators=",.", margin=dict(t=10, b=30, l=56, r=12),
        font=dict(family=ui_theme.FONT, size=12, color=TOKENS["muted"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=TOKENS["surface"], bordercolor=TOKENS["border"],
                        font=dict(family=ui_theme.FONT, size=12, color=TOKENS["text"])),
    )
    fig.update_xaxes(gridcolor=ui_theme.GRID, zeroline=False, linecolor=TOKENS["border"])
    fig.update_yaxes(gridcolor=ui_theme.GRID, zeroline=False)
    fig.update_layout(**layout)
    return fig


CHART_CONFIG = {"displayModeBar": False}
RANGES = {"24h": timedelta(hours=24), "7 días": timedelta(days=7), "Todo": None}
EVENT_COLOR = "#64748B"     # eventos del calendario en el gráfico: un solo gris, el texto está en el tooltip


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
    plan, eventos del calendario y tus compras y ventas de Operaciones."""
    hist = fc27_history.price_history(conn, ea_id)
    if len(hist) < 2:
        _empty("Aún no hay historial de esta carta: hacen falta al menos dos instantáneas. Activa las "
               "instantáneas automáticas para que crezca solo.")
        return
    rng = st.segmented_control("Rango", list(RANGES), default="7 días", required=True, key=f"rng_{key}",
                               label_visibility="collapsed")
    end = hist["fetched_at"].max()
    start = end - RANGES[rng] if RANGES[rng] is not None else hist["fetched_at"].min()
    view = hist[hist["fetched_at"] >= start]
    if len(view) < 2:
        view = hist.tail(2)
        start = view["fetched_at"].min()

    fig = go.Figure(go.Scatter(
        x=view["fetched_at"], y=view["price"], mode="lines+markers", name="Precio (consola)",
        line=dict(color=TOKENS["blue"], width=2), marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(29,78,216,0.06)",
        hovertemplate="%{x|%a %d %b %H:%M} UTC<br>%{y:,} monedas<extra></extra>",
    ))
    lo, hi = float(view["price"].min()), float(view["price"].max())

    if row is not None:
        level_colors = {"Objetivo": TOKENS["green"], "Stop": TOKENS["red"], "Entrada": TOKENS["yellow"],
                        "Caída posible": TOKENS["red"], "Invalidación": TOKENS["muted"]}
        for name, value in fc27_signals.plan_levels(row).items():
            color = level_colors.get(name, TOKENS["muted"])
            fig.add_hline(y=value, line=dict(color=color, width=1, dash="dot"),
                          annotation_text=f"{name} {_fmt_short(value)}", annotation_position="top left",
                          annotation_font=dict(size=11, color=color))
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
                fig.add_vline(x=e.when, line=dict(color=TOKENS["border-strong"], width=1, dash="dash"))

    trades = fc27_history.trades(conn)
    if not trades.empty:
        mine = trades[trades["ea_id"] == ea_id]
        for tr in mine.itertuples(index=False):
            for when, price, label, color in ((tr.buy_at, tr.buy_price, "Compra", TOKENS["green"]),
                                              (tr.sell_at, tr.sell_price, "Venta", TOKENS["red"])):
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
            marker=dict(symbol="triangle-down", size=10, color=EVENT_COLOR),
            text=events_at["label"], hovertemplate="Evento: %{text}<br>%{x|%a %d %b %H:%M} UTC<extra></extra>",
        ))
    _style_fig(fig, height, yaxis=dict(title="Monedas (consola)", range=[max(0, lo - pad), hi + pad]),
               showlegend=False, hovermode="x unified")
    st.plotly_chart(fig, width="stretch", key=f"chart_{key}", config=CHART_CONFIG)
    st.markdown("<div class='fc-note'>Precio de consola en tus instantáneas · punteadas: niveles del plan · "
                "▼: eventos del calendario · verticales: tus compras y ventas (PC)</div>", unsafe_allow_html=True)


def _calibration_short(evaluated: pd.DataFrame, kind: str) -> str:
    sub = evaluated[evaluated["kind"] == kind] if not evaluated.empty else evaluated
    n = len(sub)
    if n < MIN_CALIBRATION:
        return f"sin calibrar ({n}/{MIN_CALIBRATION})"
    return f"{sub['hit'].mean() * 100:.0f}% en {n} señales"


# ---------------------------------------------------------------------------
# Cabecera, resumen de hoy, estado del mercado y avisos
# ---------------------------------------------------------------------------


def render_header(conn, snap: fc27_market.MarketSnapshot, live: bool, now: datetime) -> None:
    status = ("<span class='fc-live-tag'><i></i>Mercado en vivo</span>" if live
              else "<span class='fc-live-tag off'><i></i>Sin conexión · datos guardados</span>")
    meta = (f"<span>{len(snap.movers)} cartas</span><span>{fc27_history.snapshot_count(conn)} instantáneas</span>"
            "<span>Precios de consola</span>")
    st.markdown(
        "<div class='fc-head'><div>"
        "<div class='fc-title'>Mercado <span>FC 27</span></div>"
        "<div class='fc-sub'>Inteligencia de mercado para Ultimate Team</div>"
        f"<div class='fc-live'>{status}<span title='{fc27_signals.format_when(now)}'>"
        f"Última actualización: {_age_text(now)}</span></div></div>"
        f"<div class='fc-meta'>{meta}</div></div>",
        unsafe_allow_html=True)


def _hero(cls: str, label: str, ident: str, value: str, why: str, rows: list[tuple[str, str]]) -> str:
    """Una tarjeta del resumen: etiqueta, identidad, cifra grande, contexto y pie con datos."""
    foot = "".join(f"<div class='fc-hero-row'><span>{k}</span><b>{v}</b></div>" for k, v in rows)
    why_html = f"<div class='fc-hero-why' title='{html.escape(why)}'>{html.escape(why)}</div>" if why else ""
    return (f"<div class='fc-hero {cls}'><div class='fc-hero-k'>{label}</div>"
            f"<div class='fc-hero-body'>{ident}<div class='fc-hero-v'>{value}</div>{why_html}</div>"
            f"<div class='fc-hero-foot'>{foot}</div></div>")


def _hero_ident(name: str, meta: str, thumb: str = "") -> str:
    return (f"<div class='fc-hero-id'>{thumb}<div style='min-width:0'><div class='fc-hero-n'>{html.escape(name)}</div>"
            f"<div class='fc-hero-m'>{html.escape(meta)}</div></div></div>")


def render_today(signals: pd.DataFrame, events: pd.DataFrame, now: datetime, tone: str) -> None:
    """Mejor compra, mayor riesgo y próximo evento: tres tarjetas en una sola rejilla, así tienen la misma altura."""
    buy = fc27_signals.top_by_signal(signals, "COMPRAR", 1)
    risk = fc27_signals.top_by_signal(signals, "RIESGO", 1)
    nxt = events[events["when"] >= now].head(1)
    counts = signals["signal"].value_counts()

    if buy.empty:
        card_buy = _hero("buy", "▲ Mejor compra", _hero_ident("Nada claro hoy", "Ninguna carta cumple las condiciones"),
                         "0 <small class='fc-flat'>señales</small>", "Esperar también es una decisión.",
                         [("Comprar", "0 cartas"), ("Vigilar", f"{int(counts.get('VIGILAR', 0))} cartas")])
    else:
        r = buy.iloc[0]
        plan = fc27_signals.trade_plan(r)
        card_buy = _hero("buy", "▲ Mejor compra", _hero_ident(_card_title(r), str(r["rarity"]), _thumb(r, "sm")),
                         f"{_fmt_short(r['price'])} <small>{_pct_html(r['pct_24h'])}</small>",
                         (r["reasons"] or [""])[0],
                         [("Market Score", f"{int(r['market_score'])}"), ("Entrada", _short_numbers(plan["zona"])),
                          ("Objetivo", _short_numbers(plan["objetivo"]))])

    if risk.empty:
        card_risk = _hero("risk", "▼ Mayor riesgo",
                          _hero_ident("Sin riesgos fuertes", "Ninguna carta con señal de caída"),
                          "0 <small class='fc-flat'>señales</small>", "",
                          [("Riesgo", "0 cartas"), ("Mercado 24h", tone)])
    else:
        r = risk.iloc[0]
        levels = fc27_signals.plan_levels(r)
        card_risk = _hero("risk", "▼ Mayor riesgo", _hero_ident(_card_title(r), str(r["rarity"]), _thumb(r, "sm")),
                          f"{_fmt_short(r['price'])} <small>{_pct_html(r['pct_24h'])}</small>",
                          (r["risks"] or r["reasons"] or ["Risk Score alto."])[0],
                          [("Risk Score", f"{int(r['risk_score'])}"),
                           ("Posible caída", f"~{_fmt_short(levels['Caída posible'])}")])

    if nxt.empty:
        card_event = _hero("event", "Calendario", _hero_ident("Sin eventos próximos", "Próximos 30 días"), "—",
                           "Añade fechas en data/fc27_analyst.json.", [("Fuente", "Calendario del analista")])
    else:
        e = nxt.iloc[0]
        hours = (e["when"] - now).total_seconds() / 3600
        left = f"{hours:.0f} h" if hours < 48 else f"{hours / 24:.0f} días"
        card_event = _hero("event", "Calendario", _hero_ident(e["label"], "Próximo evento"),
                           f"{left} <small class='fc-flat'>para el evento</small>", f"{e['status']} · {e['source']}",
                           [("Fecha", fc27_signals.format_when(e["when"])), ("Fiabilidad", html.escape(e["status"]))])

    st.markdown(f"<div class='fc-cq'><div class='fc-heroes'>{card_buy}{card_risk}{card_event}</div></div>",
                unsafe_allow_html=True)


def render_status(snap: fc27_market.MarketSnapshot, signals: pd.DataFrame, tone: str) -> None:
    """Barra compacta para entender el mercado en dos segundos: tono 24h, amplitud y señales."""
    tone_cls = {"Alcista": "up", "Bajista": "down"}.get(tone, "")
    movers = snap.movers
    up = int((movers["pct_24h"] > 0).sum()) if not movers.empty else 0
    down = int((movers["pct_24h"] < 0).sum()) if not movers.empty else 0
    counts = signals["signal"].value_counts()
    sig = "".join(f"<span class='fc-count'><i class='{SIGNAL_UI[k][0]}'>{SIGNAL_UI[k][1]}</i>"
                  f"<b>{int(counts.get(k, 0))}</b>{SIGNAL_UI[k][2]}</span>" for k in ("COMPRAR", "VIGILAR", "RIESGO"))
    st.markdown(
        "<div class='fc-strip'>"
        f"<div class='fc-strip-item'><span class='fc-label'>Mercado 24h</span>"
        f"<span class='fc-tone {tone_cls}'><i></i>{tone}</span></div>"
        "<span class='fc-strip-sep'></span>"
        f"<div class='fc-strip-item'><span><span class='fc-up'>▲ {up}</span> suben</span>"
        f"<span><span class='fc-down'>▼ {down}</span> bajan</span></div>"
        "<span class='fc-strip-sep'></span>"
        "<div class='fc-strip-item' title='Cartas de tu presupuesto con cada señal'>"
        f"<span class='fc-label'>Señales</span>{sig}</div>"
        "</div>",
        unsafe_allow_html=True)


def render_headline_alerts(alerts: list[fc27_signals.Alert], followed: set[int]) -> None:
    """Un aviso destacado se muestra tal cual; si hay varios, se agrupan en un desplegable."""
    head = fc27_signals.headline_alerts(alerts, followed)
    if not head:
        return
    if len(head) == 1:
        a = head[0]
        _banner("crit" if a.level == "crítica" else "warn", f"<b>{html.escape(a.title)}</b> · {html.escape(a.detail)}")
        return
    level = "crit" if any(a.level == "crítica" for a in head) else "warn"
    items = "".join(f"<li><b>{html.escape(a.title)}</b><span>{html.escape(a.detail)}</span></li>" for a in head)
    st.markdown(f"<details class='fc-banner {level}'><summary><span class='fc-ico'>!</span>"
                f"<b>{len(head)} alertas importantes</b><span class='more'>Ver alertas</span></summary>"
                f"<ul>{items}</ul></details>", unsafe_allow_html=True)


def render_since_last_visit(conn, snap: fc27_market.MarketSnapshot, alerts: list, previous: datetime | None) -> None:
    """Novedades desde tu última visita (SBC nuevos, señales nuevas y avisos de precio). Sin novedades, no ocupa sitio."""
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
    groups = [(title, items) for title, items in (("SBC nuevos", new_sbcs), ("Nuevas compras", buys),
                                                   ("Nuevos riesgos", risks), ("Avisos de precio", price_hits)) if items]
    if not groups:
        return
    summary = " · ".join(f"{title} <b>{len(items)}</b>" for title, items in groups)
    items_html = "".join(
        f"<li><b>{title}</b><span>{', '.join(html.escape(str(i)) for i in items[:15])}"
        f"{f' y {len(items) - 15} más' if len(items) > 15 else ''}</span></li>" for title, items in groups)
    st.markdown(f"<details class='fc-banner info'><summary><span class='fc-ico'>i</span>"
                f"<span>Desde tu última visita ({_age_text(previous)}): {summary}</span>"
                f"<span class='more'>Ver novedades</span></summary><ul>{items_html}</ul></details>",
                unsafe_allow_html=True)


def render_nav() -> str:
    """Navegación con jerarquía: cuatro secciones principales y "Más" con las secundarias."""
    with st.container(key="fc_nav"):
        section = st.segmented_control("Sección", list(NAV), format_func=NAV.get, default="signals", required=True,
                                       key="fc_section", label_visibility="collapsed", width="stretch")
        if section == "more":
            with st.container(key="fc_nav_more"):
                section = st.segmented_control("Más secciones", list(NAV_MORE), format_func=NAV_MORE.get,
                                               default="fodder", required=True, key="fc_section_more",
                                               label_visibility="collapsed")
    return section


# ---------------------------------------------------------------------------
# Señales
# ---------------------------------------------------------------------------


@st.dialog("Análisis de la carta", width="large")
def card_dialog(conn, row: pd.Series, followed: set[int], analyst: dict) -> None:
    _market_sheet(conn, row, followed, analyst, key=f"dlg_{_safe_key(row['key'])}")


def _strategy(row) -> tuple[str, str]:
    """La estrategia en dos líneas cortas para la tarjeta (textos de trade_plan, con precios abreviados)."""
    plan = fc27_signals.trade_plan(row)
    horizon = f"Horizonte {row['horizon']}"
    kind = row["signal"]
    if kind == "COMPRAR":
        main, sub = f"Entrada {plan['zona']} · Objetivo {plan['objetivo']}", horizon
    elif kind == "VIGILAR":
        main, sub = plan["zona"], f"Objetivo {plan['objetivo']} · {horizon}"
    elif kind == "RIESGO":
        main, sub = plan["objetivo"], horizon
    else:
        main, sub = "Sin señal ahora mismo.", horizon
    return _short_numbers(main), _short_numbers(sub)


def _signal_card(conn, row: pd.Series, followed: set[int], series: dict[int, list[int]], analyst: dict) -> None:
    """Tarjeta de una señal. Jerarquía: jugador, precio, señal, cambio 24h, scores, estrategia e historial;
    el análisis completo queda plegado en "Ver análisis"."""
    safe = _safe_key(row["key"])
    fodder = _is_fodder(row)
    has_id = not pd.isna(row["ea_id"])
    name = html.escape(str(row["name"]))
    meta = "Mejor fodder para SBC" if fodder else html.escape(str(row["rarity"]))
    star = ("<span class='fc-star' title='En tu lista'>★</span>"
            if not fodder and has_id and int(row["ea_id"]) in followed else "")
    pts = series.get(int(row["ea_id"]), []) if has_id else []
    spark = ui_theme.sparkline_svg(pts, width=300, height=40)
    spark_html = (f"<div class='fc-spark'><span class='fc-label'>Historial · 3 días</span>{spark}</div>" if spark else
                  "<div class='fc-spark'><span class='fc-label'>Historial</span>"
                  "<span class='none'>Aún sin historial suficiente</span></div>")
    main, sub = _strategy(row)
    with st.container(key=f"fc_card_{safe}"):
        st.markdown(
            "<div class='fc-sig'>"
            f"<div class='fc-sig-top'>{_thumb(row)}<div class='fc-sig-id'>"
            f"<div class='fc-sig-name' title='{name}'>{name}{star}</div><div class='fc-sig-meta'>{meta}</div></div>"
            f"{_pill(row['signal'])}</div>"
            f"<div class='fc-sig-price'><div class='fc-price' title='{_fmt(row['price'])} monedas (consola)'>"
            f"{_fmt_short(row['price'])}<small>consola</small></div>"
            f"<div class='fc-chg'>{_pct_html(row['pct_24h'])}<small>24h</small></div></div>"
            f"{_scores_html(row)}"
            f"<div class='fc-strat'><b>{html.escape(main)}</b><span>{html.escape(sub)}</span></div>"
            f"{spark_html}</div>",
            unsafe_allow_html=True)
        with st.expander("Ver análisis"):
            st.markdown(_plan_html(row, short=True) + _why_html(row), unsafe_allow_html=True)
            c1, c2 = st.columns(2) if not fodder else (st.container(), None)
            with c1:
                if st.button("Ver ficha", icon=":material/open_in_full:", key=f"sig_open_{safe}", width="stretch",
                             help="Ficha completa con el gráfico de precio"):
                    card_dialog(conn, row, followed, analyst)
            if c2 is not None:
                with c2:
                    _follow_button(conn, row, followed, key=f"sig_follow_{safe}")


def _ordered_signals(signals: pd.DataFrame, flt: str) -> pd.DataFrame:
    """Señales en orden de relevancia. En "Todas", primero las compras y después riesgo y vigilar
    intercalados (cada lista en el orden de fc27_signals.top_by_signal)."""
    if flt in KINDS:
        return fc27_signals.top_by_signal(signals, flt, n=len(signals))
    tops = {k: fc27_signals.top_by_signal(signals, k, n=len(signals)) for k in KINDS}
    risk, watch = tops["RIESGO"], tops["VIGILAR"]
    mixed = [df.iloc[[i]] for i in range(max(len(risk), len(watch))) for df in (risk, watch) if i < len(df)]
    return pd.concat([tops["COMPRAR"], *mixed])


EMPTY_SIGNALS = {
    "ALL": "Ninguna carta de tu presupuesto tiene señal ahora mismo.",
    "COMPRAR": "Ninguna carta de tu presupuesto cumple las condiciones de compra. Esperar también es una decisión.",
    "VIGILAR": "Ninguna carta de tu presupuesto tiene motivos para vigilarla ahora mismo.",
    "RIESGO": "Ninguna carta de tu presupuesto tiene señal de caída ahora mismo.",
}


def _show_more(key: str, value: int) -> None:
    st.session_state[key] = value


def render_signals(conn, signals: pd.DataFrame, evaluated: pd.DataFrame, analyst: dict, now: datetime,
                   followed: set[int]) -> None:
    counts = signals["signal"].value_counts()
    n = {k: int(counts.get(k, 0)) for k in KINDS}
    labels = {"ALL": f"Todas · {sum(n.values())}", "COMPRAR": f":green[▲] Comprar · {n['COMPRAR']}",
              "VIGILAR": f":orange[●] Vigilar · {n['VIGILAR']}", "RIESGO": f":red[▼] Riesgo · {n['RIESGO']}"}
    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute", gap="small"):
        flt = st.segmented_control(
            "Filtrar señales", list(labels), format_func=labels.get, default="ALL", required=True, key="fc_sig_filter",
            label_visibility="collapsed", wrap=True,
            help="Comprar: Market Score ≥70 y riesgo ≤45. Vigilar: algo de interés, aún sin todas las condiciones. "
                 "Riesgo: Risk Score ≥70 o Market Score <40.")
        st.markdown(f"<div class='fc-note'>Acierto real · Comprar: {_calibration_short(evaluated, 'COMPRAR')} · "
                    f"Riesgo: {_calibration_short(evaluated, 'RIESGO')}</div>", unsafe_allow_html=True)

    ordered = _ordered_signals(signals, flt)
    if ordered.empty:
        _empty(EMPTY_SIGNALS[flt])
    else:
        limit_key = f"fc_sig_limit_{flt}"
        limit = st.session_state.get(limit_key, SIGNAL_PAGE)
        shown = ordered.head(limit)
        series = fc27_history.price_series(conn, shown["ea_id"], now - timedelta(days=3))
        with st.container(key="fc_grid"):
            for _, row in shown.iterrows():
                _signal_card(conn, row, followed, series, analyst)
        if len(ordered) > limit:
            st.button(f"Mostrar más señales ({len(ordered) - limit})", key=f"fc_sig_more_{flt}",
                      on_click=_show_more, args=(limit_key, limit + SIGNAL_PAGE), width="stretch")

    notes = fc27_signals.active_notes(analyst, now)
    if notes:
        with st.expander(f"Notas del analista ({len(notes)})", icon=":material/edit_note:"):
            st.caption(f"Tesis manuales de data/fc27_analyst.json (actualizado {analyst.get('updated', '—')}). "
                       "Desaparecen solas al caducar.")
            for note in notes:
                st.markdown(f"{_pill(note.get('signal', ''))} **{html.escape(note['player'])}**  \n"
                            f"{html.escape(note['thesis'])}  \n*Invalidación:* {html.escape(note['invalidation'])}",
                            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Ficha de mercado de una carta (tabla de Mercado y ventana de las señales)
# ---------------------------------------------------------------------------


def _sheet_html(row) -> str:
    rarity = "Mejor fodder para SBC" if _is_fodder(row) else str(row["rarity"])
    return (
        "<div class='fc-sheet'>"
        f"<div class='fc-sheet-head'>{_thumb(row, 'lg')}<div style='min-width:0'>"
        f"<div class='fc-sheet-name'>{html.escape(_card_title(row))}</div>"
        f"<div class='fc-sheet-meta'>{html.escape(rarity)}</div>"
        f"{_pill(row['signal'])} <span class='fc-chip' title='Lectura del Market Score'>"
        f"{html.escape(str(row['label']))}</span>"
        "</div></div>"
        "<div class='fc-kv'>"
        f"<div><div class='fc-label'>Precio · consola</div><div class='v'>{_fmt(row['price'])}</div></div>"
        f"<div><div class='fc-label'>Cambio 24h</div><div class='v'>{_pct_html(row['pct_24h'])}</div></div></div>"
        f"{_scores_html(row)}{_plan_html(row, short=False)}{_why_html(row)}</div>")


def _market_sheet(conn, row: pd.Series, followed: set[int], analyst: dict, key: str) -> None:
    """Ficha profesional: a la izquierda cifras, plan, razones y riesgos; a la derecha el gráfico de precio."""
    chart = not _is_fodder(row) and not pd.isna(row["ea_id"])
    left, right = st.columns([5, 7], gap="large") if chart else (st.container(), None)
    with left:
        st.markdown(_sheet_html(row), unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            _follow_button(conn, row, followed, key=f"{key}_follow")
        if row.get("url"):
            b2.link_button("Abrir en FUT.GG", f"{FUTGG}{row['url']}", icon=":material/open_in_new:", width="stretch")
        with st.expander("Desglose del Market Score"):
            bd = pd.DataFrame([(label, int(row[col]), mx) for col, label, mx in fc27_signals.SCORE_COMPONENTS],
                              columns=["Componente", "Puntos", "Máximo"])
            st.dataframe(bd, hide_index=True, width="stretch",
                         column_config={"Puntos": st.column_config.ProgressColumn(
                             "Puntos", min_value=0, max_value=20, format="%d", color=TOKENS["blue"])})
    if right is not None:
        with right:
            _section("Precio", "Consola · tus instantáneas")
            _price_chart(conn, int(row["ea_id"]), key=key, row=row, analyst=analyst, height=340)


# ---------------------------------------------------------------------------
# Mi lista y Operaciones
# ---------------------------------------------------------------------------


def render_watchlist(conn, signals: pd.DataFrame, analyst: dict) -> None:
    wl = fc27_history.watchlist(conn)
    if wl.empty:
        _empty("Tu lista está vacía. Pulsa <b>☆ Seguir</b> en una señal, o selecciona una carta en <b>Mercado</b> y "
               "añádela. Aquí apuntas tus precios de PC y la app te calcula el precio para no perder, el objetivo y "
               "el beneficio real tras el 5% de EA.")
        return
    current = signals.drop_duplicates("ea_id").set_index("ea_id")
    wl["signal"] = wl["ea_id"].map(current["signal"]).fillna("—")
    wl["link"] = FUTGG + wl["url"].fillna("")

    _section("Tus precios de PC", "FUT.GG no publica precios de PC de forma abierta: apunta a cuánto compraste y "
                                  "cuánto vale ahora (en el juego o en FUT.GG). La app calcula el resto con el 5% de EA.")
    labels = {f"{r['name']} {'' if pd.isna(r['overall']) else int(r['overall'])}".strip(): r for _, r in wl.iterrows()}
    with st.form("pc_prices_form", border=True):
        c1, c2, c3, c4 = st.columns([2.2, 1.3, 1.3, 1], vertical_alignment="bottom")
        choice = c1.selectbox("Carta", list(labels), key="pc_card")
        row = labels[choice]
        buy = c2.number_input("Compré a (PC)", min_value=0, step=500, key=f"pc_buy_{row['ea_id']}",
                              value=None if pd.isna(row["buy_price_pc"]) else int(row["buy_price_pc"]),
                              placeholder="Ej. 150000")
        now_price = c3.number_input("Vale ahora (PC)", min_value=0, step=500, key=f"pc_now_{row['ea_id']}",
                                    value=None if pd.isna(row["price_pc"]) else int(row["price_pc"]),
                                    placeholder="Ej. 171000")
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

    _section("Avisos de precio", "Te avisa arriba de la página cuando el precio de consola de FUT.GG cruza tu límite. "
                                 "Se comprueba con cada instantánea; déjalo vacío para no avisar.")
    alert_labels = {f"{r['name']} {'' if pd.isna(r['overall']) else int(r['overall'])}".strip(): r
                    for _, r in wl.iterrows()}
    with st.form("price_alerts_form", border=True):
        a1, a2, a3, a4 = st.columns([2.2, 1.3, 1.3, 1], vertical_alignment="bottom")
        pick = a1.selectbox("Carta", list(alert_labels), key="alert_card")
        r = alert_labels[pick]
        below = a2.number_input("Avisar si baja de", min_value=0, step=500, key=f"al_below_{r['ea_id']}",
                                value=None if pd.isna(r["alert_below"]) else int(r["alert_below"]),
                                placeholder=f"Ahora {_fmt_short(r['last_price'])}")
        above = a3.number_input("Avisar si sube de", min_value=0, step=500, key=f"al_above_{r['ea_id']}",
                                value=None if pd.isna(r["alert_above"]) else int(r["alert_above"]),
                                placeholder=f"Ahora {_fmt_short(r['last_price'])}")
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
    with _box("wl_chart"):
        _section("Gráfico de precio", "Consola · tus instantáneas")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        choice = c1.selectbox("Carta", list(names), key="wl_chart")
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
        line=dict(color=TOKENS["blue"], width=2, shape="hv"), marker=dict(size=7),
        customdata=curve[["card_name", "net_profit"]],
        hovertemplate="%{x|%d %b %H:%M}<br>%{customdata[0]}: %{customdata[1]:+,.0f}<br>"
                      "Acumulado %{y:+,.0f}<extra></extra>",
    ))
    if (curve["drawdown"] < 0).any():
        fig.add_trace(go.Scatter(
            x=curve["sell_at"], y=curve["drawdown"], mode="lines", name="Caída desde el máximo",
            line=dict(color=TOKENS["red"], width=1, shape="hv"), fill="tozeroy", fillcolor="rgba(220,38,38,0.10)",
            hovertemplate="%{x|%d %b}<br>Caída %{y:,.0f}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=TOKENS["faint"], width=1))
    _style_fig(fig, 260, yaxis=dict(title="Monedas (PC, neto)"), legend=dict(orientation="h", y=1.12, x=0),
               hovermode="x unified")
    with _box("equity"):
        _section("Curva de beneficio", "Beneficio neto acumulado (PC) y caída desde el máximo")
        st.plotly_chart(fig, width="stretch", key="equity_curve", config=CHART_CONFIG)


def render_trades(conn, signals: pd.DataFrame) -> None:
    df = fc27_history.trades(conn)
    s = fc27_history.trade_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beneficio neto", f"{_fmt(s['net_profit'])}", f"{s['closed']} cerradas", delta_color="off")
    c2.metric("ROI", "—" if s["roi_pct"] is None else f"{s['roi_pct']:+.1f}%".replace(".", ","),
              help="Beneficio neto / dinero invertido en operaciones cerradas")
    c3.metric("Operaciones ganadoras", "—" if s["win_rate_pct"] is None else f"{s['win_rate_pct']:.0f}%")
    c4.metric("Invertido en abiertas", _fmt(s["capital_in_open"]), f"{s['open']} abiertas", delta_color="off")
    st.markdown(f"<div class='fc-note'>Precios de PC por unidad, como en el juego. Impuesto de EA pagado en ventas: "
                f"{_fmt(s['taxes_paid'])}.</div>", unsafe_allow_html=True)
    render_equity_curve(df)

    left, right = st.columns(2)
    with left, st.form("trade_buy", border=True, clear_on_submit=True):
        _section("Registrar compra")
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
        _section("Registrar venta")
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
        _empty("Todavía no hay operaciones. Cada vez que compres o vendas en el juego, apúntalo aquí: verás tu "
               "beneficio real después del 5% de EA y qué señales te hacen ganar dinero.")
        return
    _section("Historial")
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
        _section("¿Qué señales te hacen ganar dinero?")
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


# ---------------------------------------------------------------------------
# Mercado: filtros, mapa, tabla y ficha
# ---------------------------------------------------------------------------


def _relevance(cards: pd.DataFrame) -> pd.Series:
    """Orden de relevancia para decidir qué cartas se muestran primero en el mapa y la tabla.
    Solo ordena: no cambia ninguna puntuación. Pesa el movimiento absoluto de 24h, Market Score,
    Risk Score y precio (escala logarítmica), y adelanta las cartas con señal."""
    move = cards["pct_24h"].abs().fillna(0).clip(upper=30) / 30
    price = ((np.log10(cards["price"].clip(lower=1000)) - 3) / 4).clip(0, 1)
    signal = cards["signal"].map({"COMPRAR": 0.3, "RIESGO": 0.3, "VIGILAR": 0.15}).fillna(0)
    return 0.4 * move + 0.2 * cards["market_score"] / 100 + 0.2 * cards["risk_score"] / 100 + 0.2 * price + signal


def _treemap_figure(data: pd.DataFrame) -> go.Figure:
    """Mapa de calor por rareza. Tamaño: precio (escala logarítmica, para que los Icons millonarios no tapen al
    resto). Color: variación de 24h. Cuanto más grande el bloque, más datos muestra su etiqueta; las etiquetas
    usan el nombre corto de la carta y el tooltip, el completo."""
    size = (np.log10(data["price"].clip(lower=1000)) - 2.5).to_numpy(float)
    labels = data["short_name"].fillna(data["name"]) if "short_name" in data else data["name"]
    share = size / size.sum()
    pct = data["pct_24h"].to_numpy(float)
    group = data["rarity"].fillna("Otras").replace({"Team of the week": "TOTW", "Destined for Glory": "DFG",
                                                     "Base Icon": "Icons", "Base Hero": "Heroes"}).to_numpy()
    groups = pd.Series(size).groupby(group).sum()
    medians = pd.Series(pct).groupby(group).median()

    big = "<b>%{label}</b><br>%{customdata[0]}<br>%{customdata[1]}<br><b>%{customdata[2]}</b>"
    mid = "<b>%{label} %{customdata[3]}</b><br>%{customdata[2]}"
    leaf_text = np.where(share >= 0.025, big, np.where(share >= 0.015, mid, "%{label}"))
    leaf_data = [[f"{o} {RARITY_SHORT.get(r, r)}", _fmt_short(p), _pct_txt(c), o, r, _fmt(p), int(ms), int(rk),
                  _signal_label(s), "", "", n]
                 for o, r, p, c, ms, rk, s, n in zip(data["overall"], data["rarity"], data["price"], pct,
                                                     data["market_score"], data["risk_score"], data["signal"],
                                                     data["name"])]
    group_data = [["", "", "", "", "", "", "", "", "", int((group == g).sum()), _pct_txt(medians[g]), ""]
                  for g in groups.index]
    leaf_hover = ("<b>%{customdata[11]} %{customdata[3]}</b><br>%{customdata[4]}<br><br>Precio  <b>%{customdata[5]}</b>"
                  "<br>24h  <b>%{customdata[2]}</b><br>Market Score  <b>%{customdata[6]}</b>"
                  "<br>Risk Score  <b>%{customdata[7]}</b><br>Señal  <b>%{customdata[8]}</b><extra></extra>")
    group_hover = "<b>%{label}</b><br>%{customdata[9]} cartas · mediana 24h %{customdata[10]}<extra></extra>"
    n_groups = len(groups)
    fig = go.Figure(go.Treemap(
        ids=["all", *[f"g|{g}" for g in groups.index], *[f"c|{k}" for k in data["key"]]],
        labels=["Mercado", *groups.index, *labels],
        parents=["", *["all"] * n_groups, *[f"g|{g}" for g in group]],
        values=[0, *[0] * n_groups, *size], branchvalues="remainder",
        marker=dict(colors=[0, *[0] * n_groups, *pct], colorscale=HEAT_SCALE, cmin=-HEAT_RANGE, cmax=HEAT_RANGE,
                    cmid=0, showscale=False, line=dict(width=2, color=TOKENS["surface"]), cornerradius=5),
        customdata=[[""] * 12, *group_data, *leaf_data],
        texttemplate=["", *["<b>%{label}</b>"] * n_groups, *leaf_text],
        hovertemplate=["<b>Mercado</b><extra></extra>", *[group_hover] * n_groups, *[leaf_hover] * len(data)],
        textfont=dict(family=ui_theme.FONT, size=13,
                      color=[TOKENS["text"]] * (1 + n_groups)
                      + ["#FFFFFF" if abs(c) >= 9 else TOKENS["text"] for c in pct]),
        textposition="middle center", pathbar=dict(visible=True, thickness=24), tiling=dict(pad=2), sort=True,
    ))
    return _style_fig(fig, 540, margin=dict(t=6, b=6, l=0, r=0), uniformtext=dict(minsize=10, mode="hide"))


def render_heatmap(view: pd.DataFrame, top: str) -> None:
    data = view.dropna(subset=["pct_24h"])
    shown = f"{len(data)} cartas" + (f" · {top.lower()} por relevancia" if TOP_OPTIONS[top] else "")
    ticks = "".join(f"<span>{t:+d}%</span>" if t else "<span>0%</span>" for t in range(-HEAT_RANGE, HEAT_RANGE + 1, 5))
    _section("Mapa del mercado", shown)
    st.markdown(
        "<div class='fc-legend'><div class='fc-legend-scale'><span class='fc-label'>Variación 24h</span>"
        f"<div class='fc-legend-bar'></div><div class='fc-legend-ticks'>{ticks}</div></div>"
        "<div class='fc-legend-keys'><span>Tamaño → <b>precio</b></span><span>Color → <b>movimiento 24h</b></span>"
        "<span>Clic en una rareza para ampliarla</span></div></div>",
        unsafe_allow_html=True)
    if data.empty:
        _empty("Ninguna carta de la selección tiene variación de 24h.")
        return
    st.plotly_chart(_treemap_figure(data), width="stretch", key="heatmap", config=CHART_CONFIG)


def _css_change(v) -> str:
    if pd.isna(v) or v == 0:
        return ""
    return f"color: {TOKENS['green'] if v > 0 else TOKENS['red']}"


def _css_risk(v) -> str:
    return "" if pd.isna(v) else f"color: {RISK_COLORS[_risk_level(v)]}"


def _css_signal(v) -> str:
    colors = {_signal_label(k): TOKENS[c] for k, c in (("COMPRAR", "green"), ("VIGILAR", "yellow"), ("RIESGO", "red"))}
    return f"color: {colors[v]}" if v in colors else f"color: {TOKENS['faint']}"


def render_market_table(conn, view: pd.DataFrame, followed: set[int]) -> pd.Series | None:
    """Tabla del mercado en vista básica (lo esencial, precios cortos) o avanzada (todas las columnas).
    Devuelve la fila seleccionada, si la hay."""
    # El selector de vista va a la izquierda: a la derecha, encima de la tabla, aparece su barra de herramientas.
    _section("Cartas", f"{len(view)} cartas · selecciona una fila para ver su ficha")
    mode = st.segmented_control("Vista", ["Básica", "Avanzada"], default="Básica", required=True, key="mk_mode",
                                label_visibility="collapsed")
    star = np.where(view["ea_id"].isin(followed), "★ ", "")
    t = view.assign(card=star + view["name"] + " " + view["overall"].astype(str), name=star + view["name"],
                    sig=view["signal"].map(_signal_label))
    for col in ("pct_1h", "pct_6h", "pct_24h", "pct_72h", "pct_168h"):
        t[col] = pd.to_numeric(t[col], errors="coerce")

    if mode == "Básica":
        pct_cols = ["pct_24h"]
        cols = ["card", "price", "pct_24h", "market_score", "risk_score", "sig"]
        price_fmt = _fmt_short
        caption = None
    else:
        history_cols = [c for c in ("pct_1h", "pct_6h", "pct_72h", "pct_168h") if t[c].notna().any()]
        pct_cols = [c for c in ("pct_1h", "pct_6h", "pct_24h", "pct_72h", "pct_168h") if c == "pct_24h" or c in history_cols]
        trend = fc27_history.price_series(conn, t["ea_id"], datetime.now(timezone.utc) - timedelta(days=3))
        t["trend"] = t["ea_id"].map(lambda i: trend.get(int(i)) if len(trend.get(int(i), [])) >= 2 else None)
        cols = ["name", "overall", "rarity", "price", "trend", *pct_cols, "market_score", "risk_score", "sig"]
        price_fmt = _fmt
        caption = ("Las columnas de 1h, 6h, 3d y 7d aparecen cuando tu historial las cubre."
                   if len(history_cols) < 4 else None)

    styled = (t[cols].style
              .format({"price": price_fmt, "market_score": "{:.0f}", "risk_score": "{:.0f}",
                       **{c: _pct_txt for c in pct_cols}}, na_rep="—")
              .map(_css_change, subset=pct_cols)
              .map(_css_risk, subset=["risk_score"])
              .map(_css_signal, subset=["sig"]))
    event = st.dataframe(
        styled, hide_index=True, width="stretch", height=min(470, 38 + 35 * max(len(t), 1)),
        on_select="rerun", selection_mode="single-row", key="mk_table",
        column_config={
            "card": st.column_config.TextColumn("Carta", width="medium", help="★ = está en Mi lista"),
            "name": st.column_config.TextColumn("Carta", help="★ = está en Mi lista"),
            "overall": st.column_config.NumberColumn("OVR", width="small"),
            "rarity": "Rareza",
            "price": st.column_config.Column("Precio", help="Precio de consola (FUT.GG)"),
            "trend": st.column_config.LineChartColumn("Tendencia", width="small",
                                                      help="Precio de consola en tus instantáneas de los últimos 3 días"),
            "pct_1h": "1h", "pct_6h": "6h", "pct_24h": "24h", "pct_72h": "3d", "pct_168h": "7d",
            "market_score": st.column_config.Column("Market", help="Market Score (0-100)"),
            "risk_score": st.column_config.Column("Riesgo", help="Risk Score (0-100): verde <35, amarillo <65, rojo ≥65"),
            "sig": "Señal",
        },
    )
    if caption:
        st.markdown(f"<div class='fc-note'>{caption}</div>", unsafe_allow_html=True)
    rows = event.selection.rows if event and event.selection else []
    return view.iloc[rows[0]] if rows else None


def render_market(conn, signals: pd.DataFrame, followed: set[int], analyst: dict) -> None:
    cards = signals[~signals["key"].str.startswith("fodder")].copy()
    if cards.empty:
        _empty("No hay cartas en tu presupuesto. Cambia el presupuesto en la barra lateral.")
        return
    cards["relevance"] = _relevance(cards)
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key="fc_mk_filters"):
        query = st.text_input("Buscar jugador", key="mk_q", placeholder="Ej. Mbappé")
        rarity = st.selectbox("Rareza", ["Todas", *sorted(cards["rarity"].dropna().unique())], key="mk_rarity", width=200)
        move = st.segmented_control("Movimiento 24h", ["Todas", "Suben", "Bajan"], default="Todas", required=True,
                                    key="mk_move")
        top = st.segmented_control("Cartas", list(TOP_OPTIONS), default="Top 50", required=True, key="mk_top",
                                   help="Top: primero las cartas con señal y las que más se mueven en 24h; después "
                                        "Market Score, Risk Score y precio.")
    view = cards
    if query:
        view = view[view["name"].str.contains(query, case=False, na=False, regex=False)]
    if rarity != "Todas":
        view = view[view["rarity"] == rarity]
    if move in ("Suben", "Bajan"):
        view = view[view["pct_24h"] > 0] if move == "Suben" else view[view["pct_24h"] < 0]
    view = view.sort_values("relevance", ascending=False)
    if TOP_OPTIONS[top]:
        view = view.head(TOP_OPTIONS[top])
    view = view.reset_index(drop=True)
    if view.empty:
        _empty("Ninguna carta coincide con los filtros.")
        return
    with _box("map"):
        render_heatmap(view, top)
    with _box("table"):
        row = render_market_table(conn, view, followed)
    if row is not None:
        with _box("sheet"):
            _market_sheet(conn, row, followed, analyst, key="market")


# ---------------------------------------------------------------------------
# Fodder & SBC, Alertas y Cómo funciona
# ---------------------------------------------------------------------------


# Índice de fodder: tonos de azul de claro a oscuro según el rating (información, no señal).
FODDER_COLORS = {84: "#93B4F5", 85: "#3B6FE0", 86: TOKENS["blue-ink"]}


def render_fodder_index(conn) -> None:
    """Evolución del precio de referencia del fodder 84/85/86 en tus instantáneas."""
    idx = fc27_history.fodder_index(conn, ratings=tuple(FODDER_COLORS))
    _section("Índice de fodder (84 · 85 · 86)", "Mediana de las 5 más baratas de cada rating, normalizada a 100 al "
                                               "inicio del rango. Si sube, crece la demanda de fodder para SBC.")
    if idx.empty or idx["fetched_at"].nunique() < 2:
        _empty("Se necesitan al menos dos instantáneas para dibujar el índice.")
        return
    rng = st.segmented_control("Rango del índice", list(RANGES), default="Todo", required=True, key="rng_fodder",
                               label_visibility="collapsed")
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
        summary.append(f"<b>{ovr}</b>: {_fmt(first)} → {_fmt(last)} ({_pct_html(change)})")
        fig.add_trace(go.Scatter(
            x=s["fetched_at"], y=s["price"] / first * 100, mode="lines+markers", name=f"Rating {ovr}",
            line=dict(color=color, width=2), marker=dict(size=5), customdata=s["price"],
            hovertemplate=f"Rating {ovr}<br>%{{x|%a %d %b %H:%M}} UTC<br>%{{customdata:,}} monedas"
                          "<br>Índice %{y:.0f}<extra></extra>",
        ))
        fig.add_annotation(x=s["fetched_at"].iloc[-1], y=last / first * 100, text=f"{ovr}", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=color, size=12))
    fig.add_hline(y=100, line=dict(color=TOKENS["faint"], width=1, dash="dot"))
    _style_fig(fig, 300, margin=dict(t=10, b=30, l=50, r=30), yaxis=dict(title="Índice (inicio = 100)"),
               legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    st.plotly_chart(fig, width="stretch", key="fodder_index", config=CHART_CONFIG)
    st.markdown(f"<div class='fc-note'>{' · '.join(summary)}</div>", unsafe_allow_html=True)


def render_fodder(snap: fc27_market.MarketSnapshot, conn) -> None:
    fodder = fc27_market.fodder_table(snap.cheapest)
    with _box("fodder_index"):
        render_fodder_index(conn)
    left, right = st.columns([3, 2])
    with left, _box("fodder_points"):
        _section("Monedas por punto de Item Score", "Los SBC piden puntos de Item Score: cuanto menos cueste cada "
                                                    "punto, mejor fodder. Referencia: mediana de las 5 más baratas.")
        if fodder.empty:
            _empty("No hay datos de las cartas más baratas por rating.")
        else:
            colors = [TOKENS["green"] if b else "#C7D2E0" for b in fodder["is_best"]]
            fig = go.Figure(go.Bar(
                x=fodder["overall"].astype(str), y=fodder["coins_per_point"], marker_color=colors,
                customdata=fodder[["name", "price", "item_score"]],
                hovertemplate="Rating %{x} · más barato: %{customdata[0]}<br>Referencia %{customdata[1]:,} monedas · "
                              "%{customdata[2]:,} pts<br>%{y:.2f} monedas por punto<extra></extra>",
            ))
            _style_fig(fig, 300, margin=dict(t=10, b=40, l=40, r=10), xaxis_title="Rating",
                       yaxis_title="Monedas por punto")
            st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
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
            st.markdown(f"<div class='fc-note'>Precio mínimo observado en 81-84: {_fmt(floor)} (probable suelo de "
                        "EA). Item Score de 83 y 89 tomado de tablas de terceros. En verde, el rating con el punto "
                        "más barato.</div>", unsafe_allow_html=True)
    with right, _box("sbcs"):
        _section("SBC activos", "Coste estimado por FUT.GG con la ruta más barata. Sin los SBC permanentes.")
        if snap.sbcs.empty:
            _empty("No hay datos de SBC.")
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


def render_alerts(alerts: list[fc27_signals.Alert], analyst: dict, now: datetime) -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        _section("Alertas", "Ordenadas por importancia. Las de 1h necesitan dos instantáneas separadas al menos una hora.")
        if not alerts:
            _empty("Sin alertas ahora mismo.")
        else:
            rows = "".join(
                f"<div class='fc-list-row'><span class='fc-dot {LEVEL_CLASS.get(a.level, 'info')}'></span><div>"
                f"<div class='fc-list-t'>{html.escape(a.title)} <span class='fc-chip'>{html.escape(a.status)}</span></div>"
                f"<div class='fc-list-d'>{html.escape(a.detail)}</div></div></div>" for a in alerts)
            st.markdown(f"<div class='fc-list'>{rows}</div>", unsafe_allow_html=True)
    with right:
        _section("Calendario", "Próximos 30 días · mantenido a mano en data/fc27_analyst.json")
        events = fc27_signals.upcoming_events(analyst, now)
        if events.empty:
            _empty("No hay eventos en data/fc27_analyst.json para los próximos 30 días.")
        else:
            rows = []
            for e in events.itertuples(index=False):
                day, _, hour = fc27_signals.format_when(e.when).partition(" · ")
                rows.append(f"<div class='fc-list-row'><div class='fc-when'>{day}<span>{hour}</span></div><div>"
                            f"<div class='fc-list-t'>{html.escape(e.label)}</div>"
                            f"<div class='fc-list-d'><span class='fc-chip'>{html.escape(e.status)}</span> "
                            f"{html.escape(str(e.source))}</div></div></div>")
            st.markdown(f"<div class='fc-list'>{''.join(rows)}</div>", unsafe_allow_html=True)


def render_help(evaluated: pd.DataFrame, snap: fc27_market.MarketSnapshot) -> None:
    with _box("breadth"):
        _section("Estado del mercado por rareza (24h)")
        breadth = fc27_signals.market_breadth(snap.movers)
        if not breadth.empty:
            st.dataframe(breadth, hide_index=True, width="stretch", column_config={
                "rarity": "Rareza", "cards": "Cartas", "median_24h": st.column_config.NumberColumn("Mediana 24h %", format="%+.1f"),
                "up": "Suben", "down": "Bajan", "tone": "Tono"})

    _section("¿Aciertan las señales?")
    if evaluated.empty:
        _empty("Todavía no hay señales con el horizonte cumplido. Cada señal COMPRAR o RIESGO se guarda y se "
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

    with _box("howto"):
        _section("Cómo funciona")
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

**Mapa del mercado.** "Top 50" y "Top 100" solo eligen qué cartas se ven primero: las que tienen señal y las que más
se mueven, y después Market Score, Risk Score y precio. No cambian ninguna puntuación.

**Límites.** Ninguna señal es una certeza. Las noticias, filtraciones y rumores se añaden a mano en
`data/fc27_analyst.json`.
            """
        )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------


def main() -> None:
    with st.sidebar:
        st.markdown("<div class='fc-side-label'>Datos de mercado</div>", unsafe_allow_html=True)
        refresh = st.button("Actualizar ahora", icon=":material/refresh:", type="primary", width="stretch")
        auto = st.toggle("Actualizar cada 10 min", value=False,
                         help="Vuelve a descargar FUT.GG cada 10 minutos mientras la página esté abierta.")
        st.markdown("<div class='fc-side-label'>Presupuesto</div>", unsafe_allow_html=True)
        budget = st.selectbox("Presupuesto", list(BUDGETS), key="fc_budget", label_visibility="collapsed",
                              help="Precio máximo por carta. Filtra el resumen, las señales y el mercado.")
        st.markdown("<div class='fc-side-foot'>Fuente: FUT.GG<br>Precios de mercado: consola</div>",
                    unsafe_allow_html=True)
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

        # Solo para mostrar: la imagen y el nombre corto de cada carta (si FUT.GG los publica) viajan con su fila.
        all_signals = all_signals.assign(image=all_signals["ea_id"].map(_by_card(snap.movers, "card_image")),
                                         short_name=all_signals["ea_id"].map(_by_card(snap.movers, "card_name")))
        is_fodder = all_signals["key"].str.startswith("fodder")
        signals = pd.concat([all_signals[is_fodder], _in_budget(all_signals[~is_fodder], budget)])
        tone = fc27_signals.overall_tone(snap.movers)

        render_header(conn, snap, live, now)
        if not live:
            _banner("warn", f"FUT.GG no respondió. Estás viendo los datos guardados {_age_text(now)}.")
        elif snap.errors:
            _banner("warn", "Algunas páginas de FUT.GG fallaron; los datos pueden estar incompletos.")
        render_today(signals, events, now, tone)
        render_status(snap, signals, tone)
        render_headline_alerts(alerts, followed)
        render_since_last_visit(conn, snap, alerts, st.session_state.get("fc_prev_visit"))

        section = render_nav()
        if section == "signals":
            render_signals(conn, signals, evaluated, analyst, now, followed)
        elif section == "market":
            render_market(conn, signals, followed, analyst)
        elif section == "watchlist":
            render_watchlist(conn, all_signals, analyst)
        elif section == "trades":
            render_trades(conn, all_signals)
        elif section == "fodder":
            render_fodder(snap, conn)
        elif section == "alerts":
            render_alerts(alerts, analyst, now)
        else:
            render_help(evaluated, snap)

    body()


main()
