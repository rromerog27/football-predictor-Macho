"""Sistema de diseño del dashboard: paleta, CSS global y componentes visuales
(topbar, hero de bienvenida, tarjetas de estadísticas, card de partido).

Este módulo solo construye HTML/CSS estático a partir de datos ya calculados
por el resto de la app — no contiene lógica de negocio ni cálculos. Las
funciones que devuelven `str` se pasan a `st.markdown(..., unsafe_allow_html=True)`;
las que necesitan interactividad real (contenedores de la barra lateral) usan
los primitivos nativos de Streamlit directamente en `app.py`.

Paleta: azul, verde, blanco y gris (pedido explícito del usuario), con roles
fijos (fondo, tinta primaria/secundaria, bordes) para que todo el dashboard
se sienta consistente. Las tarjetas usan un fondo claro explícito a propósito
—no reaccionan al tema Light/Dark de Streamlit— porque así se leen igual de
bien sobre cualquiera de los dos fondos, en vez de intentar adivinar el
contraste correcto para cada combinación.
"""

from __future__ import annotations

import html

# --------------------------------------------------------------------------
# Paleta (azul / verde / blanco / gris)
# --------------------------------------------------------------------------

BLUE = "#1D4ED8"
BLUE_DARK = "#1E3A8A"
BLUE_LIGHT = "#DBEAFE"
GREEN = "#16A34A"
GREEN_DARK = "#15803D"
GREEN_LIGHT = "#DCFCE7"
INK_900 = "#0F172A"
INK_600 = "#475569"
INK_400 = "#94A3B8"
GRAY_50 = "#F8FAFC"
GRAY_100 = "#F1F5F9"
GRAY_200 = "#E2E8F0"
WHITE = "#FFFFFF"


def inject_global_css() -> str:
    """CSS global: chrome de Streamlit, tipografía, tabs, botones y las
    clases `.fp-*` que usan los componentes de este módulo."""
    return f"""
    <style>
    /* -- Chrome de Streamlit: oculta el footer y el botón Deploy; deja el
       menú hamburguesa visible (ahí vive el selector de tema Light/Dark). -- */
    footer {{ visibility: hidden; }}
    [data-testid="stAppDeployButton"] {{ display: none; }}

    /* -- Tipografía base -- */
    html, body, [class*="css"] {{
        font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    }}

    /* -- Tarjetas nativas st.metric: mismo lenguaje visual que las nuevas -- */
    div[data-testid="stMetric"] {{
        background-color: {WHITE};
        border: 1px solid {GRAY_200};
        border-radius: 14px;
        padding: 14px 18px 10px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{ color: {INK_600} !important; }}
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] p {{ color: {INK_900} !important; font-weight: 700; }}

    /* -- Botones primarios (Ejecutar análisis, etc.) -- */
    button[kind="primary"] {{
        background: linear-gradient(135deg, {BLUE} 0%, {GREEN} 100%) !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(29, 78, 216, 0.25);
    }}

    /* -- Navegación superior (st.tabs) en estilo píldora -- */
    div[data-testid="stTabs"] div[role="tablist"] {{
        gap: 4px;
        border-bottom: 1px solid {GRAY_200};
    }}
    div[data-testid="stTabs"] div[data-testid="stTab"] {{
        height: 44px;
        border-radius: 10px 10px 0 0;
        padding: 0 18px;
        font-weight: 600;
        color: {INK_600};
    }}
    div[data-testid="stTabs"] div[data-testid="stTab"][aria-selected="true"] {{
        color: {BLUE};
        background-color: {BLUE_LIGHT};
        box-shadow: inset 0 -3px 0 0 {BLUE};
    }}
    div[data-testid="stTabs"] div[data-testid="stTab"] p {{
        font-size: 14.5px;
    }}

    /* -- Bloques de la barra lateral (containers con borde) -- */
    section[data-testid="stSidebar"] div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"] {{
        border-radius: 12px !important;
        border-color: {GRAY_200} !important;
        background-color: rgba(29, 78, 216, 0.02);
        margin-bottom: 14px;
    }}
    .fp-sidebar-block-title {{
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .04em;
        color: {INK_600};
        margin: 0 0 8px 0;
    }}

    .confidence-badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    .confidence-alta {{ background-color: {GREEN_LIGHT}; color: {GREEN_DARK}; }}
    .confidence-media {{ background-color: #FFF8E1; color: #E65100; }}
    .confidence-baja {{ background-color: #FFEBEE; color: #B71C1C; }}

    /* -- Topbar de marca -- */
    .fp-topbar {{ display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }}
    .fp-topbar-badge {{
        width: 52px; height: 52px; border-radius: 14px; flex-shrink: 0;
        background: linear-gradient(135deg, {BLUE} 0%, {GREEN} 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 26px; box-shadow: 0 4px 12px rgba(29, 78, 216, 0.25);
    }}
    .fp-topbar-title {{ font-size: 28px; font-weight: 800; color: {INK_900}; line-height: 1.15; }}
    .fp-topbar-subtitle {{ font-size: 14px; color: {INK_600}; margin-top: 2px; }}

    /* -- Tarjetas de estadísticas (KPI row) -- */
    .fp-stat-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 14px 0 22px 0; }}
    .fp-stat-card {{
        flex: 1 1 200px;
        background: {WHITE};
        border: 1px solid {GRAY_200};
        border-radius: 14px;
        padding: 16px 18px;
        display: flex; align-items: center; gap: 14px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }}
    .fp-stat-icon {{
        width: 44px; height: 44px; border-radius: 12px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center; font-size: 21px;
    }}
    .fp-accent-blue .fp-stat-icon {{ background: {BLUE_LIGHT}; color: {BLUE}; }}
    .fp-accent-green .fp-stat-icon {{ background: {GREEN_LIGHT}; color: {GREEN_DARK}; }}
    .fp-accent-gray .fp-stat-icon {{ background: {GRAY_100}; color: {INK_600}; }}
    .fp-accent-amber .fp-stat-icon {{ background: #FEF3C7; color: #B45309; }}
    .fp-stat-value {{ font-size: 25px; font-weight: 700; color: {INK_900}; line-height: 1.1; }}
    .fp-stat-label {{ font-size: 12.5px; color: {INK_600}; margin-top: 2px; }}

    /* -- Card visual del partido seleccionado -- */
    .fp-match-card {{
        display: flex; align-items: center; justify-content: space-between;
        background: linear-gradient(135deg, {BLUE_LIGHT} 0%, {GREEN_LIGHT} 100%);
        border: 1px solid {GRAY_200}; border-radius: 16px;
        padding: 18px 26px; margin: 6px 0 20px 0;
    }}
    .fp-match-team {{ flex: 1; text-align: center; min-width: 0; }}
    .fp-match-team-badge {{ font-size: 26px; margin-bottom: 2px; }}
    .fp-match-team-name {{
        font-size: 18px; font-weight: 700; color: {INK_900};
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}
    .fp-match-team-role {{
        font-size: 11px; color: {INK_600}; text-transform: uppercase;
        letter-spacing: .05em; margin-top: 2px;
    }}
    .fp-match-vs {{
        flex: 0 0 auto; width: 40px; height: 40px; border-radius: 50%;
        background: {BLUE}; color: {WHITE}; font-weight: 700; font-size: 12px;
        display: flex; align-items: center; justify-content: center; margin: 0 16px;
        box-shadow: 0 2px 8px rgba(29, 78, 216, 0.35);
    }}

    /* -- Hero de bienvenida (estado vacío) -- */
    .fp-hero {{
        display: flex; align-items: center; gap: 36px; flex-wrap: wrap;
        padding: 8px 4px 26px 4px;
    }}
    .fp-hero-copy {{ flex: 1 1 340px; min-width: 280px; }}
    .fp-hero-art {{ flex: 1 1 320px; min-width: 260px; max-width: 420px; }}
    .fp-hero-art svg {{ width: 100%; height: auto; display: block; }}
    .fp-pill {{
        display: inline-block; background: {BLUE_LIGHT}; color: {BLUE_DARK};
        font-size: 13px; font-weight: 600; padding: 5px 14px; border-radius: 999px;
        margin-bottom: 14px;
    }}
    .fp-hero-title {{
        font-size: 32px; font-weight: 800; color: {INK_900}; line-height: 1.25;
        margin: 0 0 12px 0;
    }}
    .fp-hero-text {{ font-size: 15px; color: {INK_600}; line-height: 1.6; max-width: 46ch; }}

    .fp-steps {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 6px; }}
    .fp-step {{
        flex: 1 1 220px; display: flex; gap: 14px;
        background: {WHITE}; border: 1px solid {GRAY_200}; border-radius: 14px;
        padding: 16px 18px;
    }}
    .fp-step-num {{
        flex-shrink: 0; width: 30px; height: 30px; border-radius: 50%;
        background: {GREEN_LIGHT}; color: {GREEN_DARK}; font-weight: 700; font-size: 14px;
        display: flex; align-items: center; justify-content: center;
    }}
    .fp-step-title {{ font-size: 14.5px; font-weight: 700; color: {INK_900}; }}
    .fp-step-text {{ font-size: 13px; color: {INK_600}; margin-top: 2px; line-height: 1.45; }}
    </style>
    """


def render_topbar(subtitle: str | None = None) -> str:
    """Encabezado de marca (icono + título + subtítulo)."""
    subtitle = subtitle or "Estadística aplicada al fútbol y predicciones basadas en tus propios datos"
    return f"""
    <div class="fp-topbar">
        <div class="fp-topbar-badge">⚽</div>
        <div>
            <div class="fp-topbar-title">Football Predictor</div>
            <div class="fp-topbar-subtitle">{html.escape(subtitle)}</div>
        </div>
    </div>
    """


def stat_card(icon: str, label: str, value: str, accent: str = "blue") -> str:
    """Una tarjeta de estadística individual (para armar con `stat_row`)."""
    return f"""
    <div class="fp-stat-card fp-accent-{accent}">
        <div class="fp-stat-icon">{icon}</div>
        <div>
            <div class="fp-stat-value">{html.escape(str(value))}</div>
            <div class="fp-stat-label">{html.escape(label)}</div>
        </div>
    </div>
    """


def stat_row(cards: list[dict]) -> str:
    """Fila de tarjetas de estadísticas (KPI row). Cada elemento de `cards`:
    {"icon": "📁", "label": "Archivos cargados", "value": "3", "accent": "blue"}."""
    cards_html = "".join(
        stat_card(c["icon"], c["label"], c["value"], c.get("accent", "blue")) for c in cards
    )
    return f'<div class="fp-stat-row">{cards_html}</div>'


def render_match_card(home_team: str, away_team: str) -> str:
    """Card visual del partido seleccionado (local vs. visitante)."""
    return f"""
    <div class="fp-match-card">
        <div class="fp-match-team">
            <div class="fp-match-team-badge">🏠</div>
            <div class="fp-match-team-name">{html.escape(home_team)}</div>
            <div class="fp-match-team-role">Local</div>
        </div>
        <div class="fp-match-vs">VS</div>
        <div class="fp-match-team">
            <div class="fp-match-team-badge">🚌</div>
            <div class="fp-match-team-name">{html.escape(away_team)}</div>
            <div class="fp-match-team-role">Visitante</div>
        </div>
    </div>
    """


def _hero_illustration_svg() -> str:
    """Ilustración inline (sin recursos externos): cancha de fútbol con una
    línea de tendencia de datos superpuesta, para transmitir "análisis
    estadístico aplicado al fútbol" sin depender de imágenes externas."""
    return f"""
    <svg viewBox="0 0 480 320" xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="Ilustración de una cancha de fútbol con una línea de tendencia de datos">
        <rect x="0" y="0" width="480" height="320" rx="24" fill="{GREEN_LIGHT}"/>
        <rect x="40" y="40" width="400" height="240" rx="14" fill="{GREEN}"/>
        <rect x="52" y="52" width="376" height="216" rx="8" fill="none" stroke="{WHITE}" stroke-width="3" opacity="0.85"/>
        <line x1="240" y1="52" x2="240" y2="268" stroke="{WHITE}" stroke-width="3" opacity="0.85"/>
        <circle cx="240" cy="160" r="38" fill="none" stroke="{WHITE}" stroke-width="3" opacity="0.85"/>
        <circle cx="240" cy="160" r="3" fill="{WHITE}" opacity="0.85"/>
        <rect x="52" y="112" width="46" height="96" fill="none" stroke="{WHITE}" stroke-width="3" opacity="0.85"/>
        <rect x="382" y="112" width="46" height="96" fill="none" stroke="{WHITE}" stroke-width="3" opacity="0.85"/>
        <polyline points="70,232 150,192 220,206 290,132 360,152 420,82" fill="none"
                  stroke="{BLUE}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="70" cy="232" r="6" fill="{BLUE}" stroke="{WHITE}" stroke-width="2"/>
        <circle cx="150" cy="192" r="6" fill="{BLUE}" stroke="{WHITE}" stroke-width="2"/>
        <circle cx="220" cy="206" r="6" fill="{BLUE}" stroke="{WHITE}" stroke-width="2"/>
        <circle cx="290" cy="132" r="6" fill="{BLUE}" stroke="{WHITE}" stroke-width="2"/>
        <circle cx="360" cy="152" r="6" fill="{BLUE}" stroke="{WHITE}" stroke-width="2"/>
        <g transform="translate(420,82)">
            <circle r="15" fill="{WHITE}" stroke="{INK_900}" stroke-width="2"/>
            <path d="M0,-15 L4.5,-4.5 L-4.5,-4.5 Z M0,15 L4.5,4.5 L-4.5,4.5 Z M-15,0 L-4.5,4.5 L-4.5,-4.5 Z M15,0 L4.5,4.5 L4.5,-4.5 Z"
                  fill="{INK_900}" opacity="0.75"/>
        </g>
    </svg>
    """


def render_hero_empty_state() -> str:
    """Estado vacío (antes de cargar un archivo): hero ilustrado + copy +
    pasos numerados con instrucciones claras."""
    return f"""
    <div class="fp-hero">
        <div class="fp-hero-copy">
            <span class="fp-pill">📊 Estadística aplicada al fútbol</span>
            <h1 class="fp-hero-title">Convierte datos de partidos en predicciones claras</h1>
            <p class="fp-hero-text">
                Cargá tus archivos de resultados, comparé equipos y generá predicciones con
                modelos estadísticos (Poisson + regresión logística) — todo con tus propios
                datos, sin conexión a internet.
            </p>
        </div>
        <div class="fp-hero-art">{_hero_illustration_svg()}</div>
    </div>
    <div class="fp-steps">
        <div class="fp-step">
            <div class="fp-step-num">1</div>
            <div>
                <div class="fp-step-title">Descargá tus datos</div>
                <div class="fp-step-text">Bajá el CSV de resultados de tu liga (por ejemplo, football-data.co.uk).</div>
            </div>
        </div>
        <div class="fp-step">
            <div class="fp-step-num">2</div>
            <div>
                <div class="fp-step-title">Subilos en la barra lateral</div>
                <div class="fp-step-text">Podés cargar uno o varios archivos (varias temporadas) a la vez.</div>
            </div>
        </div>
        <div class="fp-step">
            <div class="fp-step-num">3</div>
            <div>
                <div class="fp-step-title">Analizá y predecí</div>
                <div class="fp-step-text">Elegí dos equipos y mirá estadísticas, predicción y rendimiento del modelo.</div>
            </div>
        </div>
    </div>
    """


def sidebar_block_title(icon: str, title: str) -> str:
    """Título pequeño en mayúsculas para el encabezado de un bloque de la
    barra lateral (se usa dentro de un `st.sidebar.container(border=True)`)."""
    return f'<div class="fp-sidebar-block-title">{icon} {html.escape(title)}</div>'
