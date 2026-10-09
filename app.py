"""Punto de entrada de la app: configuración común y navegación entre páginas.

Ejecutar con:
    streamlit run app.py

Cada página vive en `views/`. La navegación usa `st.navigation`, así cada
página tiene nombre e icono propios en la barra lateral (antes el archivo de
entrada aparecía como "app").
"""

from __future__ import annotations

import importlib
import os
import sys
import time

import streamlit as st

# Streamlit Community Cloud puede traer código nuevo sin reiniciar el proceso: las páginas se
# releen en cada ejecución, pero los módulos de src/ ya importados seguirían en su versión
# vieja (y la página fallaría con AttributeError o ImportError). Se recargan, en orden de
# dependencias, los que cambiaron en disco desde que se cargaron.
for _name in ("src.utils", "src.fc27_market", "src.fc27_history", "src.fc27_signals", "src.match_model",
              "src.understat_source", "src.espn_source", "src.football_data_source", "src.market_signal",
              "src.lineups", "src.competitions", "src.backtest", "src.performance", "src.ui_theme"):
    _module = sys.modules.get(_name)
    if _module is not None and os.path.getmtime(_module.__file__) > getattr(_module, "_loaded_at", 0):
        importlib.reload(_module)
        _module._loaded_at = time.time()

from src import ui_theme  # noqa: E402  (después de recargar los módulos de src/)

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="auto",  # abierta en escritorio, cerrada en móvil
)
st.markdown(ui_theme.inject_global_css(), unsafe_allow_html=True)
with st.container(key="fc_theme"):  # botón de modo claro/oscuro, fijo arriba a la derecha
    st.html(ui_theme.theme_toggle_html(), unsafe_allow_javascript=True)

pages = {
    "Fútbol real": [
        st.Page("views/partidos_del_dia.py", title="Partidos del día", icon="📅", url_path="partidos", default=True),
        st.Page("views/predictor.py", title="Predictor de partidos", icon="⚽", url_path="predictor"),
    ],
    "FC 27 Ultimate Team": [
        st.Page("views/fc27_mercado.py", title="Mercado FC 27", icon="📈", url_path="fc27"),
    ],
}

st.navigation(pages, position="sidebar").run()
