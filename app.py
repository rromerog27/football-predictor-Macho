"""Punto de entrada de la app: configuración común y navegación entre páginas.

Ejecutar con:
    streamlit run app.py

Cada página vive en `views/`. La navegación usa `st.navigation`, así cada
página tiene nombre e icono propios en la barra lateral (antes el archivo de
entrada aparecía como "app").
"""

from __future__ import annotations

import streamlit as st

from src import ui_theme

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="auto",  # abierta en escritorio, cerrada en móvil
)
st.markdown(ui_theme.inject_global_css(), unsafe_allow_html=True)

pages = {
    "FC 27 Ultimate Team": [
        st.Page("views/fc27_mercado.py", title="Mercado FC 27", icon="📈", url_path="fc27", default=True),
    ],
    "Fútbol real": [
        st.Page("views/predictor.py", title="Predictor de partidos", icon="⚽", url_path="predictor"),
    ],
}

st.navigation(pages, position="sidebar").run()
