"""Construcción de gráficos interactivos con Plotly para el dashboard.

Todas las funciones devuelven un objeto `plotly.graph_objects.Figure` listo
para pasar a `st.plotly_chart`. Ninguna función aquí calcula estadísticas:
solo recibe datos ya calculados (por `statistics.py` / `poisson_model.py`)
y los dibuja.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_HOME = "#2E7D32"  # verde (local)
COLOR_AWAY = "#C62828"  # rojo (visitante)
COLOR_DRAW = "#9E9E9E"  # gris (empate)

TEMPLATE = "plotly_white"


def bar_comparison_chart(
    home_team: str,
    away_team: str,
    metrics: dict[str, tuple[float | None, float | None]],
    title: str = "Comparación de métricas",
) -> go.Figure:
    """Gráfico de barras agrupadas comparando dos equipos en varias métricas.

    `metrics` = {"Goles por partido": (1.8, 1.2), "% Victorias": (55.0, 40.0), ...}
    Los valores None se dibujan como 0 con una nota, ya que Plotly no
    admite huecos parciales en barras agrupadas sin distorsionar la escala.
    """
    labels = list(metrics.keys())
    home_values = [v[0] if v[0] is not None else 0 for v in metrics.values()]
    away_values = [v[1] if v[1] is not None else 0 for v in metrics.values()]

    fig = go.Figure()
    fig.add_bar(name=home_team, x=labels, y=home_values, marker_color=COLOR_HOME)
    fig.add_bar(name=away_team, x=labels, y=away_values, marker_color=COLOR_AWAY)
    fig.update_layout(
        title=title,
        barmode="group",
        template=TEMPLATE,
        legend_title_text="Equipo",
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def radar_comparison_chart(
    home_team: str,
    away_team: str,
    metrics: dict[str, tuple[float, float]],
) -> go.Figure:
    """Radar de rendimiento. Se espera que los valores ya vengan
    normalizados a una escala comparable (por ejemplo 0-100)."""
    categories = list(metrics.keys())
    home_values = [v[0] for v in metrics.values()]
    away_values = [v[1] for v in metrics.values()]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=home_values + [home_values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=home_team,
            line_color=COLOR_HOME,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=away_values + [away_values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=away_team,
            line_color=COLOR_AWAY,
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template=TEMPLATE,
        title="Radar de rendimiento (escala 0-100)",
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig


def form_evolution_chart(team: str, dates: list, results: list[str]) -> go.Figure:
    """Evolución de puntos acumulados en los últimos partidos.

    `results` debe venir en orden cronológico ascendente (más antiguo
    primero) para que la línea se lea de izquierda (pasado) a derecha
    (presente).
    """
    points_map = {"W": 3, "D": 1, "L": 0}
    cumulative = []
    total = 0
    for r in results:
        total += points_map.get(r, 0)
        cumulative.append(total)

    x_labels = [str(d)[:10] if d is not None else str(i + 1) for i, d in enumerate(dates)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=cumulative,
            mode="lines+markers+text",
            text=results,
            textposition="top center",
            line=dict(color=COLOR_HOME, width=2),
            marker=dict(size=10),
        )
    )
    fig.update_layout(
        title=f"Evolución de la forma reciente — {team}",
        xaxis_title="Partido",
        yaxis_title="Puntos acumulados",
        template=TEMPLATE,
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def goals_per_match_chart(
    home_team: str,
    away_team: str,
    home_gf: float | None,
    home_ga: float | None,
    away_gf: float | None,
    away_ga: float | None,
) -> go.Figure:
    """Comparación de goles anotados vs. recibidos por partido, por equipo."""
    fig = go.Figure()
    fig.add_bar(
        name="Goles anotados/partido",
        x=[home_team, away_team],
        y=[home_gf or 0, away_gf or 0],
        marker_color="#1565C0",
    )
    fig.add_bar(
        name="Goles recibidos/partido",
        x=[home_team, away_team],
        y=[home_ga or 0, away_ga or 0],
        marker_color="#EF6C00",
    )
    fig.update_layout(
        title="Goles por partido",
        barmode="group",
        template=TEMPLATE,
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def result_distribution_pie(home_pct: float | None, draw_pct: float | None, away_pct: float | None) -> go.Figure:
    """Distribución de resultados (victoria local / empate / victoria
    visitante) en todo el conjunto de datos."""
    labels = ["Victoria local", "Empate", "Victoria visitante"]
    values = [home_pct or 0, draw_pct or 0, away_pct or 0]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=[COLOR_HOME, COLOR_DRAW, COLOR_AWAY],
                hole=0.4,
            )
        ]
    )
    fig.update_layout(title="Distribución de resultados en el dataset", template=TEMPLATE)
    return fig


def team_results_pie(team: str, win_pct: float | None, draw_pct: float | None, loss_pct: float | None) -> go.Figure:
    """Distribución de Victorias/Empates/Derrotas de un equipo específico."""
    labels = ["Victorias", "Empates", "Derrotas"]
    values = [win_pct or 0, draw_pct or 0, loss_pct or 0]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=[COLOR_HOME, COLOR_DRAW, COLOR_AWAY],
                hole=0.4,
            )
        ]
    )
    fig.update_layout(title=f"Distribución de resultados — {team}", template=TEMPLATE)
    return fig


def score_matrix_heatmap(matrix: pd.DataFrame, home_team: str, away_team: str) -> go.Figure:
    """Mapa de calor de la matriz de probabilidades de marcador (Poisson).

    `matrix` tiene los goles del local en las filas y del visitante en las
    columnas, con valores ya en porcentaje (0-100).
    """
    fig = px.imshow(
        matrix,
        labels=dict(x=f"Goles {away_team}", y=f"Goles {home_team}", color="Probabilidad (%)"),
        x=matrix.columns,
        y=matrix.index,
        color_continuous_scale="Greens",
        text_auto=".1f",
        aspect="auto",
    )
    fig.update_layout(
        title=f"Matriz de marcadores probables — {home_team} vs {away_team}",
        template=TEMPLATE,
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def feature_importance_bar(importance: dict[str, float], labels_map: dict[str, str], top_n: int = 12) -> go.Figure:
    """Barra horizontal con las variables más influyentes del modelo de
    regresión logística (magnitud promedio de coeficientes estandarizados)."""
    items = list(importance.items())[:top_n][::-1]
    labels = [labels_map.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#1565C0"))
    fig.update_layout(
        title="Importancia de variables (regresión logística)",
        xaxis_title="Magnitud del coeficiente (estandarizado)",
        template=TEMPLATE,
        margin=dict(t=60, b=40, l=220, r=20),
    )
    return fig


def confusion_matrix_heatmap(matrix: list[list[int]], labels: list[str], title: str) -> go.Figure:
    """Mapa de calor de una matriz de confusión (filas = real, columnas = predicho)."""
    df = pd.DataFrame(matrix, index=[f"Real: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])
    fig = px.imshow(df, text_auto=True, color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(title=title, template=TEMPLATE, margin=dict(t=60, b=40, l=40, r=20))
    return fig


def top_scores_bar(top_scores: list[tuple[str, float]]) -> go.Figure:
    """Barra horizontal con los marcadores exactos más probables."""
    scores = [s for s, _ in top_scores][::-1]
    probs = [p for _, p in top_scores][::-1]
    fig = go.Figure(go.Bar(x=probs, y=scores, orientation="h", marker_color=COLOR_HOME))
    fig.update_layout(
        title="Marcadores exactos más probables",
        xaxis_title="Probabilidad (%)",
        template=TEMPLATE,
        margin=dict(t=60, b=40, l=60, r=20),
    )
    return fig
