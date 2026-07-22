"""Generación de reportes y exportaciones (CSV, Excel, HTML).

Este módulo solo formatea y exporta datos que ya fueron calculados por
`statistics.py`, `poisson_model.py` y `prediction_model.py`: no vuelve a
calcular estadísticas ni inventa información. Todas las funciones devuelven
`bytes` listos para `st.download_button`, salvo `build_html_report`, que
devuelve un string HTML autocontenido (sin recursos externos).
"""

from __future__ import annotations

import html
import io
from datetime import datetime

import pandas as pd

from src import statistics
from src.utils import get_logger

logger = get_logger(__name__)

REPORT_DATE_FORMAT = "%Y-%m-%d %H:%M"


def now_str() -> str:
    return datetime.now().strftime(REPORT_DATE_FORMAT)


# --------------------------------------------------------------------------
# Exportaciones CSV
# --------------------------------------------------------------------------


def stats_table_to_csv(historical_df: pd.DataFrame, recent_n: int = 5) -> bytes:
    """Tabla con las estadísticas generales de todos los equipos detectados
    en el archivo cargado (una fila por equipo)."""
    teams = statistics.get_team_list(historical_df)
    rows = []
    for team in teams:
        profile = statistics.build_team_profile(historical_df, team, recent_n)
        r = profile.overall
        rows.append(
            {
                "Equipo": team,
                "Partidos jugados": r.played,
                "Victorias": r.wins,
                "Empates": r.draws,
                "Derrotas": r.losses,
                "% Victorias": r.win_pct,
                "Goles anotados": r.goals_for,
                "Goles recibidos": r.goals_against,
                "Goles anotados/partido": r.avg_goals_for,
                "Goles recibidos/partido": r.avg_goals_against,
                "Diferencia de goles": r.goal_difference,
                "% Porterías a cero": r.clean_sheet_pct,
                "% Ambos anotan": r.both_teams_scored_pct,
                "% Más de 2.5 goles": r.over_pct.get(2.5),
                f"Forma (últimos {recent_n})": "".join(profile.recent_form.results),
            }
        )
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8-sig")


def prediction_to_csv(record: dict) -> bytes:
    """Predicción de un partido como una única fila CSV (mismo formato que
    el historial, para que el usuario pueda combinarlas manualmente)."""
    df = pd.DataFrame([record])
    return df.to_csv(index=False).encode("utf-8-sig")


def score_matrix_to_csv(matrix: pd.DataFrame) -> bytes:
    """Matriz de marcadores (probabilidades en %) como archivo CSV, con los
    goles del local como índice de filas y los del visitante como columnas."""
    return matrix.to_csv(index_label="Goles local \\ Goles visitante").encode("utf-8-sig")


def variables_summary_to_csv(
    mapping: dict[str, str | None],
    field_labels: dict[str, str],
    feature_importance: dict[str, float] | None = None,
) -> bytes:
    """Resumen de qué columna del archivo se usó para cada variable, y
    (cuando está disponible) la importancia de cada variable en el modelo
    de regresión logística."""
    rows = [
        {"Variable": field_labels.get(field, field), "Columna en el archivo": col or "No disponible"}
        for field, col in mapping.items()
    ]
    df = pd.DataFrame(rows)

    if feature_importance:
        importance_df = pd.DataFrame(
            [{"Variable del modelo": k, "Importancia (coef. estandarizado)": v} for k, v in feature_importance.items()]
        )
        buffer = io.StringIO()
        buffer.write("Mapeo de columnas del archivo\n")
        df.to_csv(buffer, index=False)
        buffer.write("\nImportancia de variables (regresión logística)\n")
        importance_df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8-sig")

    return df.to_csv(index=False).encode("utf-8-sig")


# --------------------------------------------------------------------------
# Exportación Excel (comparación de equipos)
# --------------------------------------------------------------------------


def record_rows(home_team: str, away_team: str, home_r: statistics.TeamRecord, away_r: statistics.TeamRecord) -> list[dict]:
    return [
        {"Métrica": "Partidos jugados", home_team: home_r.played, away_team: away_r.played},
        {"Métrica": "Victorias", home_team: home_r.wins, away_team: away_r.wins},
        {"Métrica": "Empates", home_team: home_r.draws, away_team: away_r.draws},
        {"Métrica": "Derrotas", home_team: home_r.losses, away_team: away_r.losses},
        {"Métrica": "% Victorias", home_team: home_r.win_pct, away_team: away_r.win_pct},
        {"Métrica": "Goles anotados/partido", home_team: home_r.avg_goals_for, away_team: away_r.avg_goals_for},
        {"Métrica": "Goles recibidos/partido", home_team: home_r.avg_goals_against, away_team: away_r.avg_goals_against},
        {"Métrica": "% Porterías a cero", home_team: home_r.clean_sheet_pct, away_team: away_r.clean_sheet_pct},
        {"Métrica": "% Ambos anotan", home_team: home_r.both_teams_scored_pct, away_team: away_r.both_teams_scored_pct},
        {"Métrica": "% Más de 2.5 goles", home_team: home_r.over_pct.get(2.5), away_team: away_r.over_pct.get(2.5)},
    ]


def comparison_to_excel(
    home_team: str,
    away_team: str,
    home_profile: statistics.TeamProfile,
    away_profile: statistics.TeamProfile,
) -> bytes:
    """Comparación de dos equipos exportada como Excel con varias hojas:
    general, como local, como visitante y forma reciente."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(record_rows(home_team, away_team, home_profile.overall, away_profile.overall)).to_excel(
            writer, sheet_name="General", index=False
        )
        pd.DataFrame(record_rows(home_team, away_team, home_profile.as_home, away_profile.as_home)).to_excel(
            writer, sheet_name="Como local", index=False
        )
        pd.DataFrame(record_rows(home_team, away_team, home_profile.as_away, away_profile.as_away)).to_excel(
            writer, sheet_name="Como visitante", index=False
        )

        form_rows = [
            {
                "Equipo": home_team,
                "Partidos considerados": home_profile.recent_form.n_available,
                "Forma": "".join(home_profile.recent_form.results),
                "Puntos/partido": home_profile.recent_form.points_per_game,
                "Goles anotados/partido (reciente)": home_profile.recent_form.avg_goals_for,
                "Goles recibidos/partido (reciente)": home_profile.recent_form.avg_goals_against,
            },
            {
                "Equipo": away_team,
                "Partidos considerados": away_profile.recent_form.n_available,
                "Forma": "".join(away_profile.recent_form.results),
                "Puntos/partido": away_profile.recent_form.points_per_game,
                "Goles anotados/partido (reciente)": away_profile.recent_form.avg_goals_for,
                "Goles recibidos/partido (reciente)": away_profile.recent_form.avg_goals_against,
            },
        ]
        pd.DataFrame(form_rows).to_excel(writer, sheet_name="Forma reciente", index=False)

    buffer.seek(0)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Reporte HTML completo
# --------------------------------------------------------------------------


def _matrix_to_html_table(matrix: pd.DataFrame, home_team: str, away_team: str) -> str:
    max_val = float(matrix.values.max()) if matrix.values.size else 1.0
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in matrix.columns)
    body_rows = []
    for row_label, row in matrix.iterrows():
        cells = []
        for val in row:
            intensity = min(val / max_val, 1.0) if max_val > 0 else 0
            color = f"rgba(46, 125, 50, {0.08 + intensity * 0.75:.2f})"
            cells.append(f'<td style="background-color:{color};text-align:center;">{val:.1f}%</td>')
        body_rows.append(f"<tr><th>{row_label}</th>{''.join(cells)}</tr>")
    return (
        f'<p class="caption">Filas: goles de {html.escape(home_team)} — Columnas: goles de {html.escape(away_team)}</p>'
        f'<table class="matrix"><thead><tr><th></th>{header}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _dict_rows_to_html_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>Sin datos.</p>"
    headers = list(rows[0].keys())
    header_html = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def build_html_report(context: dict) -> str:
    """Construye el reporte HTML completo del análisis.

    `context` debe incluir (las claves ausentes se muestran como "N/D"):
    file_name, analysis_date, competition, season_labels, n_matches,
    period, home_team, away_team, model_choice, prob_home_win, prob_draw,
    prob_away_win, expected_home_goals, expected_away_goals, top_score,
    confidence, explanation, warnings (list[str]), comparison_rows
    (list[dict]), score_matrix (DataFrame | None).
    """
    g = lambda k, default="N/D": context.get(k, default)  # noqa: E731

    warnings_html = (
        "<ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in context.get("warnings", [])) + "</ul>"
        if context.get("warnings")
        else "<p>Sin advertencias.</p>"
    )

    matrix = context.get("score_matrix")
    matrix_html = _matrix_to_html_table(matrix, g("home_team"), g("away_team")) if matrix is not None else "<p>No disponible para este modelo.</p>"

    comparison_html = _dict_rows_to_html_table(context.get("comparison_rows", []))

    return f"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte — {html.escape(g('home_team'))} vs {html.escape(g('away_team'))}</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; color: #1a1a1a; max-width: 900px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ color: #1B5E20; }}
  h2 {{ color: #1B5E20; border-bottom: 2px solid #E8F5E9; padding-bottom: 4px; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; font-size: 0.9rem; }}
  th {{ background-color: #F1F8E9; text-align: left; }}
  .meta {{ color: #555; font-size: 0.9rem; }}
  .caption {{ color: #555; font-size: 0.85rem; margin-bottom: 4px; }}
  .disclaimer {{ background-color: #FFF8E1; border: 1px solid #FFE082; padding: 10px 14px; border-radius: 6px; margin-top: 32px; font-size: 0.9rem; }}
  table.matrix th, table.matrix td {{ text-align: center; }}
</style>
</head>
<body>
  <h1>⚽ Football Predictor — Reporte de análisis</h1>
  <p class="meta">
    Generado el {html.escape(g('analysis_date'))} · Archivo: {html.escape(g('file_name'))} ·
    Competición: {html.escape(g('competition'))} · Temporada(s): {html.escape(g('season_labels'))}
  </p>

  <h2>Resumen del conjunto de datos</h2>
  <p>Partidos analizados: {html.escape(str(g('n_matches')))} · Periodo: {html.escape(g('period'))}</p>

  <h2>Comparación: {html.escape(g('home_team'))} vs {html.escape(g('away_team'))}</h2>
  {comparison_html}

  <h2>Predicción ({html.escape(g('model_choice'))})</h2>
  <p>
    Victoria {html.escape(g('home_team'))}: <b>{g('prob_home_win')}%</b> ·
    Empate: <b>{g('prob_draw')}%</b> ·
    Victoria {html.escape(g('away_team'))}: <b>{g('prob_away_win')}%</b>
  </p>
  <p>
    Goles esperados: {html.escape(g('home_team'))} {g('expected_home_goals')} — {g('expected_away_goals')} {html.escape(g('away_team'))} ·
    Marcador más probable: {html.escape(str(g('top_score')))} · Nivel de confianza: {html.escape(g('confidence'))}
  </p>
  <p>{html.escape(g('explanation', ''))}</p>

  <h2>Matriz de marcadores</h2>
  {matrix_html}

  <h2>Advertencias</h2>
  {warnings_html}

  <p class="disclaimer">
    Este reporte se generó únicamente a partir de los datos del archivo cargado.
    Ninguna probabilidad mostrada garantiza el resultado real del partido.
  </p>
</body>
</html>
"""
