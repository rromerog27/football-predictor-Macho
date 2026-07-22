"""Football Predictor — Fase 2.

Dashboard en Streamlit para cargar partidos de fútbol (Excel o CSV),
analizar el rendimiento de los equipos y generar predicciones estadísticas
mediante un modelo de distribución de Poisson y, cuando hay datos
suficientes, un modelo de respaldo de regresión logística calibrada.

Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import (
    column_mapper,
    data_cleaner,
    data_loader,
    feature_engineering,
    market_odds,
    poisson_model,
    prediction_history,
    prediction_model,
    report_generator,
    statistics,
    visualizations,
)
from src.utils import MIN_MATCHES_FOR_ML_MODEL, MIN_MATCHES_RELIABLE, format_metric, format_pct, get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Oculta el botón "Deploy" y el pie de página "Made with Streamlit"
    para que se vea más como una app propia. El menú hamburguesa (☰) se
    deja visible a propósito: ahí vive el selector de tema Light/Dark. */
    footer { visibility: hidden; }
    [data-testid="stAppDeployButton"] { display: none; }
    div[data-testid="stMetric"] {
        background-color: rgba(46, 125, 50, 0.06);
        border: 1px solid rgba(46, 125, 50, 0.15);
        border-radius: 10px;
        padding: 12px 14px 8px 14px;
    }
    .confidence-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .confidence-alta { background-color: #E8F5E9; color: #1B5E20; }
    .confidence-media { background-color: #FFF8E1; color: #E65100; }
    .confidence-baja { background-color: #FFEBEE; color: #B71C1C; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Entrenamiento y validación de modelos (cacheado por contenido del dataset)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _train_logistic_cached(df: pd.DataFrame, recent_n: int) -> prediction_model.LogisticModelResult:
    return prediction_model.train_logistic_model(df, recent_n=recent_n)


@st.cache_data(show_spinner=False)
def _poisson_backtest_cached(df: pd.DataFrame) -> poisson_model.BacktestMetrics | None:
    if len(df) < 30:
        return None
    train, test = poisson_model.chronological_split(df, test_fraction=prediction_model.TEST_FRACTION)
    if len(train) < 10 or len(test) < 5:
        return None
    return poisson_model.backtest(train, test)


@st.cache_data(show_spinner=False)
def _poisson_split_cached(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if len(df) < 30:
        return None
    train, test = poisson_model.chronological_split(df, test_fraction=prediction_model.TEST_FRACTION)
    if len(train) < 10 or len(test) < 5:
        return None
    return train, test


@st.cache_data(show_spinner=False)
def _poisson_test_predictions_cached(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict[str, float]]:
    return poisson_model.predict_probs_for_rows(train_df, test_df)


# ---------------------------------------------------------------------------
# Carga de archivos (barra lateral)
# ---------------------------------------------------------------------------


def _load_uploaded_files(uploaded_files) -> list[data_loader.LoadedFile]:
    loaded: list[data_loader.LoadedFile] = []
    for uploaded in uploaded_files:
        try:
            file_type = data_loader.detect_file_type(uploaded.name)
            if file_type == "excel":
                sheet_names = data_loader.get_excel_sheet_names(uploaded)
                chosen_sheet = st.sidebar.selectbox(
                    f"Hoja a analizar — {uploaded.name}",
                    sheet_names,
                    key=f"sheet_{uploaded.name}",
                )
                lf = data_loader.load_file(uploaded, uploaded.name, sheet_name=chosen_sheet)
            else:
                lf = data_loader.load_file(uploaded, uploaded.name)
            loaded.append(lf)
            if lf.dropped_odds_columns:
                st.sidebar.caption(
                    f"'{uploaded.name}': se descartaron {len(lf.dropped_odds_columns)} "
                    "columnas de cuotas de apuestas."
                )
            for w in lf.warnings:
                st.sidebar.warning(f"'{uploaded.name}': {w}")
        except ValueError as exc:
            st.sidebar.error(str(exc))
    return loaded


# ---------------------------------------------------------------------------
# Secciones del dashboard
# ---------------------------------------------------------------------------


def render_inicio(
    loaded_files: list[data_loader.LoadedFile],
    season_labels: list[str],
    competition_value: str | None,
    historical_df: pd.DataFrame,
    cleaning_report: data_cleaner.CleaningReport,
    has_xg: bool,
) -> None:
    st.title("⚽ Football Predictor")
    st.markdown(
        "Aplicación local para analizar partidos de fútbol y generar predicciones "
        "estadísticas (modelo de Poisson y, cuando hay datos suficientes, un modelo de "
        "respaldo de regresión logística) a partir **únicamente** de los archivos que "
        "cargues. No se usa ninguna fuente de datos externa."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Archivos cargados", len(loaded_files))
    col2.metric("Partidos con resultado", cleaning_report.final_historical_rows)
    col3.metric("Partidos pendientes", cleaning_report.pending_rows)

    if cleaning_report.sufficient_data:
        col4.markdown(
            '<span class="confidence-badge confidence-alta">Datos suficientes</span>',
            unsafe_allow_html=True,
        )
    else:
        col4.markdown(
            '<span class="confidence-badge confidence-baja">Datos insuficientes</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    info_cols = st.columns(3)
    info_cols[0].markdown(f"**Competición:** {competition_value or 'No identificada / múltiples'}")
    info_cols[1].markdown(f"**Temporada(s) detectada(s):** {', '.join(season_labels) if season_labels else 'Desconocida'}")
    info_cols[2].markdown(f"**xG en el archivo:** {'Disponible' if has_xg else 'xG no disponible (se usa solo goles reales)'}")

    if not cleaning_report.sufficient_data:
        st.warning(
            f"Hay menos de {MIN_MATCHES_RELIABLE} partidos con resultado disponibles. "
            "Las estadísticas y la predicción se mostrarán igual, pero deben tomarse "
            "con mucha cautela."
        )

    st.info(
        "Configura el equipo local, el equipo visitante y la cantidad de partidos "
        "recientes en la barra lateral, y presiona **Ejecutar análisis** para ver "
        "la comparación y la predicción."
    )


def render_resumen(
    combined_raw: pd.DataFrame,
    historical_df: pd.DataFrame,
    cleaning_report: data_cleaner.CleaningReport,
    recent_n: int,
) -> None:
    st.header("📊 Resumen del conjunto de datos")

    summary = statistics.compute_dataset_summary(historical_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Partidos analizados", summary.n_matches)
    col2.metric("Equipos detectados", summary.n_teams)
    period = (
        f"{summary.period_start.date()} → {summary.period_end.date()}"
        if summary.period_start is not None and summary.period_end is not None
        else "Desconocido"
    )
    col3.metric("Periodo cubierto", period)

    col4, col5 = st.columns(2)
    col4.metric("Promedio de goles por partido", format_metric(summary.avg_goals_per_match))
    col5.plotly_chart(
        visualizations.result_distribution_pie(summary.home_win_pct, summary.draw_pct, summary.away_win_pct),
        use_container_width=True,
    )

    result_cols = st.columns(3)
    result_cols[0].metric("% Victorias locales", format_pct(summary.home_win_pct))
    result_cols[1].metric("% Empates", format_pct(summary.draw_pct))
    result_cols[2].metric("% Victorias visitantes", format_pct(summary.away_win_pct))

    st.markdown("---")
    st.subheader("Calidad de los datos")

    overview = data_loader.dataset_overview(combined_raw)
    qcol1, qcol2, qcol3 = st.columns(3)
    qcol1.metric("Filas originales", overview["n_rows"])
    qcol2.metric("Columnas detectadas", overview["n_columns"])
    qcol3.metric("Filas duplicadas eliminadas", cleaning_report.duplicates_removed)

    with st.expander("Ver columnas detectadas y datos faltantes"):
        st.write("**Columnas:**", ", ".join(str(c) for c in overview["columns"]))
        if overview["missing_by_column"]:
            missing_df = pd.DataFrame(
                [
                    {"Columna": col, "Faltantes": v["count"], "% Faltante": v["pct"]}
                    for col, v in overview["missing_by_column"].items()
                ]
            )
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.write("No se detectaron valores faltantes en el archivo original.")

    if cleaning_report.implausible_value_warnings or cleaning_report.general_warnings:
        with st.expander("⚠ Advertencias de limpieza y validación", expanded=not cleaning_report.sufficient_data):
            for w in cleaning_report.general_warnings:
                st.warning(w)
            for w in cleaning_report.implausible_value_warnings:
                st.warning(w)

    with st.expander("Vista previa de los datos"):
        st.dataframe(combined_raw.head(20), use_container_width=True)

    st.download_button(
        "⬇ Descargar tabla de estadísticas de todos los equipos (CSV)",
        data=report_generator.stats_table_to_csv(historical_df, recent_n),
        file_name="estadisticas_equipos.csv",
        mime="text/csv",
    )


def _team_metric_columns(container, profile: statistics.TeamProfile, stat_flags: dict[str, bool]) -> None:
    r = profile.overall
    container.metric("Partidos jugados", r.played)
    container.metric("Victorias / Empates / Derrotas", f"{r.wins} / {r.draws} / {r.losses}")
    container.metric("% Victorias", format_pct(r.win_pct))
    container.metric("Goles anotados/partido", format_metric(r.avg_goals_for))
    container.metric("Goles recibidos/partido", format_metric(r.avg_goals_against))
    container.metric("Porterías a cero", format_pct(r.clean_sheet_pct))
    container.metric("Ambos anotan", format_pct(r.both_teams_scored_pct))
    container.metric("+2.5 goles", format_pct(r.over_pct.get(2.5)))

    hr = profile.as_home
    ar = profile.as_away
    container.markdown("**Como local:** " + f"{hr.played} PJ, {format_pct(hr.win_pct)} victorias")
    container.markdown("**Como visitante:** " + f"{ar.played} PJ, {format_pct(ar.win_pct)} victorias")

    extra = profile.extra_overall
    extra_labels = {
        "shots": "Tiros/partido",
        "shots_on_target": "Tiros a puerta/partido",
        "corners": "Córners/partido",
        "yellow_cards": "Tarjetas amarillas/partido",
        "red_cards": "Tarjetas rojas/partido",
        "possession": "Posesión (%)",
        "xg": "xG/partido",
    }
    with container.expander("Estadísticas adicionales"):
        for key, label in extra_labels.items():
            if not stat_flags.get(key, False):
                st.write(f"{label}: dato no disponible en el archivo")
            else:
                st.write(f"{label}: {format_metric(extra.get(key))}")

        form = profile.recent_form
        st.write(
            f"**Forma últimos {form.n_requested} partidos** "
            f"({form.n_available} disponibles): {' '.join(form.results) if form.results else '—'}"
        )


def render_comparacion(historical_df: pd.DataFrame, home_team: str, away_team: str, recent_n: int) -> None:
    st.header("🆚 Comparación de equipos")

    home_profile = statistics.build_team_profile(historical_df, home_team, recent_n)
    away_profile = statistics.build_team_profile(historical_df, away_team, recent_n)
    stat_flags = statistics.stat_columns_available(historical_df)

    col_home, col_away = st.columns(2)
    col_home.subheader(home_team)
    col_away.subheader(away_team)
    _team_metric_columns(col_home, home_profile, stat_flags)
    _team_metric_columns(col_away, away_profile, stat_flags)

    st.markdown("---")
    st.subheader("Gráficos comparativos")

    metrics = {
        "% Victorias": (home_profile.overall.win_pct, away_profile.overall.win_pct),
        "Goles anotados/partido": (home_profile.overall.avg_goals_for, away_profile.overall.avg_goals_for),
        "Goles recibidos/partido": (home_profile.overall.avg_goals_against, away_profile.overall.avg_goals_against),
        "Porterías a cero (%)": (home_profile.overall.clean_sheet_pct, away_profile.overall.clean_sheet_pct),
        "Ambos anotan (%)": (home_profile.overall.both_teams_scored_pct, away_profile.overall.both_teams_scored_pct),
        "+2.5 goles (%)": (home_profile.overall.over_pct.get(2.5), away_profile.overall.over_pct.get(2.5)),
    }
    st.plotly_chart(
        visualizations.bar_comparison_chart(home_team, away_team, metrics), use_container_width=True
    )

    def _norm_goals(v: float | None, cap: float = 4.0) -> float:
        return min((v or 0) / cap * 100, 100)

    radar_metrics = {
        "% Victorias": (home_profile.overall.win_pct or 0, away_profile.overall.win_pct or 0),
        "Goles anotados": (_norm_goals(home_profile.overall.avg_goals_for), _norm_goals(away_profile.overall.avg_goals_for)),
        "Solidez defensiva": (
            100 - _norm_goals(home_profile.overall.avg_goals_against),
            100 - _norm_goals(away_profile.overall.avg_goals_against),
        ),
        "Porterías a cero": (home_profile.overall.clean_sheet_pct or 0, away_profile.overall.clean_sheet_pct or 0),
        "Ambos anotan": (home_profile.overall.both_teams_scored_pct or 0, away_profile.overall.both_teams_scored_pct or 0),
    }
    col_a, col_b = st.columns(2)
    col_a.plotly_chart(
        visualizations.radar_comparison_chart(home_team, away_team, radar_metrics), use_container_width=True
    )
    col_b.plotly_chart(
        visualizations.goals_per_match_chart(
            home_team,
            away_team,
            home_profile.overall.avg_goals_for,
            home_profile.overall.avg_goals_against,
            away_profile.overall.avg_goals_for,
            away_profile.overall.avg_goals_against,
        ),
        use_container_width=True,
    )

    col_c, col_d = st.columns(2)
    col_c.plotly_chart(
        visualizations.form_evolution_chart(
            home_team,
            list(reversed(home_profile.recent_form.dates)),
            list(reversed(home_profile.recent_form.results)),
        ),
        use_container_width=True,
    )
    col_d.plotly_chart(
        visualizations.form_evolution_chart(
            away_team,
            list(reversed(away_profile.recent_form.dates)),
            list(reversed(away_profile.recent_form.results)),
        ),
        use_container_width=True,
    )

    col_e, col_f = st.columns(2)
    col_e.plotly_chart(
        visualizations.team_results_pie(
            home_team, home_profile.overall.win_pct, home_profile.overall.draw_pct, home_profile.overall.loss_pct
        ),
        use_container_width=True,
    )
    col_f.plotly_chart(
        visualizations.team_results_pie(
            away_team, away_profile.overall.win_pct, away_profile.overall.draw_pct, away_profile.overall.loss_pct
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Enfrentamientos directos en el archivo")
    h2h = statistics.head_to_head(historical_df, home_team, away_team)
    if h2h.empty:
        st.write("No hay enfrentamientos directos entre estos dos equipos en el archivo cargado.")
    else:
        st.dataframe(h2h, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Descargar esta comparación (Excel)",
        data=report_generator.comparison_to_excel(home_team, away_team, home_profile, away_profile),
        file_name=f"comparacion_{home_team}_vs_{away_team}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _confidence_badge(label: str) -> str:
    return f'<span class="confidence-badge confidence-{label}">{label.upper()}</span>'


def _render_probability_row(home_team: str, away_team: str, prob_home: float, prob_draw: float, prob_away: float) -> None:
    st.markdown("### Probabilidades del resultado")
    p1, p2, p3 = st.columns(3)
    p1.metric(f"Victoria {home_team}", format_pct(prob_home))
    p2.metric("Empate", format_pct(prob_draw))
    p3.metric(f"Victoria {away_team}", format_pct(prob_away))
    st.caption(f"Suma de probabilidades: {prob_home + prob_draw + prob_away:.1f}%")


def _render_goals_markets_and_matrix(prediction: poisson_model.PoissonPrediction, home_team: str, away_team: str) -> None:
    st.markdown("### Goles esperados y marcador más probable")
    st.caption("Estas cifras siempre provienen del modelo de Poisson: es el único que estima goles y marcador exacto.")
    g1, g2, g3 = st.columns(3)
    g1.metric(f"Goles esperados — {home_team}", format_metric(prediction.expected_home_goals))
    g2.metric(f"Goles esperados — {away_team}", format_metric(prediction.expected_away_goals))
    top_score, top_prob = prediction.top_scores[0]
    g3.metric("Marcador exacto más probable", f"{top_score} ({top_prob:.1f}%)")

    st.markdown("---")
    st.subheader("Mercados estadísticos")

    dnb_total = prediction.prob_home_win + prediction.prob_away_win
    dnb_home = round(prediction.prob_home_win / dnb_total * 100, 1) if dnb_total > 0 else None
    dnb_away = round(prediction.prob_away_win / dnb_total * 100, 1) if dnb_total > 0 else None

    market_rows = [
        {"Mercado": "Más de 1.5 goles", "Probabilidad": format_pct(prediction.prob_over[1.5])},
        {"Mercado": "Menos de 1.5 goles", "Probabilidad": format_pct(prediction.prob_under[1.5])},
        {"Mercado": "Más de 2.5 goles", "Probabilidad": format_pct(prediction.prob_over[2.5])},
        {"Mercado": "Menos de 2.5 goles", "Probabilidad": format_pct(prediction.prob_under[2.5])},
        {"Mercado": "Más de 3.5 goles", "Probabilidad": format_pct(prediction.prob_over[3.5])},
        {"Mercado": "Menos de 3.5 goles", "Probabilidad": format_pct(prediction.prob_under[3.5])},
        {"Mercado": "Ambos equipos anotan", "Probabilidad": format_pct(prediction.prob_btts_yes)},
        {"Mercado": "Ambos equipos NO anotan", "Probabilidad": format_pct(prediction.prob_btts_no)},
        {"Mercado": f"Portería a cero — {home_team}", "Probabilidad": format_pct(prediction.prob_clean_sheet_home)},
        {"Mercado": f"Portería a cero — {away_team}", "Probabilidad": format_pct(prediction.prob_clean_sheet_away)},
        {
            "Mercado": "Doble oportunidad 1X",
            "Probabilidad": format_pct(round(prediction.prob_home_win + prediction.prob_draw, 1)),
        },
        {
            "Mercado": "Doble oportunidad X2",
            "Probabilidad": format_pct(round(prediction.prob_draw + prediction.prob_away_win, 1)),
        },
        {
            "Mercado": "Doble oportunidad 12",
            "Probabilidad": format_pct(round(prediction.prob_home_win + prediction.prob_away_win, 1)),
        },
        {"Mercado": f"Victoria sin empate — {home_team}", "Probabilidad": format_pct(dnb_home)},
        {"Mercado": f"Victoria sin empate — {away_team}", "Probabilidad": format_pct(dnb_away)},
        {
            "Mercado": "Total de goles esperados",
            "Probabilidad": format_metric(round(prediction.expected_home_goals + prediction.expected_away_goals, 2)),
        },
    ]
    st.dataframe(pd.DataFrame(market_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Matriz de marcadores")
    st.plotly_chart(
        visualizations.score_matrix_heatmap(prediction.score_matrix, home_team, away_team),
        use_container_width=True,
    )
    st.plotly_chart(visualizations.top_scores_bar(prediction.top_scores), use_container_width=True)


def render_prediccion(
    historical_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    sufficient_data: bool,
    recent_n: int,
    model_choice: str,
    logistic_result: prediction_model.LogisticModelResult,
    poisson_backtest: poisson_model.BacktestMetrics | None,
    stat_completeness_pct: float,
    file_label: str,
    competition_label: str,
    season_label: str,
) -> None:
    st.header("🔮 Predicción principal")
    st.caption(f"Modelo utilizado: {model_choice}")

    poisson_pred = poisson_model.predict_match(historical_df, home_team, away_team)

    for w in poisson_pred.warnings:
        st.warning(w)
    if not sufficient_data:
        st.warning(
            f"El archivo tiene menos de {MIN_MATCHES_RELIABLE} partidos con resultado en total. "
            "Esta predicción es orientativa."
        )

    logistic_probs = None
    if model_choice in ("Regresión logística", "Combinado") and logistic_result.trained:
        logistic_probs = prediction_model.predict_logistic(logistic_result, historical_df, home_team, away_team, recent_n)

    expected_home_goals = None
    expected_away_goals = None
    top_score_final = None

    if model_choice == "Poisson":
        confidence = poisson_pred.confidence
        st.markdown(f"**Nivel de confianza:** {_confidence_badge(confidence)}", unsafe_allow_html=True)
        st.caption(
            f"Basado en {poisson_pred.home_matches_used} partidos de {home_team} como local y "
            f"{poisson_pred.away_matches_used} partidos de {away_team} como visitante."
        )
        _render_probability_row(home_team, away_team, poisson_pred.prob_home_win, poisson_pred.prob_draw, poisson_pred.prob_away_win)
        st.markdown("---")
        _render_goals_markets_and_matrix(poisson_pred, home_team, away_team)

        top_score, top_prob = poisson_pred.top_scores[0]
        explanation = (
            f"Con base en los {poisson_pred.home_matches_used} partidos de {home_team} como local y "
            f"los {poisson_pred.away_matches_used} partidos de {away_team} como visitante presentes en el "
            f"archivo cargado, el modelo de Poisson estima un marcador esperado cercano a "
            f"{poisson_pred.expected_home_goals:.1f} - {poisson_pred.expected_away_goals:.1f}. "
            f"La probabilidad de victoria de {home_team} es {poisson_pred.prob_home_win:.1f}%, la de empate "
            f"{poisson_pred.prob_draw:.1f}%, y la de victoria de {away_team} {poisson_pred.prob_away_win:.1f}%. "
            f"El marcador exacto más probable es {top_score} con {top_prob:.1f}% de probabilidad. "
            f"El nivel de confianza es '{confidence}', calculado según la cantidad de partidos "
            "disponibles para cada equipo en su condición correspondiente."
        )
        final_prob_home, final_prob_draw, final_prob_away = (
            poisson_pred.prob_home_win,
            poisson_pred.prob_draw,
            poisson_pred.prob_away_win,
        )
        expected_home_goals, expected_away_goals = poisson_pred.expected_home_goals, poisson_pred.expected_away_goals
        top_score_final = top_score

    elif model_choice == "Regresión logística":
        if logistic_probs is None:
            st.error(
                "El modelo de regresión logística no está disponible (datos insuficientes). "
                "Selecciona 'Poisson' en la barra lateral."
            )
            return
        sorted_probs = sorted(logistic_probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        log_loss_value = logistic_result.metrics.log_loss if logistic_result.metrics else None
        confidence = prediction_model.simple_confidence(margin, log_loss_value)
        st.markdown(f"**Nivel de confianza:** {_confidence_badge(confidence)}", unsafe_allow_html=True)
        st.caption(
            f"Entrenado con {logistic_result.metrics.n_train} partidos y validado con "
            f"{logistic_result.metrics.n_test} partidos de prueba (ver pestaña 'Rendimiento del modelo')."
        )
        _render_probability_row(home_team, away_team, logistic_probs["home_win"], logistic_probs["draw"], logistic_probs["away_win"])
        st.info(
            "La regresión logística solo predice el resultado 1X2: no calcula goles esperados, "
            "marcador exacto ni mercados de goles. Usa Poisson o Combinado para eso."
        )
        explanation = (
            f"Según el modelo de regresión logística (entrenado con {logistic_result.metrics.n_train} "
            f"partidos previos y variables como forma reciente, rendimiento local/visitante y goles "
            f"promedio), la probabilidad de victoria de {home_team} es {logistic_probs['home_win']:.1f}%, "
            f"la de empate {logistic_probs['draw']:.1f}%, y la de victoria de {away_team} "
            f"{logistic_probs['away_win']:.1f}%. El nivel de confianza es '{confidence}'."
        )
        final_prob_home, final_prob_draw, final_prob_away = (
            logistic_probs["home_win"],
            logistic_probs["draw"],
            logistic_probs["away_win"],
        )

    else:  # Combinado
        combined = prediction_model.combine_predictions(
            poisson_pred=poisson_pred,
            logistic_probs=logistic_probs,
            poisson_backtest=poisson_backtest,
            logistic_metrics=logistic_result.metrics if logistic_result.trained else None,
            stat_completeness_pct=stat_completeness_pct,
        )
        confidence = combined.confidence
        st.markdown(f"**Nivel de confianza:** {_confidence_badge(confidence)}", unsafe_allow_html=True)
        st.caption(combined.weighting_explanation)
        with st.expander("Ver desglose del nivel de confianza"):
            for factor, score in combined.confidence_breakdown.items():
                st.write(f"- {factor}: {score:.1f} puntos")

        _render_probability_row(home_team, away_team, combined.prob_home_win, combined.prob_draw, combined.prob_away_win)
        st.markdown("---")
        _render_goals_markets_and_matrix(poisson_pred, home_team, away_team)

        top_score, top_prob = poisson_pred.top_scores[0]
        explanation = (
            f"La predicción combinada pondera el modelo de Poisson ({combined.weight_poisson:.0f}%) y la "
            f"regresión logística ({combined.weight_logistic:.0f}%) según su desempeño de validación. "
            f"El resultado combinado da {combined.prob_home_win:.1f}% de victoria para {home_team}, "
            f"{combined.prob_draw:.1f}% de empate y {combined.prob_away_win:.1f}% de victoria para "
            f"{away_team}. Los goles esperados y el marcador más probable ({top_score}, {top_prob:.1f}%) "
            f"provienen del modelo de Poisson. El nivel de confianza combinado es '{confidence}'."
        )
        final_prob_home, final_prob_draw, final_prob_away = (
            combined.prob_home_win,
            combined.prob_draw,
            combined.prob_away_win,
        )
        expected_home_goals, expected_away_goals = poisson_pred.expected_home_goals, poisson_pred.expected_away_goals
        top_score_final = top_score

    st.markdown("---")
    st.subheader("Explicación de la predicción")
    st.write(explanation)
    st.caption(
        "Esta es una estimación estadística basada únicamente en los datos del archivo cargado. "
        "Ninguna probabilidad garantiza el resultado real del partido."
    )

    # ----------------------------------------------------------------
    # Guardar en el historial y exportar (Fase 3)
    # ----------------------------------------------------------------
    predicted_outcome = max(
        {"H": final_prob_home, "D": final_prob_draw, "A": final_prob_away}.items(), key=lambda kv: kv[1]
    )[0]

    record = {
        "timestamp": report_generator.now_str(),
        "file_name": file_label,
        "competition": competition_label,
        "season": season_label,
        "home_team": home_team,
        "away_team": away_team,
        "model_used": model_choice,
        "prob_home_win": final_prob_home,
        "prob_draw": final_prob_draw,
        "prob_away_win": final_prob_away,
        "predicted_outcome": predicted_outcome,
        "confidence": confidence,
        "expected_home_goals": expected_home_goals if expected_home_goals is not None else "",
        "expected_away_goals": expected_away_goals if expected_away_goals is not None else "",
        "actual_home_goals": "",
        "actual_away_goals": "",
        "actual_outcome": "",
        "correct": "",
    }

    st.markdown("---")
    st.subheader("Guardar y exportar")

    save_col, csv_col, matrix_col, html_col = st.columns(4)

    with save_col:
        if st.button("💾 Guardar en el historial", use_container_width=True):
            saved_id = prediction_history.append_prediction(record)
            st.success(f"Predicción #{saved_id} guardada. Revisa la pestaña 'Historial de predicciones'.")

    with csv_col:
        st.download_button(
            "⬇ Predicción (CSV)",
            data=report_generator.prediction_to_csv(record),
            file_name=f"prediccion_{home_team}_vs_{away_team}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with matrix_col:
        st.download_button(
            "⬇ Matriz de marcadores (CSV)",
            data=report_generator.score_matrix_to_csv(poisson_pred.score_matrix),
            file_name=f"matriz_{home_team}_vs_{away_team}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with html_col:
        home_profile = statistics.build_team_profile(historical_df, home_team, recent_n)
        away_profile = statistics.build_team_profile(historical_df, away_team, recent_n)
        comparison_rows = report_generator.record_rows(home_team, away_team, home_profile.overall, away_profile.overall)
        summary = statistics.compute_dataset_summary(historical_df)
        period = (
            f"{summary.period_start.date()} → {summary.period_end.date()}"
            if summary.period_start is not None and summary.period_end is not None
            else "Desconocido"
        )
        html_report = report_generator.build_html_report(
            {
                "file_name": file_label,
                "analysis_date": report_generator.now_str(),
                "competition": competition_label,
                "season_labels": season_label,
                "n_matches": summary.n_matches,
                "period": period,
                "home_team": home_team,
                "away_team": away_team,
                "model_choice": model_choice,
                "prob_home_win": final_prob_home,
                "prob_draw": final_prob_draw,
                "prob_away_win": final_prob_away,
                "expected_home_goals": expected_home_goals if expected_home_goals is not None else "N/D",
                "expected_away_goals": expected_away_goals if expected_away_goals is not None else "N/D",
                "top_score": top_score_final or "N/D",
                "confidence": confidence,
                "explanation": explanation,
                "warnings": poisson_pred.warnings,
                "comparison_rows": comparison_rows,
                "score_matrix": poisson_pred.score_matrix,
            }
        )
        st.download_button(
            "⬇ Reporte completo (HTML)",
            data=html_report.encode("utf-8"),
            file_name=f"reporte_{home_team}_vs_{away_team}.html",
            mime="text/html",
            use_container_width=True,
        )


def render_rendimiento(
    logistic_result: prediction_model.LogisticModelResult,
    poisson_backtest: poisson_model.BacktestMetrics | None,
    manual_mapping: dict[str, str | None],
    poisson_test_df: pd.DataFrame | None,
    poisson_test_predictions: list[dict[str, float]] | None,
) -> None:
    st.header("🧮 Rendimiento del modelo")
    st.caption(
        "Evaluación sobre un único split cronológico (entrenamiento con los partidos más "
        "antiguos, prueba con los más recientes). Con más partidos o temporadas, estas "
        "métricas pueden variar."
    )

    poisson_market_rows = None
    if poisson_test_df is not None and market_odds.has_market_odds(poisson_test_df):
        poisson_market_rows = market_odds.build_rows_from_df(poisson_test_df)
    poisson_market_bt = market_odds.market_backtest(poisson_market_rows) if poisson_market_rows is not None else None

    comparison_rows = []
    if poisson_backtest is not None:
        comparison_rows.append(
            {
                "Modelo": "Poisson",
                "Partidos de prueba": poisson_backtest.n_test,
                "Accuracy": format_pct(round(poisson_backtest.accuracy * 100, 1)),
                "Log loss": format_metric(poisson_backtest.log_loss, decimals=3),
                "Brier score": format_metric(poisson_backtest.brier_score, decimals=3),
            }
        )
    if logistic_result.trained:
        metrics = logistic_result.metrics
        comparison_rows.append(
            {
                "Modelo": "Regresión logística",
                "Partidos de prueba": metrics.n_test,
                "Accuracy": format_pct(round(metrics.accuracy * 100, 1)),
                "Log loss": format_metric(metrics.log_loss, decimals=3),
                "Brier score": format_metric(metrics.brier_score, decimals=3),
            }
        )
    if poisson_market_bt is not None:
        comparison_rows.append(
            {
                "Modelo": "Mercado (cuotas de referencia)",
                "Partidos de prueba": poisson_market_bt.n_test,
                "Accuracy": format_pct(round(poisson_market_bt.accuracy * 100, 1)),
                "Log loss": format_metric(poisson_market_bt.log_loss, decimals=3),
                "Brier score": format_metric(poisson_market_bt.brier_score, decimals=3),
            }
        )

    if comparison_rows:
        st.subheader("Comparación de modelos")
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        if poisson_market_bt is not None:
            st.caption(
                f"La fila 'Mercado' se evalúa sobre los mismos {poisson_market_bt.n_test} partidos de "
                f"prueba de Poisson. Margen de casa promedio (overround): "
                f"{format_pct(poisson_market_bt.avg_overround_pct)}."
            )
    else:
        st.info("No hay suficientes partidos para calcular métricas de validación todavía.")

    if not logistic_result.trained:
        for w in logistic_result.warnings:
            st.warning(w)
    else:
        metrics = logistic_result.metrics
        st.markdown("---")
        st.subheader("Métricas detalladas — Regresión logística")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", format_pct(round(metrics.accuracy * 100, 1)))
        m2.metric("Precisión (macro)", format_pct(round(metrics.precision_macro * 100, 1)))
        m3.metric("Recall (macro)", format_pct(round(metrics.recall_macro * 100, 1)))
        m4.metric("F1 (macro)", format_pct(round(metrics.f1_macro * 100, 1)))
        st.caption(
            f"Entrenado con {metrics.n_train} partidos, evaluado con {metrics.n_test}. "
            f"Probabilidades calibradas: {'Sí (CalibratedClassifierCV)' if logistic_result.calibrated else 'No'}."
        )

        col_a, col_b = st.columns(2)
        col_a.plotly_chart(
            visualizations.confusion_matrix_heatmap(
                metrics.confusion_matrix, metrics.labels, "Matriz de confusión — Regresión logística"
            ),
            use_container_width=True,
        )
        col_b.plotly_chart(
            visualizations.feature_importance_bar(logistic_result.feature_importance, feature_engineering.FEATURE_LABELS),
            use_container_width=True,
        )

        if logistic_result.warnings:
            st.markdown("---")
            st.subheader("Advertencias y limitaciones")
            for w in logistic_result.warnings:
                st.warning(w)

        st.caption(
            f"Modelo guardado en models/{prediction_model.MODEL_FILENAME} "
            f"(entrenado el {logistic_result.trained_at})."
        )

    # ------------------------------------------------------------
    # Análisis de valor (apuestas)
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("💰 Análisis de valor (apuestas)")

    logistic_market_rows = None
    if (
        logistic_result.trained
        and logistic_result.test_meta is not None
        and market_odds.has_market_odds(logistic_result.test_meta)
    ):
        logistic_market_rows = logistic_result.test_meta[[*market_odds.MARKET_ODDS_COLUMNS, "actual_outcome"]]

    if poisson_market_rows is None and logistic_market_rows is None:
        st.info(
            "Este archivo no trae cuotas de mercado utilizables (o no hay suficientes partidos de "
            "prueba). Esta sección se activa automáticamente con archivos de football-data.co.uk "
            "que incluyan columnas de cuotas (Avg*, B365*, PS*)."
        )
    else:
        st.caption(
            "Compara, sobre los partidos de prueba (nunca usados para entrenar), la probabilidad "
            "implícita del mercado (cuotas, sin el margen de la casa) contra la del modelo. Cuando "
            "el modelo se aleja del mercado por más del umbral elegido, se simula una apuesta de 1 "
            "unidad — en retrospectiva, no una recomendación real."
        )
        edge_threshold = st.slider(
            "Umbral de valor (diferencia mínima modelo vs. mercado, en puntos porcentuales)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
        )

        value_rows = []
        if poisson_market_rows is not None and poisson_test_predictions is not None:
            poisson_value = market_odds.simulate_value_bets(poisson_market_rows, poisson_test_predictions, edge_threshold)
            if poisson_value is not None:
                value_rows.append(
                    {
                        "Modelo": "Poisson",
                        "Apuestas simuladas": poisson_value.n_bets,
                        "Aciertos": poisson_value.n_wins,
                        "% Acierto": format_pct(poisson_value.win_rate_pct),
                        "ROI simulado": format_pct(poisson_value.roi_pct),
                        "Ganancia/pérdida (unidades)": format_metric(poisson_value.total_profit),
                    }
                )
        if logistic_market_rows is not None:
            logistic_value = market_odds.simulate_value_bets(
                logistic_market_rows, logistic_result.test_predictions, edge_threshold
            )
            if logistic_value is not None:
                value_rows.append(
                    {
                        "Modelo": "Regresión logística",
                        "Apuestas simuladas": logistic_value.n_bets,
                        "Aciertos": logistic_value.n_wins,
                        "% Acierto": format_pct(logistic_value.win_rate_pct),
                        "ROI simulado": format_pct(logistic_value.roi_pct),
                        "Ganancia/pérdida (unidades)": format_metric(logistic_value.total_profit),
                    }
                )

        if value_rows:
            st.dataframe(pd.DataFrame(value_rows), use_container_width=True, hide_index=True)
        else:
            st.write("Ningún modelo generó apuestas simuladas con este umbral.")

        st.warning(
            "Esta simulación es retrospectiva, sobre una muestra pequeña de partidos y un único "
            "split de validación: no es una predicción de rendimiento futuro ni una recomendación "
            "de apuesta. El tamaño de muestra suele ser demasiado chico para sacar conclusiones "
            "estadísticamente sólidas. Apostar siempre implica riesgo real de pérdida de dinero."
        )

    st.markdown("---")
    st.download_button(
        "⬇ Resumen de variables utilizadas (CSV)",
        data=report_generator.variables_summary_to_csv(
            manual_mapping,
            column_mapper.FIELD_LABELS,
            logistic_result.feature_importance if logistic_result.trained else None,
        ),
        file_name="variables_utilizadas.csv",
        mime="text/csv",
    )


def render_historial() -> None:
    st.header("📜 Historial de predicciones")
    st.caption(
        "Se guarda localmente en data/prediction_history.csv. Completa el resultado real de un "
        "partido para que cuente en el rendimiento histórico del sistema."
    )

    history_df = prediction_history.load_history()

    if history_df.empty:
        st.info(
            "Todavía no has guardado ninguna predicción. Ve a la pestaña 'Predicción principal', "
            "genera una predicción y usa el botón 'Guardar en el historial'."
        )
        return

    display_cols = [
        "id", "timestamp", "home_team", "away_team", "model_used",
        "prob_home_win", "prob_draw", "prob_away_win", "predicted_outcome",
        "confidence", "actual_outcome", "correct",
    ]
    st.dataframe(history_df[display_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Completar resultado real de un partido")

    pending = prediction_history.pending_records(history_df)
    if pending.empty:
        st.write("No hay predicciones pendientes de resultado.")
    else:
        options = {
            f"#{row.id} — {row.home_team} vs {row.away_team} ({row.timestamp})": int(row.id)
            for row in pending.itertuples()
        }
        choice_label = st.selectbox("Predicción a completar", list(options.keys()))
        chosen_id = options[choice_label]

        g1, g2, g3 = st.columns([1, 1, 1])
        actual_home_goals = g1.number_input("Goles del local", min_value=0, max_value=20, step=1, key="actual_home_goals")
        actual_away_goals = g2.number_input("Goles del visitante", min_value=0, max_value=20, step=1, key="actual_away_goals")
        if g3.button("Guardar resultado real", use_container_width=True):
            prediction_history.update_actual_result(chosen_id, int(actual_home_goals), int(actual_away_goals))
            st.success(f"Resultado real guardado para la predicción #{chosen_id}.")
            st.rerun()

    st.markdown("---")
    st.subheader("Rendimiento histórico del sistema")
    performance = prediction_history.compute_performance(history_df)

    p1, p2, p3 = st.columns(3)
    p1.metric("Predicciones guardadas", performance["total_predictions"])
    p2.metric("Con resultado cargado", performance["resolved"])
    p3.metric("Accuracy general", format_pct(performance["accuracy"]))

    if performance["by_model"]:
        st.write("**Accuracy por modelo utilizado:**")
        by_model_rows = [
            {"Modelo": model, "Predicciones resueltas": v["n"], "Accuracy": format_pct(v["accuracy"])}
            for model, v in performance["by_model"].items()
        ]
        st.dataframe(pd.DataFrame(by_model_rows), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Descargar historial completo (CSV)",
        data=history_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="historial_predicciones.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------


def main() -> None:
    st.sidebar.header("⚽ Football Predictor")
    uploaded_files = st.sidebar.file_uploader(
        "Cargar archivo(s) de partidos",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Puedes cargar varios archivos (por ejemplo, varias temporadas de la misma liga).",
    )

    if not uploaded_files:
        st.title("⚽ Football Predictor")
        st.markdown(
            "Carga uno o más archivos Excel o CSV con resultados de partidos desde la "
            "barra lateral para comenzar. La aplicación reconoce automáticamente el "
            "formato de football-data.co.uk (Div, Date, HomeTeam, AwayTeam, FTHG, FTAG...) "
            "y también intenta detectar variantes con otros nombres de columna."
        )
        st.stop()

    loaded_files = _load_uploaded_files(uploaded_files)
    loaded_files = [lf for lf in loaded_files if lf.raw_df is not None and not lf.raw_df.empty]

    if not loaded_files:
        st.error("No se pudo cargar ningún archivo válido.")
        st.stop()

    combined_raw = data_loader.concat_multiple_files(loaded_files)
    if combined_raw.empty:
        st.error("Los archivos cargados no contienen datos.")
        st.stop()

    auto_mapping = column_mapper.auto_map_columns(combined_raw)
    missing_required = column_mapper.get_missing_required(auto_mapping)

    with st.sidebar.expander("🔧 Revisar mapeo de columnas", expanded=bool(missing_required)):
        st.caption("Los campos marcados con * son obligatorios.")
        manual_mapping: dict[str, str | None] = {}
        options = ["-- No disponible --"] + [str(c) for c in combined_raw.columns]
        for field in column_mapper.ALL_FIELDS:
            label = column_mapper.FIELD_LABELS.get(field, field)
            marker = " *" if field in column_mapper.REQUIRED_FIELDS else ""
            default_col = auto_mapping.get(field)
            default_index = options.index(default_col) if default_col in options else 0
            choice = st.selectbox(f"{label}{marker}", options, index=default_index, key=f"map_{field}")
            manual_mapping[field] = None if choice == "-- No disponible --" else choice

    missing_required = column_mapper.get_missing_required(manual_mapping)
    if missing_required:
        labels = [column_mapper.FIELD_LABELS.get(f, f) for f in missing_required]
        st.error(
            "No se pudieron identificar columnas obligatorias: "
            f"{', '.join(labels)}. Selecciónalas manualmente en 'Revisar mapeo de columnas' "
            "en la barra lateral."
        )
        st.stop()

    mapped_df = column_mapper.apply_mapping(combined_raw, manual_mapping)

    season_labels: list[str] = []
    for lf in loaded_files:
        lf_mapped = column_mapper.apply_mapping(lf.raw_df, manual_mapping)
        if "date" in lf_mapped.columns:
            dates = pd.to_datetime(lf_mapped["date"], dayfirst=True, errors="coerce")
            season_labels.append(column_mapper.infer_season_label(dates))
    season_labels = sorted(set(season_labels))

    competition_value = None
    if "competition" in mapped_df.columns:
        auto_competition = column_mapper.detect_single_value_competition(mapped_df)
        if auto_competition:
            competition_value = auto_competition
        else:
            unique_competitions = sorted(mapped_df["competition"].dropna().unique().tolist())
            if len(unique_competitions) > 1:
                chosen = st.sidebar.selectbox("Competición", ["Todas"] + unique_competitions)
                competition_value = None if chosen == "Todas" else chosen

    if competition_value:
        mapped_df = mapped_df[mapped_df["competition"] == competition_value]

    has_xg = column_mapper.has_xg_data(manual_mapping)

    with st.spinner("📊 Procesando archivo(s) y calculando estadísticas..."):
        historical_df, pending_df, cleaning_report = data_cleaner.clean_data(mapped_df)

    if historical_df.empty:
        st.error("No hay partidos con resultado final válido para analizar en los datos cargados.")
        st.stop()

    teams = statistics.get_team_list(historical_df)
    if len(teams) < 2:
        st.error("Se necesitan al menos dos equipos distintos en los datos para poder comparar.")
        st.stop()

    st.sidebar.markdown("---")
    home_team = st.sidebar.selectbox("Equipo local", teams, key="home_team_select")
    default_away_index = 1 if len(teams) > 1 and teams[1] != home_team else 0
    away_team = st.sidebar.selectbox("Equipo visitante", teams, index=default_away_index, key="away_team_select")

    recent_n = st.sidebar.slider("Partidos recientes para calcular la forma", min_value=3, max_value=15, value=5)

    with st.spinner("🧠 Entrenando y validando los modelos (Poisson + regresión logística)..."):
        logistic_result = _train_logistic_cached(historical_df, recent_n)
        poisson_bt = _poisson_backtest_cached(historical_df)

        poisson_split = _poisson_split_cached(historical_df)
        if poisson_split is not None:
            poisson_train_df, poisson_test_df = poisson_split
            poisson_test_predictions = _poisson_test_predictions_cached(poisson_train_df, poisson_test_df)
        else:
            poisson_test_df, poisson_test_predictions = None, None

    if logistic_result.trained:
        model_options = ["Poisson", "Regresión logística", "Combinado"]
        default_model_index = 2
    else:
        model_options = ["Poisson"]
        default_model_index = 0
        st.sidebar.caption(
            f"Regresión logística y Combinado no disponibles: se necesitan al menos "
            f"{MIN_MATCHES_FOR_ML_MODEL} partidos utilizables tras calcular variables "
            f"(hay {logistic_result.n_usable_matches}). Ver pestaña 'Rendimiento del modelo'."
        )

    model_choice = st.sidebar.selectbox(
        "Modelo de predicción",
        model_options,
        index=default_model_index,
        key="model_select",
    )

    run = st.sidebar.button("▶ Ejecutar análisis", type="primary", use_container_width=True)

    team_errors = statistics.validate_team_selection(historical_df, home_team, away_team)

    stat_flags = statistics.stat_columns_available(historical_df)
    stat_completeness_pct = sum(stat_flags.values()) / len(stat_flags) * 100 if stat_flags else 0.0

    file_label = "; ".join(lf.filename for lf in loaded_files)
    competition_label = competition_value or "N/D"
    season_label = ", ".join(season_labels) if season_labels else "N/D"

    tab_inicio, tab_resumen, tab_comparacion, tab_prediccion, tab_rendimiento, tab_historial = st.tabs(
        [
            "🏠 Inicio",
            "📊 Resumen del dataset",
            "🆚 Comparación de equipos",
            "🔮 Predicción principal",
            "🧮 Rendimiento del modelo",
            "📜 Historial de predicciones",
        ]
    )

    with tab_inicio:
        render_inicio(loaded_files, season_labels, competition_value, historical_df, cleaning_report, has_xg)

    with tab_resumen:
        render_resumen(combined_raw, historical_df, cleaning_report, recent_n)

    with tab_comparacion:
        if not run:
            st.info("Presiona **Ejecutar análisis** en la barra lateral para ver la comparación.")
        elif team_errors:
            for e in team_errors:
                st.error(e)
        else:
            render_comparacion(historical_df, home_team, away_team, recent_n)

    with tab_prediccion:
        if not run:
            st.info("Presiona **Ejecutar análisis** en la barra lateral para ver la predicción.")
        elif team_errors:
            for e in team_errors:
                st.error(e)
        else:
            render_prediccion(
                historical_df,
                home_team,
                away_team,
                cleaning_report.sufficient_data,
                recent_n,
                model_choice,
                logistic_result,
                poisson_bt,
                stat_completeness_pct,
                file_label,
                competition_label,
                season_label,
            )

    with tab_rendimiento:
        render_rendimiento(logistic_result, poisson_bt, manual_mapping, poisson_test_df, poisson_test_predictions)

    with tab_historial:
        render_historial()


if __name__ == "__main__":
    main()
