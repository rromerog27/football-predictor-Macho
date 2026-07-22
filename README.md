# Football Predictor

Aplicación web local (Streamlit) para cargar archivos de partidos de fútbol
(Excel o CSV), analizar el rendimiento de los equipos y generar predicciones
estadísticas de un partido mediante un modelo de distribución de Poisson.

**Todo el análisis se basa únicamente en los datos del archivo que cargues.**
No se consulta internet, rankings, lesiones, alineaciones ni datos
históricos externos.

## Estado actual: Fase 3 — Exportación e historial (completa)

Implementado hasta ahora:

**Fase 1:**
- Carga de archivos `.xlsx`, `.xls` y `.csv` (uno o varios a la vez).
- Reconocimiento automático de columnas, con el formato football-data.co.uk
  como caso principal soportado.
- Mapeo manual de columnas cuando el reconocimiento automático falla.
- Limpieza y validación de datos (fechas, duplicados, valores imposibles,
  separación de partidos jugados vs. pendientes).
- Estadísticas por equipo (general, como local, como visitante, forma
  reciente).
- Modelo de predicción por distribución de Poisson (goles esperados,
  probabilidades 1X2, mercados de goles, matriz de marcadores).

**Fase 2:**
- `feature_engineering.py`: variables pre-partido (forma, rendimiento
  local/visitante, descanso) calculadas cronológicamente sin fuga de
  información.
- `prediction_model.py`: regresión logística multiclase de respaldo,
  calibrada con `CalibratedClassifierCV` cuando el volumen de datos lo
  permite, con división cronológica de entrenamiento/prueba.
- Backtest de Poisson sobre el mismo split cronológico, para comparar
  ambos modelos de forma justa.
- Predicción combinada: pondera Poisson y regresión logística según su
  log loss de validación (no una ponderación arbitraria), con nivel de
  confianza calculado a partir de cantidad de datos, margen entre
  probabilidades, consistencia entre modelos, desempeño de validación y
  completitud de los datos.
- Selector de modelo en el dashboard (Poisson / Regresión logística /
  Combinado), deshabilitado a las opciones no disponibles cuando hay
  menos de 100 partidos utilizables.
- Sección "Rendimiento del modelo": métricas de validación, matriz de
  confusión, importancia de variables, comparación Poisson vs. regresión
  logística.
- Modelos entrenados guardados en `models/` con `joblib`.

**Fase 3:**
- `report_generator.py`: exportación de la tabla de estadísticas de todos
  los equipos (CSV), la predicción de un partido (CSV), la comparación de
  dos equipos (Excel con varias hojas), la matriz de marcadores (CSV) y un
  reporte HTML completo autocontenido (fecha del análisis, archivo,
  equipos comparados, resultados y advertencias).
- `prediction_history.py`: historial de predicciones persistido en
  `data/prediction_history.csv`. Cada predicción guardada registra fecha,
  partido, probabilidades, resultado pronosticado, modelo utilizado y
  versión del modelo; permite completar después el resultado real y
  calcula el rendimiento histórico (accuracy general y por modelo).
- Pestaña "Historial de predicciones" en el dashboard, con botón para
  guardar cada predicción, formulario para cargar resultados reales, y
  métricas de rendimiento histórico.
- Botones de descarga distribuidos en cada sección relevante del
  dashboard (Resumen, Comparación, Predicción, Rendimiento del modelo).

No implementado (fuera del alcance pedido para esta app): exportación a
PDF y matriz de marcadores como imagen (se exporta como CSV en su lugar,
ver sección "Mejoras futuras").

**Extra — Análisis de valor (apuestas):**
- `data_loader.py` ahora conserva (en vez de descartar) la cuota de cierre
  promedio 1X2 de cada partido (o la mejor alternativa disponible: apertura
  promedio, Bet365 o Pinnacle) cuando el archivo las trae.
- `market_odds.py`: convierte esas cuotas a probabilidad implícita quitando
  el margen de la casa (overround), evalúa qué tan bien predice el propio
  mercado los partidos de prueba (comparable con Poisson y la regresión
  logística en la misma tabla de "Rendimiento del modelo"), y simula en
  retrospectiva una estrategia simple de apuestas de valor (cuándo el
  modelo se aleja del mercado por más de un umbral configurable) con
  ganancia/pérdida y ROI simulados.
- Todo esto se activa automáticamente cuando el archivo cargado trae
  columnas de cuotas reconocibles; si no las trae, esta sección se oculta
  con una nota explicativa y el resto de la app sigue igual.
- Advertencia explícita en la propia sección: es una simulación
  retrospectiva sobre muestras pequeñas, no una recomendación de apuesta
  ni una garantía de resultados futuros.

## Requisitos

- Python 3.11 (o una versión 3.11+ estable compatible).

## Instalación

```bash
cd football_predictor
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

Se abrirá automáticamente en el navegador (por defecto en
`http://localhost:8501`).

## Estructura del proyecto

```
football_predictor/
│
├── app.py                    # Punto de entrada del dashboard (Streamlit)
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Este archivo
│
├── data/                     # Archivos de datos locales (ejemplo + historial de predicciones)
├── models/                   # Modelos entrenados guardados (joblib)
├── reports/                  # Carpeta reservada para futuros reportes guardados a disco
│
├── src/
│   ├── data_loader.py          # Lectura de Excel/CSV, detección de hojas,
│   │                             #   eliminación de columnas de cuotas de apuestas
│   ├── column_mapper.py        # Reconocimiento y mapeo de columnas a nombres canónicos
│   ├── data_cleaner.py         # Limpieza, validación, separación histórico/pendiente
│   ├── statistics.py           # Estadísticas por equipo, forma reciente, comparación
│   ├── poisson_model.py        # Modelo Poisson (goles esperados, mercados) + backtest
│   ├── feature_engineering.py  # Variables pre-partido sin fuga de información
│   ├── prediction_model.py     # Regresión logística de respaldo + predicción combinada
│   ├── report_generator.py     # Exportación a CSV, Excel y HTML
│   ├── prediction_history.py   # Historial de predicciones persistido en CSV
│   ├── market_odds.py          # Comparación contra cuotas de mercado + simulación de value bets
│   ├── visualizations.py       # Construcción de gráficos Plotly
│   └── utils.py                 # Utilidades comunes (safe_divide, logging, formateo)
│
└── tests/
    ├── test_data_loader.py     # Pruebas de carga y limpieza de columnas de cuotas
    ├── test_statistics.py      # Pruebas de cálculo de estadísticas
    ├── test_predictions.py     # Pruebas de feature engineering, regresión logística y combinación
    ├── test_reports.py         # Pruebas de exportación e historial de predicciones
    └── test_market_odds.py     # Pruebas de probabilidad implícita y simulación de apuestas de valor
```

### Para qué sirve cada archivo

| Archivo | Responsabilidad |
|---|---|
| `app.py` | Orquesta el dashboard: sidebar, pestañas, y llama a las funciones de `src/` para mostrar resultados. No contiene lógica de negocio. |
| `src/data_loader.py` | Lee archivos Excel/CSV, lista hojas de Excel, descarta columnas de cuotas de apuestas cuando detecta el formato football-data.co.uk, y genera el resumen técnico del archivo (filas, columnas, faltantes, duplicados). |
| `src/column_mapper.py` | Detecta automáticamente qué columna del archivo corresponde a cada campo canónico (equipo local, goles, fecha, etc.), con nombres alternativos como respaldo. Infiere la temporada a partir de las fechas y detecta la competición cuando hay un único valor. |
| `src/data_cleaner.py` | Limpia strings, convierte fechas y columnas numéricas, detecta valores imposibles, elimina duplicados, separa partidos jugados de partidos pendientes y ordena cronológicamente. |
| `src/statistics.py` | Calcula el récord, promedios de goles, mercados (%btts, over/under), forma reciente, enfrentamientos directos y el resumen general del dataset. |
| `src/poisson_model.py` | Calcula fortalezas de ataque/defensa relativas a la liga, goles esperados, matriz de marcadores, mercados derivados (1X2, over/under, ambos anotan, etc.) y el backtest cronológico usado para comparar con la regresión logística. |
| `src/feature_engineering.py` | Construye las variables pre-partido (forma, rendimiento local/visitante, descanso) para cada fila de entrenamiento, usando solo partidos anteriores a la fecha del partido (sin fuga de información). |
| `src/prediction_model.py` | Entrena y calibra la regresión logística de respaldo, calcula sus métricas de validación, y combina sus probabilidades con las de Poisson ponderando por desempeño de validación (log loss). |
| `src/report_generator.py` | Exporta a CSV/Excel/HTML lo que ya calcularon los demás módulos: tabla de estadísticas, predicción de un partido, comparación de equipos, matriz de marcadores y el reporte HTML completo. |
| `src/prediction_history.py` | Guarda cada predicción en `data/prediction_history.csv`, permite cargar después el resultado real y calcula el rendimiento histórico del sistema (accuracy general y por modelo). |
| `src/market_odds.py` | Convierte cuotas 1X2 a probabilidad implícita (quitando el margen de la casa), evalúa qué tan bien predice el mercado los partidos de prueba, y simula en retrospectiva una estrategia de apuestas de valor comparando el modelo contra el mercado. |
| `src/visualizations.py` | Construye los gráficos Plotly (barras, radar, evolución de forma, mapas de calor, importancia de variables) a partir de datos ya calculados. |
| `src/utils.py` | Funciones auxiliares compartidas: división segura, formateo de porcentajes/métricas, logging, semilla aleatoria y umbrales de suficiencia de datos. |

## Formato de archivo esperado

El caso principal soportado es el formato **football-data.co.uk**, como el
archivo de referencia usado para construir este proyecto (`E0 (2).csv`,
Premier League inglesa, temporada 2025-2026, 380 partidos).

Columnas reconocidas automáticamente en ese formato:

```
Div, Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
HTHG, HTAG, HTR, Referee,
HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR
```

Notas sobre este formato:

- **Cuotas de apuestas**: todas las columnas de casas de apuestas (B365,
  Pinnacle, Max, Avg, Betfair Exchange, hándicap asiático, over/under 2.5,
  versiones de apertura y cierre) se descartan automáticamente al cargar
  el archivo — no se usan en ningún cálculo.
- **xG**: este formato no incluye Expected Goals. La aplicación funciona
  igual usando solo goles reales, e indica "xG no disponible" en vez de
  omitir la fila silenciosamente.
- **Competición**: la columna `Div` (ej. `"E0"`) se mapea directo a
  competición; si el archivo tiene un único valor, se asigna
  automáticamente sin pedir selección manual.
- **Temporada**: no hay columna de temporada explícita. Cada archivo se
  trata como una sola temporada y la etiqueta (ej. `"2025-2026"`) se
  infiere a partir del rango de fechas. Si cargas varios archivos de
  distintas temporadas, se concatenan respetando el orden cronológico.
- **HTHG/HTAG/HTR/Referee**: se detectan y se conservan en el dataframe,
  aunque no se usan todavía en ningún cálculo.

Para archivos con otros nombres de columna (español, snake_case, etc.), el
mapeo automático intenta reconocer variantes comunes (`Local`/`Equipo
Local`/`home_team`, etc.). Si una columna obligatoria no se puede
identificar, la aplicación pide seleccionarla manualmente desde un menú
desplegable en la barra lateral antes de continuar.

## Pruebas

```bash
pytest tests/ -v
```

## Reglas de diseño

- No se inventan estadísticas: cuando no hay datos suficientes para
  calcular una métrica, se muestra "Datos insuficientes" en vez de un
  valor inventado.
- No se usan datos futuros para predecir partidos históricos: los cálculos
  respetan el orden cronológico.
- Las semillas aleatorias están fijadas (`RANDOM_SEED = 42` en
  `src/utils.py`) para resultados reproducibles en las fases con modelos
  entrenados.
- Ninguna probabilidad mostrada garantiza el resultado real de un partido.

## Hoja de ruta

Las tres fases planeadas (base funcional, modelo de respaldo, exportación
e historial) están completas.

**Mejoras futuras** (no implementadas todavía, mencionadas pero fuera de
alcance por ahora): Random Forest / Gradient Boosting / XGBoost como
alternativas al modelo de respaldo, SHAP para explicabilidad, exportación
a PDF, matriz de marcadores como imagen (PNG) en vez de CSV, tema oscuro
nativo, mercados de córners y tarjetas cuando el archivo lo permita.
