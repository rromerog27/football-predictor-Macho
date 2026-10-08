# Football Predictor

Aplicación web local (Streamlit) para cargar archivos de partidos de fútbol
(Excel o CSV), analizar el rendimiento de los equipos y generar predicciones
estadísticas de un partido mediante un modelo de distribución de Poisson.

**Todo el análisis se basa únicamente en los datos del archivo que cargues.**
No se consulta internet, rankings, lesiones, alineaciones ni datos
históricos externos.

La app incluye además una sección independiente, **FC 27 Mercado**, que sí
usa internet: descarga datos en vivo de FUT.GG para buscar oportunidades de
trading en EA SPORTS FC 27 Ultimate Team (ver "Sección FC 27 Mercado" más
abajo). No comparte datos con el predictor de partidos.

La página **Partidos del día** también usa internet: descarga de Understat el
xG partido a partido y muestra, para los partidos de la fecha elegida, la
predicción de un ensemble Poisson + regresión logística; también permite
correr el modelo para cualquier cruce de una liga. El mismo modelo se puede
usar desde la terminal con **`modelo_prediccion.py`** (ver "Partidos del día
y predicción por terminal" más abajo). No comparte datos con el predictor de
archivos.

## Estado actual: Fase 3 — Exportación (completa)

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
├── app.py                    # Punto de entrada: configuración y navegación entre páginas
├── views/
│   ├── fc27_mercado.py       # Página Mercado FC 27
│   ├── partidos_del_dia.py   # Página Partidos del día (predicciones con xG de Understat)
│   └── predictor.py          # Página Predictor de partidos
├── modelo_prediccion.py      # Predicción por terminal con xG de Understat (Poisson + logística)
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Este archivo
│
├── data/                     # Archivos de datos locales (ejemplo)
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
│   ├── market_odds.py          # Comparación contra cuotas de mercado + simulación de value bets
│   ├── understat_model.py      # Descarga de Understat + modelo Poisson/logística (Partidos del día y terminal)
│   ├── visualizations.py       # Construcción de gráficos Plotly
│   └── utils.py                 # Utilidades comunes (safe_divide, logging, formateo)
│
└── tests/
    ├── test_data_loader.py     # Pruebas de carga y limpieza de columnas de cuotas
    ├── test_statistics.py      # Pruebas de cálculo de estadísticas
    ├── test_predictions.py     # Pruebas de feature engineering, regresión logística y combinación
    ├── test_reports.py         # Pruebas de exportación (CSV/Excel/HTML)
    ├── test_market_odds.py     # Pruebas de probabilidad implícita y simulación de apuestas de valor
    ├── test_fc27.py            # Pruebas de la sección FC 27 Mercado (lectura, historial, señales)
    └── test_modelo_prediccion.py # Pruebas sin red del script de predicción con Understat
```

### Para qué sirve cada archivo

| Archivo | Responsabilidad |
|---|---|
| `app.py` | Punto de entrada: configuración común y navegación (`st.navigation`) entre las dos páginas. Recarga los módulos de `src/` que cambiaron en disco (Streamlit Cloud puede actualizar el código sin reiniciar el proceso). |
| `views/predictor.py` | Página del predictor: sidebar, pestañas, y llama a las funciones de `src/` para mostrar resultados. No contiene lógica de negocio. |
| `src/data_loader.py` | Lee archivos Excel/CSV, lista hojas de Excel, descarta columnas de cuotas de apuestas cuando detecta el formato football-data.co.uk, y genera el resumen técnico del archivo (filas, columnas, faltantes, duplicados). |
| `src/column_mapper.py` | Detecta automáticamente qué columna del archivo corresponde a cada campo canónico (equipo local, goles, fecha, etc.), con nombres alternativos como respaldo. Infiere la temporada a partir de las fechas y detecta la competición cuando hay un único valor. |
| `src/data_cleaner.py` | Limpia strings, convierte fechas y columnas numéricas, detecta valores imposibles, elimina duplicados, separa partidos jugados de partidos pendientes y ordena cronológicamente. |
| `src/statistics.py` | Calcula el récord, promedios de goles, mercados (%btts, over/under), forma reciente, enfrentamientos directos y el resumen general del dataset. |
| `src/poisson_model.py` | Calcula fortalezas de ataque/defensa relativas a la liga, goles esperados, matriz de marcadores, mercados derivados (1X2, over/under, ambos anotan, etc.) y el backtest cronológico usado para comparar con la regresión logística. |
| `src/feature_engineering.py` | Construye las variables pre-partido (forma, rendimiento local/visitante, descanso) para cada fila de entrenamiento, usando solo partidos anteriores a la fecha del partido (sin fuga de información). |
| `src/prediction_model.py` | Entrena y calibra la regresión logística de respaldo, calcula sus métricas de validación, y combina sus probabilidades con las de Poisson ponderando por desempeño de validación (log loss). |
| `src/report_generator.py` | Exporta a CSV/Excel/HTML lo que ya calcularon los demás módulos: tabla de estadísticas, predicción de un partido, comparación de equipos, matriz de marcadores y el reporte HTML completo. |
| `src/market_odds.py` | Convierte cuotas 1X2 a probabilidad implícita (quitando el margen de la casa), evalúa qué tan bien predice el mercado los partidos de prueba, y simula en retrospectiva una estrategia de apuestas de valor comparando el modelo contra el mercado. |
| `views/partidos_del_dia.py` | Página Partidos del día: lista los partidos de la fecha (en la zona horaria del navegador) con la predicción de cada uno, y la sección "Analizar un partido". Solo presenta: el modelo vive en `src/understat_model.py`. |
| `src/understat_model.py` | Descarga el xG de Understat (con caché en disco), calcula la fuerza ajustada por rival, entrena Poisson + regresión logística por liga y predice partidos. Lo usan la página Partidos del día y `modelo_prediccion.py`. |
| `src/visualizations.py` | Construye los gráficos Plotly (barras, radar, evolución de forma, mapas de calor, importancia de variables) a partir de datos ya calculados. |
| `src/utils.py` | Funciones auxiliares compartidas: división segura, formateo de porcentajes/métricas, logging, semilla aleatoria y umbrales de suficiencia de datos. |

## Sección FC 27 Mercado

Página `views/fc27_mercado.py`. Es la página que se abre por defecto al
ejecutar `streamlit run app.py` ("Mercado FC 27" en el menú lateral).

Orden de la página: cabecera (mercado en vivo y última actualización),
**Resumen de hoy** (mejor compra, mayor riesgo y próximo evento, tres tarjetas
de la misma altura), una barra de estado (tono del mercado en 24h, cuántas
cartas suben y bajan, y cuántas señales hay de cada tipo) y las alertas
importantes agrupadas en un desplegable. Debajo, la navegación: **Señales**,
**Mercado** (mapa, tabla y ficha de la carta), **Mi lista** (cartas que
sigues, con tus precios de PC) y **Operaciones**; en **Más** están Fodder &
SBC, Alertas y Cómo funciona. Solo se dibuja la sección abierta. El selector
**Presupuesto** de la barra lateral filtra el resumen, las señales y el
mercado.

**Qué hace al abrirla:**

1. Descarga de FUT.GG las ~280 cartas con más movimiento en 24h, las cartas
   más baratas por rating (fodder) y los SBC activos (puntos exigidos, coste
   en consola y PC, fecha de fin y carta de premio). Reutiliza la descarga
   durante ~10 minutos; el botón **Actualizar ahora** fuerza una nueva y el
   interruptor **Actualizar cada 10 min** la repite mientras la página esté
   abierta.
2. Guarda una instantánea en `data/fc27_market.sqlite` (máximo una cada 10
   minutos; el archivo no se versiona).
3. Calcula para cada carta el **Market Score** (0-100), el **Risk Score**
   (0-100) y una señal **COMPRAR / VIGILAR / RIESGO**, con zona de compra,
   objetivo e invalidación.
4. Muestra alertas (SBC nuevos o por expirar, subidas verticales, sustitutos
   más baratos por SBC, caídas fuertes en 1h) y el calendario.

Si FUT.GG no responde, se muestra la última instantánea guardada con su
antigüedad.

**PC y consola.** FUT.GG solo publica abiertamente precios de cartas de
**consola**; sus precios de PC pasan por un servicio protegido contra
descargas automáticas, que la app no intenta saltarse. Por eso:

- Los precios de cartas se muestran marcados como "(consola)".
- La tabla de SBC muestra primero el coste en PC, que FUT.GG sí publica.
- En **⭐ Mi lista** apuntas a cuánto compraste cada carta en PC y cuánto vale
  ahora; la app calcula el precio para no perder (compra / 0,95), el objetivo
  (+10% neto), la invalidación (−10%) y el beneficio neto si vendes ya, todo
  con el 5% de impuesto de EA.

**Historial y calibración.** Las variaciones de 1h, 6h, 3 días y 7 días, y la
tasa de acierto de las señales, se calculan con el historial propio, así que
aparecen a medida que se acumulan instantáneas. Para que crezca sin abrir la
app:

- **Windows:** doble clic en `scripts/programar_snapshots_windows.bat`. Crea una
  tarea del Programador de tareas que guarda una instantánea cada 30 minutos,
  sin abrir ventanas, mientras tu sesión esté abierta. Para quitarla:
  `scripts/quitar_snapshots_windows.bat`.
- **Linux/macOS:** programa `python scripts/fc27_snapshot.py` con cron.

Cada ejecución deja una línea en `data/fc27_snapshot.log` (no se versiona).

**Visualizaciones.**
- *Señales*: filtro Todas / Comprar / Vigilar / Riesgo y rejilla de tarjetas
  (3, 2 o 1 por fila según el ancho) ordenadas por relevancia. Cada tarjeta
  muestra jugador, precio, señal, cambio de 24h, Market Score y Risk Score,
  la estrategia y el historial; **Ver análisis** despliega el plan, las
  razones y los riesgos, y **Ver ficha** abre la ficha completa con gráfico.
- *Mapa del mercado* (Mercado): filtros de jugador, rareza, movimiento y
  cuántas cartas ver (Top 50 por defecto, Top 100 o todas). El Top ordena por
  relevancia (movimiento de 24h, señal, Market Score, Risk Score y precio)
  solo para elegir qué se ve: no cambia ninguna puntuación. Bloques
  agrupados por rareza; tamaño según precio (escala logarítmica) y color
  según la variación de 24h; los bloques grandes muestran más datos.
- *Tabla del mercado*: vista básica (carta, precio corto, 24h, Market,
  Riesgo y señal) o avanzada (todas las columnas). Al seleccionar una fila se
  abre la ficha de mercado de la carta: cifras, plan, razones, riesgos y
  gráfico de precio.
- *Gráfico de cada carta* (ficha en Mercado y Mi lista): rango 24h / 7 días /
  todo, niveles del plan (objetivo, stop…), eventos del calendario y tus
  compras y ventas de Operaciones.
- *Índice de fodder* (Fodder & SBC): precio de referencia de los ratings 84,
  85 y 86 en tus instantáneas, normalizado a 100.

**Diseño.** Los colores, sombras y radios están en un solo sitio:
`TOKENS` en `src/ui_theme.py`, publicados como variables CSS `--fc-*` (verde
= comprar/sube, amarillo = vigilar, rojo = riesgo/baja, azul = información y
Market Score, gris = información secundaria). El tema de Streamlit
(tipografía Inter, bordes, radios) está en `.streamlit/config.toml`. Las
imágenes de las cartas y su nombre corto salen de la misma página de FUT.GG
que ya se descarga; solo se muestran (no se guardan en el historial) y, si
faltan, la carta se muestra con una insignia con su media.

**Más ayudas.**
- *🔔 Avisos de precio* (Mi lista): "avísame si baja de X / sube de Y" sobre el
  precio de consola; se muestran arriba de la página cuando se cumplen.
- *🆕 Desde tu última visita*: SBC nuevos, señales nuevas de compra o riesgo y
  avisos de precio que saltaron mientras no estabas.
- *💰 Curva de beneficio* (Operaciones): beneficio neto acumulado y caída desde
  el máximo.

**📒 Operaciones.** Diario de compras y ventas reales, con precios de PC por
unidad. Muestra el beneficio neto (con el 5% de EA), el ROI, el % de
operaciones ganadoras, el capital invertido en operaciones abiertas y qué
señal había al comprar, para ver qué señales te hacen ganar dinero.

**Calendario y notas del analista.** Las fechas de promos, rumores y tesis
manuales viven en `data/fc27_analyst.json`. Las notas desaparecen solas al
pasar su fecha `expires`; un evento de tipo `promo` a menos de 48h activa la
regla de "promo próxima" en las puntuaciones.

| Archivo | Responsabilidad |
|---|---|
| `src/fc27_market.py` | Descarga y lee las páginas de FUT.GG; tabla de coste por punto de Item Score del fodder. |
| `src/fc27_history.py` | Historial SQLite: instantáneas, variaciones por ventana, registro y evaluación de señales (con el 5% de impuesto de EA). |
| `src/fc27_signals.py` | Reglas de Market Score, Risk Score, señales, plan de operación, estado del mercado y alertas. |
| `views/fc27_mercado.py` | La página de Streamlit. |
| `scripts/fc27_snapshot.py` | Guarda una instantánea desde la línea de comandos. |
| `scripts/programar_snapshots_windows.bat` / `quitar_snapshots_windows.bat` | Activan o quitan la instantánea automática cada 30 min en Windows. |
| `data/fc27_analyst.json` | Calendario y notas mantenidos a mano. |

Las señales son heurísticas: el Market Score es un indicador comparativo, no
una probabilidad de ganar. FUTBIN no se usa porque bloquea las peticiones
automáticas.

## Partidos del día y predicción por terminal

La página **Partidos del día** (`streamlit run app.py` → "Fútbol real") tiene
dos secciones:

- **Partidos del día:** elige la fecha y las ligas; cada partido muestra la
  barra 1X2, Over 2.5, ambos anotan, el marcador más probable y, en "Ver
  análisis", Poisson vs. logística, la racha y el xG de cada equipo. En
  partidos ya jugados muestra el resultado y si el pronóstico acertó. Si no
  hay partidos ese día, ofrece saltar a la próxima fecha con partidos.
- **Analizar un partido:** elige liga, local y visitante y pulsa "Correr
  modelo". Si el partido está en el calendario de los próximos 14 días se
  usa su fecha; si no, los datos disponibles hasta hoy.

Cada liga se entrena una vez y queda en caché 3 horas (la primera visita
tarda unos segundos por liga); predecir cada partido es instantáneo. Las
horas y el "día" usan la zona horaria del navegador (se puede cambiar en la
barra lateral).

Desde la terminal:

```bash
python3 modelo_prediccion.py "Arsenal" "Leeds" --liga "Premier League"
python3 modelo_prediccion.py "Arsenal" "Leeds" --detalle      # + los 10 partidos usados de cada equipo
python3 modelo_prediccion.py "Real Madrid" "Barcelona" --liga LaLiga --refrescar
```

Ligas disponibles (las que cubre Understat): Premier League, LaLiga,
Bundesliga, Serie A, Ligue 1 y la Liga Premier de Rusia.

**Datos.** Descarga de `https://understat.com/getLeagueData/<liga>/<temporada>`
el calendario con goles y xG de cada partido de la temporada en curso y de
las 5 anteriores (la más antigua solo sirve de historial previo). Las
temporadas cerradas quedan en caché en `data/understat_cache/`; la temporada
en curso se vuelve a descargar pasadas 3 horas (o con `--refrescar`). Si la
descarga falla, el script se detiene: nunca rellena con datos inventados.
Understat no publica lesiones ni sanciones, y el descanso se calcula solo con
partidos de liga (no ve copas ni competiciones europeas).

**Modelo.**

1. *Fuerza ajustada por rival:* con los partidos de los últimos 12 meses se
   ajusta el modelo multiplicativo `xG = media_sede · ataque · defensa_rival`
   (ajuste proporcional iterativo, equivalente a máxima verosimilitud de
   Poisson sobre el xG).
2. *Forma reciente:* ataque/defensa de los últimos 10 partidos de cada equipo
   (xG real / xG esperado según sede y rival), mezclados con la fuerza de 12
   meses. El peso de la mezcla se elige con datos históricos (máxima
   verosimilitud de los goles reales en las temporadas de entrenamiento).
3. *Poisson puro:* `λ = goles medios reales de la liga por sede · ataque ·
   defensa rival`. El xG fija la fuerza relativa y los goles reales el nivel,
   porque el xG de Understat va por encima de los goles marcados en las
   últimas temporadas. De la matriz de marcadores salen el 1X2, el Over/Under
   2.5, ambos anotan y los marcadores más probables.
4. *Regresión logística multinomial:* reajusta la señal de Poisson
   (`log(λ local/λ visitante)`, `log(λ total)`) con la racha de puntos de los
   últimos 5 partidos, los últimos 5 en casa del local y fuera del visitante
   (localía) y el descanso en categorías (corto ≤4 días, normal 5-7, largo
   ≥8). Se entrena con instantáneas pre-partido sin fuga de información y
   elige su regularización con validación cruzada temporal.
5. *Ensemble:* el peso Poisson/logística minimiza el log loss sobre la
   temporada anterior y la actual, que la logística no vio al entrenar. Para
   Over/Under y ambos anotan, la matriz de Poisson se reescala para que
   reproduzca el 1X2 final.

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

Las tres fases planeadas (base funcional, modelo de respaldo, exportación)
están completas.

**Mejoras futuras** (no implementadas todavía, mencionadas pero fuera de
alcance por ahora): Random Forest / Gradient Boosting / XGBoost como
alternativas al modelo de respaldo, SHAP para explicabilidad, exportación
a PDF, matriz de marcadores como imagen (PNG) en vez de CSV, tema oscuro
nativo, mercados de córners y tarjetas cuando el archivo lo permita.
