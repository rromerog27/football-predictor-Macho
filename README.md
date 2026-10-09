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

La página **Partidos del día** también usa internet: descarga resultados y
xG (Understat) o tiros y cuotas (ESPN) de 33 ligas y copas y muestra, para
los partidos de la fecha elegida, la predicción de un modelo de Poisson
(Dixon-Coles) comparada con el mercado; también
permite correr el modelo para cualquier cruce de una competición. El mismo modelo se puede
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

- Python 3.11 o más nuevo (las pruebas pasan con 3.11, 3.12 y 3.13).

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

## Publicar en Streamlit Community Cloud

La app funciona tal cual en [Streamlit Community Cloud](https://share.streamlit.io)
(gratis): no usa claves ni secretos, la zona horaria sale del navegador de cada
visitante y las cachés se escriben junto al código.

1. Entra en [share.streamlit.io](https://share.streamlit.io) con tu cuenta de
   GitHub y autoriza el acceso al repositorio.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repositorio `rromerog27/football-predictor-Macho`, rama `master`, archivo
   principal `app.py`.
4. En **App URL** elige la dirección (por ejemplo
   `football-predictor-macho.streamlit.app`).
5. En **Advanced settings**, Python 3.12 o 3.13. No hace falta cargar secretos.
6. **Deploy**. Streamlit instala `requirements.txt` (las dependencias de
   pruebas están aparte, en `requirements-dev.txt`).

Qué esperar:

- **Primera visita.** El servidor arranca sin datos: descarga el historial de
  las ligas del día y entrena sus modelos. Medido desde cero: unos 70 segundos
  un viernes con 7 ligas; mientras tanto la página muestra qué modelo está
  preparando. Las visitas siguientes salen de la caché en 1-2 segundos, y los
  modelos se vuelven a entrenar cada 3 horas para sumar los resultados nuevos.
- **Reposo.** Si nadie la abre en 12 horas, Streamlit la pone a dormir. La
  siguiente visita la despierta (con un botón) y vuelve a empezar sin caché.
- **Actualizaciones.** Cada cambio que llega a `master` se publica solo.
  `app.py` recarga los módulos de `src/` que cambiaron, así no hace falta
  reiniciar la app.
- **Memoria.** Unos 600-700 MB con un día cargado (pico medido: 720 MB con
  11 ligas), por debajo del límite de Community Cloud (2,7 GB).
- **Mercado FC 27.** El historial (`data/fc27_market.sqlite`) se guarda en el
  servidor solo mientras la app está despierta: se pierde al dormirse o
  reiniciarse, y las instantáneas programadas de `scripts/` son para tu
  computadora. Si FUT.GG no responde desde el servidor, la sección lo avisa.

## Estructura del proyecto

```
football_predictor/
│
├── app.py                    # Punto de entrada: configuración y navegación entre páginas
├── views/
│   ├── fc27_mercado.py       # Página Mercado FC 27
│   ├── partidos_del_dia.py   # Página Partidos del día (predicciones de 33 ligas y copas)
│   └── predictor.py          # Página Predictor de partidos
├── modelo_prediccion.py      # Predicción por terminal (el modelo de Partidos del día)
├── requirements.txt          # Dependencias de la app (las que instala Streamlit Cloud)
├── requirements-dev.txt      # requirements.txt + pytest, para correr las pruebas
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
│   ├── match_model.py          # Modelo Poisson (Dixon-Coles), independiente de la fuente
│   ├── competitions.py         # Registro de ligas y copas, carga de datos, calendario y cuotas del día
│   ├── understat_source.py     # Descarga de Understat (xG)
│   ├── espn_source.py          # Descarga de ESPN (resultados, tiros, calendario, cuotas)
│   ├── football_data_source.py # Descarga de football-data.co.uk (cuotas de cierre: señal de mercado y backtest)
│   ├── market_signal.py        # Goles esperados implícitos en las cuotas de cierre de partidos anteriores
│   ├── lineups.py              # Alineaciones de ESPN: rotación de cada equipo y ajuste del 1X2
│   ├── backtest.py             # Backtest de las predicciones de validación contra las cuotas de cierre
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
    └── test_modelo_prediccion.py # Pruebas sin red del modelo, sus fuentes (Understat, ESPN) y el script
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
| `views/partidos_del_dia.py` | Página Partidos del día: lista los partidos de la fecha (en la zona horaria del navegador) con la predicción y el mercado de cada uno, y la sección "Analizar un partido". Solo presenta: el modelo vive en `src/match_model.py`. |
| `src/match_model.py` | Modelo independiente de la fuente: fuerza ajustada por rival (señal y goles), Poisson con Dixon-Coles, validación y predicción de un partido (con las probabilidades del mercado si hay cuotas). |
| `src/competitions.py` | Registro de las 33 ligas y copas (fuente, región, pool de ligas de las copas), carga de sus datos, partidos del día y cruce de cuotas de ESPN con los partidos de Understat. |
| `src/understat_source.py` | Descarga y caché del xG de Understat, convertido a las columnas estándar del modelo. |
| `src/football_data_source.py` | Descarga y caché de football-data.co.uk: resultados con cuotas de cierre (Pinnacle o media del mercado, 1X2 y Over/Under), descartando cuotas corruptas. |
| `src/market_signal.py` | Despeja los goles esperados que implican las cuotas de cierre de cada partido jugado (football-data.co.uk o ESPN), empareja nombres por resultados y los añade como señal de mercado. |
| `src/lineups.py` | Lee los titulares de ESPN (~1 h antes del partido), mide cuánto rota cada equipo respecto a sus 10 partidos anteriores y ajusta el 1X2 final. |
| `src/backtest.py` | Cruza las predicciones de validación con las cuotas de cierre (football-data.co.uk o ESPN) y compara log loss, mezcla modelo + mercado, Over/Under y ROI simulado. |
| `src/espn_source.py` | Descarga y caché del marcador de ESPN (resultados, tiros, calendario, campo neutral, cuotas de DraftKings) y cálculo del xG aproximado con tiros. |
| `src/visualizations.py` | Construye los gráficos Plotly (barras, radar, evolución de forma, mapas de calor, importancia de variables) a partir de datos ya calculados. |
| `src/utils.py` | Funciones auxiliares compartidas: división segura, formateo de porcentajes/métricas, logging, semilla aleatoria y umbrales de suficiencia de datos. |

## Sección FC 27 Mercado

Página `views/fc27_mercado.py` ("Mercado FC 27" en el menú lateral). Al
ejecutar `streamlit run app.py` se abre por defecto "Partidos del día".

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

La página **Partidos del día** es la que se abre por defecto al ejecutar
`streamlit run app.py`. Cubre **33 ligas y copas** (Europa, América, copas
internacionales y Japón). Tiene dos secciones:

- **Partidos del día:** un panel con la fecha (con ‹ › para cambiar de día),
  un buscador de equipos ("man u" encuentra Manchester United) y la
  selección de ligas y copas (por defecto 15: las 5 grandes, Portugal,
  Países Bajos, Liga MX, Argentina, Brasil, MLS, Champions, Europa League,
  Libertadores y Sudamericana). Debajo:
  - **Destacados**, en cuatro vistas: los favoritos más claros, los
    partidos más parejos, donde el modelo más se aleja del mercado y, cerca
    del inicio, los partidos con alineaciones confirmadas. Cada tarjeta lleva
    a su partido en la tabla.
  - **Todos los partidos**, en una tabla ordenable por hora, por liga
    (agrupada) o por favorito. Cada fila muestra los escudos (de ESPN), la
    barra 1X2, Over 2.5 y el pronóstico ("Parejo" si ningún resultado pasa
    del 45%). Una etiqueta amarilla ("+11 al 1") marca dónde el modelo da 10
    o más puntos más que el mercado (≈1 de cada 10 partidos; no es una
    recomendación de apuesta). Al tocar una fila se despliega el análisis:
    modelo, alineaciones, mercado, racha y señal de cada equipo.
  - En partidos ya jugados muestra el resultado y si el pronóstico acertó.
    Si no hay partidos ese día, ofrece saltar a la próxima fecha con
    partidos. `?fecha=AAAA-MM-DD` en la URL abre la página en ese día.
- **Analizar un partido:** elige liga o copa, local y visitante y pulsa
  "Correr modelo". Si el partido está en el calendario de los próximos 14
  días se usan su fecha y sus cuotas; si no, los datos disponibles hasta hoy.

Cada competición se entrena una vez y queda en caché 3 horas (unos segundos
por competición la primera vez); predecir cada partido es instantáneo. Las horas
y el "día" usan la zona horaria del navegador (se puede cambiar en la barra
lateral). Toda la app arranca en el modo claro u oscuro del sistema, y el
botón de luna/sol de arriba a la derecha (en todas las páginas, también en el
celular) cambia de modo: la elección queda guardada en ese navegador y se
aplica al volver a entrar. `.streamlit/config.toml` define los dos temas y
`src/ui_theme.py` el botón y los colores de los componentes propios, que
siguen al tema activo.

Desde la terminal:

```bash
python3 modelo_prediccion.py "Arsenal" "Leeds" --liga "Premier League"
python3 modelo_prediccion.py "América" "Monterrey" --liga "Liga MX" --detalle   # + los 10 partidos de cada equipo
python3 modelo_prediccion.py "Real Madrid" "Bayern Munich" --liga Champions --refrescar
python3 modelo_prediccion.py "Arsenal" "Leeds" --backtest                         # + comparación con las cuotas de cierre
python3 modelo_prediccion.py --listar                                            # ligas y copas disponibles
```

**Datos** (`src/competitions.py`, `src/understat_source.py`, `src/espn_source.py`).

- **Understat** (xG real): Premier League, LaLiga, Bundesliga, Serie A,
  Ligue 1 y liga rusa. `https://understat.com/getLeagueData/<liga>/<temporada>`,
  temporada en curso y 5 anteriores.
- **ESPN** (resto de ligas y copas): el marcador público
  `https://site.api.espn.com/apis/site/v2/sports/soccer/<slug>/scoreboard`
  da por año natural los resultados con **tiros y tiros a puerta**, el
  calendario, el campo neutral y, en partidos por jugar, las **cuotas de
  DraftKings** (1X2 y Over/Under 2.5). Sin xG, la señal es un **xG
  aproximado** = 0.2295 · tiros a puerta + 0.0647 · tiros fuera, calibrado
  con una regresión del xG de Understat sobre los tiros de ESPN en 5.838
  partidos-equipo de las 5 grandes ligas (correlación con el xG real 0.73,
  frente a 0.61 de los goles). Uruguay y Paraguay no tienen tiros en ESPN:
  allí la señal son los goles. Si ESPN da 0 tiros a ambos equipos (lo hace
  en cientos de partidos de la Championship, la segunda argentina o Chipre),
  se toma como estadística ausente y ese partido usa los goles. ESPN también da el calendario del día y las
  cuotas de las ligas de Understat (cruzadas por hora y nombre de equipo).
- **Copas internacionales** (Champions, Europa League, Conference League,
  Libertadores, Sudamericana, Concacaf Champions Cup): se modelan junto con
  las ligas de sus participantes (20 ligas UEFA, 10 CONMEBOL, 5 CONCACAF),
  así la fuerza de cada equipo sale sobre todo de su liga y los cruces entre
  países calibran unas ligas frente a otras.
- **football-data.co.uk** (señal de mercado y backtest): resultados con
  cuotas de cierre de las ligas principales y sus segundas divisiones, y de
  Liga MX, MLS, Argentina, Brasil, Japón, Rusia, Grecia, Austria, Dinamarca,
  Noruega y Suecia.
- **Cuotas pasadas de ESPN**: la ficha de cada partido
  (`.../<slug>/summary?event=<id>`) guarda las cuotas previas de DraftKings
  desde finales de 2025 (tan precisas como las de cierre de football-data:
  mismo log loss en 200 partidos de la Premier). Se usan en las copas y en
  las ligas que football-data no cubre (Colombia, Chile, Perú, Ecuador,
  Uruguay, Paraguay...), una petición por partido la primera vez.
- Caché en `data/understat_cache/`, `data/espn_cache/` (con las cuotas
  pasadas en `odds_<slug>.json`) y `data/football_data_cache/`: lo pasado no
  caduca; la temporada o el año en curso, a las 3 horas (o con
  `--refrescar`). Si una descarga falla, el script se detiene y la página lo
  avisa: nunca se rellenan datos inventados. Las lesiones y sanciones solo
  entran a través de las alineaciones (cuando se publican) y, de forma
  indirecta, de las cuotas de cierre de los partidos anteriores.

**Recién ascendidos.** En las ligas que tienen segunda división en ESPN
(las 5 grandes, Championship, Serie B, Eredivisie, Süper Lig, Escocia,
Argentina, Brasil, Colombia y Chile) esa división entra como historial: un
ascendido llega con sus partidos de la temporada anterior. En las ligas de
Understat los nombres se emparejan entre fuentes ("Leeds United" ↔ "Leeds")
solo si se parecen mucho y nunca coinciden en la misma temporada. El salto de
división se calibra con datos: un recién llegado (menos de 10 partidos en la
competición en 12 meses) marca ~10-20% menos y recibe ~20-35% más de lo que
diría su fuerza en la división inferior.

**Modelo** (`src/match_model.py`).

1. *Fuerza ajustada por rival:* para cada señal (xG o xG aproximado, goles
   y, donde hay cuotas, mercado) se ajusta con los partidos de los 12 meses anteriores al día del
   partido el modelo multiplicativo `señal = media_sede · ataque ·
   defensa_rival` (ajuste proporcional iterativo, equivalente a máxima
   verosimilitud de Poisson). Cada partido pesa según su antigüedad (el peso
   se reduce a la mitad cada 120 días) y cada equipo arranca con 2 partidos
   "de media de la liga" que evitan fuerzas extremas o nulas.
2. *Señal de mercado* (`src/market_signal.py`): de cada partido anterior con
   cuotas de cierre se despejan los goles esperados λ local/visitante que
   reproducen su 1X2 y su Over/Under (Poisson con Dixon-Coles, mínimos
   cuadrados vectorizados). Esos λ recogen lo que el mercado sabía antes de
   cada partido (fichajes, lesiones, alineaciones) y forman la tercera señal.
   Nunca entran las cuotas del propio partido que se predice. Los nombres de
   football-data se emparejan con los de la fuente por resultados (misma
   fecha ±1 día y marcador), no por parecido de nombre.
3. *Mezcla de señales y forma:* la fuerza por señal se mezcla con la fuerza
   por goles y con la de mercado, y la de 12 meses con la de los últimos 10
   partidos. Los pesos se eligen con datos (máxima verosimilitud de los
   goles reales en el periodo de entrenamiento; se probó elegirlos por log
   loss 1X2 y fue peor). Si solo hay cuotas recientes (las de ESPN), el
   entrenamiento no puede estimar el peso del mercado y se usa el típico de
   las ligas con histórico completo (70%), en proporción a los partidos
   recientes de cada equipo que tienen cuotas.
4. *Calibraciones de nivel* (también con datos de entrenamiento): goles por
   competición y sede cuando se mezclan competiciones (κ; no en las copas,
   donde empeoraba la validación), recién llegados, y el total de goles de
   cada partido se acerca a la media de su competición (el producto ataque ×
   defensa exagera las diferencias de total entre partidos: sin esto, las
   probabilidades de Over 2.5 salían demasiado extremas).
5. *Poisson con Dixon-Coles:* `λ = goles medios de la competición por sede ·
   ataque · defensa rival` (en campo neutral, la media de ambas sedes), con
   la corrección de Dixon-Coles para los marcadores bajos (ρ por máxima
   verosimilitud). De la matriz de marcadores salen el 1X2, el Over/Under
   2.5, ambos anotan y los marcadores más probables.
6. *Favorito claro:* Poisson es prudente de más con los favoritos fuertes
   (cuando decía 64%, ganaban el 70%; con 74%, el 80%; con 83%, el 90%).
   Si un resultado pasa del 55%, su probabilidad se estira en log-odds,
   `logit(p') = logit(p) + 0.5 · (logit(p) − logit(0.55))`, y los otros dos
   se reparten el resto en la misma proporción; por debajo del 55% no cambia
   nada. Umbral y estiramiento se eligieron con validación "una liga afuera"
   (se ajustan con las demás competiciones y se miden en la que queda
   afuera) sobre 8.846 partidos con cuotas de 29 competiciones: 0.55 / 0.5
   salió en 27 de las 29. Resultado: −0.0011 ± 0.0005 de log loss en total;
   −0.0073 ± 0.0024 en los partidos con un favorito claro del mercado (≥60%,
   1 de cada 5) y −0.030 con favoritos de 75% o más; +0.0004 en los
   parejos. Mejora 18 de las 29 competiciones y el Over/Under (0.6791 →
   0.6788). Con favoritos de 60-70%, 70-80% y 80% o más el modelo ahora dice
   68%, 81% y 91%; ganaron el 70%, el 80% y el 90%.
7. *Alineaciones* (`src/lineups.py`): cuando ESPN publica los titulares
   (~1 h antes), la rotación de cada equipo es la suma de titularidades
   (en sus 10 partidos anteriores de la competición) de sus 11 habituales
   que no salen de titulares, dividida por la de esos 11. El 1X2 final se
   desplaza en log-odds local/visitante en ±0.85 · (rotación local −
   rotación visitante) / 2. La diferencia entre el mercado y el modelo
   correlaciona −0.28/−0.45 con las rotaciones (buena parte de la ventaja
   del cierre son las alineaciones); con un efecto común a todas las ligas
   (por liga es demasiado ruidoso), la validación cruzada en dos mitades
   sobre 5.871 partidos de 17 ligas mejora el log loss en 0.0021. Probado y
   descartado: anticipar las rotaciones con el calendario (copas a pocos
   días), que no las predice. Solo en ligas, desde 2 h antes del inicio
   hasta 3 h después.
   Con alineaciones, la matriz de Poisson se reescala para que reproduzca
   el 1X2 ajustado (Over/Under y ambos anotan).

Hasta octubre de 2026 había además una regresión logística (racha de puntos,
localía y descanso sobre la señal de Poisson) y un ensemble de ambos. Se
quitó: su peso se elegía con los mismos partidos con que se medía, lo que
inflaba su aporte. Elegido en una mitad de la validación y medido en la otra,
el ensemble daba 1.0089 frente a 1.0084 de Poisson solo (7.325 partidos de
21 ligas), y la logística sola era peor que Poisson en 17 de las 21. Sin ella,
cada competición entrena ~40% más rápido y nada se elige mirando la
validación.

En las ligas, las filas de entrenamiento son solo partidos de la propia
competición (la división inferior solo aporta historial). Se entrena con los
partidos en los que ambos equipos tienen al menos 5 partidos de historial y
se valida con todos, como se usa el modelo.

**Validación** (log loss 1X2 en el 30% más reciente; la referencia es
predecir siempre las frecuencias de 1/X/2 del entrenamiento): las 33
competiciones mejoran la referencia, de +0.013 (Uruguay, solo goles) a
+0.171 (Primeira Liga). Lo único elegido con esos partidos son las dos
constantes del favorito claro, y siempre dejando afuera la liga que se mide. La sección "Rendimiento del modelo" de la página
muestra la validación y el backtest de cada competición.

**Backtest contra el mercado** (`src/backtest.py`, `src/football_data_source.py`).
Las predicciones de validación se cruzan (fecha ±1 día, marcador y nombres)
con las **cuotas de cierre** de football-data.co.uk (Pinnacle; si no, la
media del mercado; se descartan cuotas corruptas con margen fuera de 0-25%)
en las 21 competiciones que publica: las 5 grandes y sus segundas
divisiones, liga rusa, Portugal, Países Bajos, Bélgica, Turquía, Escocia,
Liga MX, MLS, Argentina, Brasil y Japón. En las demás ligas y copas se usan
las cuotas previas que ESPN guarda en la ficha de cada partido (desde finales
de 2025; hacen falta al menos 50 partidos de validación con cuotas).

Resultado en 7.325 partidos de las 21 competiciones:

| | Modelo | Cierre del mercado |
|---|---|---|
| Log loss 1X2 | **1.0073** | 0.9944 |
| Log loss Over/Under 2.5 (5.331 partidos) | 0.6788 | 0.6730 |

- Distancia al cierre: +0.013 (+0.014 antes de calibrar al favorito
  claro). Ejemplos: Premier League 1.036 vs 1.018, LaLiga 0.961 vs 0.954,
  Bundesliga 0.966 vs 0.960, Primeira Liga 0.912 vs 0.904. LaLiga 2
  (+0.0001) y la liga rusa (−0.026) quedan a la par o por delante del
  cierre. En aciertos (el resultado más probable), 50% del modelo contra 51%
  del mercado.
- Las cifras anteriores de este README (1.0145 → 1.0072 al sumar la señal
  de mercado) se midieron con el ensemble Poisson/logística, cuyo peso se
  elegía con estos mismos partidos; con esa ventaja quitada, el modelo de
  entonces daba 1.0089. La señal de mercado mejoró 20 de las 21 ligas (las
  que más, Serie A −0.016, LaLiga 2 −0.017, Primeira Liga −0.014, LaLiga
  −0.013 y Süper Lig −0.011); esas comparaciones siguen valiendo porque
  ambos lados se midieron igual.
- Con cuotas de ESPN: Colombia +0.012, Chile −0.007 (mejor que el mercado),
  Perú +0.035, Ecuador +0.024, Uruguay +0.022, Paraguay +0.023,
  Libertadores +0.073 y Sudamericana +0.049 (52 y 63 partidos). En esas
  competiciones la señal de mercado mejoró la validación de −0.001
  (Colombia, Paraguay) a −0.027 (Sudamericana). En las copas UEFA, con
  menos de 30 partidos con cuotas por copa, no hay backtest.
- Mezclar modelo y mercado sigue sin mejorar al mercado solo: el cierre ya
  contiene la información del modelo. El ROI simulado apostando cuando el
  modelo se aleja del mercado es negativo en casi todas las ligas.
- Probado y descartado, sobre los mismos partidos: vida media propia para
  la señal de mercado (60 o 30 días: ±0.0002), forma reciente propia para
  el mercado (−0.0002), elegir los pesos por log loss 1X2 en lugar de
  verosimilitud de goles (+0.0013, peor), cuotas recientes de ESPN en las
  divisiones inferiores (empeoraba Argentina, Países Bajos y Serie B) y
  añadir a las copas UEFA las ligas que ESPN no tiene (Suiza, Rumanía,
  Polonia, Finlandia e Irlanda, desde football-data: peor en las tres
  copas), estirar la diferencia local/visitante (−0.0003: incluso la mejor
  calibración a posteriori gana solo 0.0004, así que la prudencia del
  modelo con los favoritos es información que no tiene, no mala
  calibración) y descontar la temporada anterior tras el parón (−0.0001).
- Probado y descartado en octubre de 2026 (diferencia total sobre los
  mismos 7.325 partidos ± su error estándar; nada supera el ruido):
  - xG aproximado con los penales convertidos: se parece más al xG real
    (correlación 0.744 → 0.776 en 15.888 partidos-equipo de las 5 grandes,
    fuera de muestra) pero no predice mejor (+0.00002 ± 0.00015; sin
    penales, −0.00007 ± 0.00010). Córners, posesión y pases de tiro casi no
    lo acercan al xG real (+0.002 de correlación entre los tres).
  - Pesar menos los partidos con una roja temprana (antes del minuto 60 o
    75, peso 0.5 o 0.2): entre +0.00002 y +0.00005.
  - Localía propia de cada equipo (altura, viajes): no se repite de una
    temporada a la siguiente (correlación entre temporadas ~0 en Liga MX,
    MLS, Colombia, Brasil y las 5 grandes): es ruido.
  - Parámetros comunes a todas las ligas en vez de los de cada una: ρ de
    Dixon-Coles −0.06 (+0.0003), peso de la forma reciente (+0.00004),
    acercamiento del total (−0.00001); ventana de 18 meses (+0.0011, peor);
    1 o 4 partidos de suavizado inicial (±0.0001).
  - xG real de Understat en lugar del aproximado: en las 5 grandes da lo
    mismo (−0.0009 ± 0.0015), así que llevarlo a las copas no cambiaría nada.
  - Descendidos en las segundas divisiones (Championship, LaLiga 2,
    2. Bundesliga, Serie B, Ligue 2): meter la primera división en el
    cálculo de fuerzas empeora (+0.003, y sobre todo en los partidos sin
    descendidos: mezclar divisiones cambia las referencias de todos los
    equipos; Serie B +0.012). Usarla solo para reconocer a los descendidos
    y darles una calibración propia también empeora (+0.0014; Ligue 2
    +0.011): con 2 o 3 descendidos por temporada y liga, lo que se aprende
    es ruido. En 88 partidos de validación con un descendido, este ganó el
    42,0%; el modelo le daba 41,4% y el mercado 44,7%. El modelo no los
    subestima: el mercado los sobrevalora.
  - La calibración ya es buena (empates y favoritos salen con frecuencias
    muy cercanas a las reales, igual que en el mercado) y mezclar el modelo
    con las cuotas no mejora a las cuotas solas (+0.0014): la distancia al
    cierre es información que el modelo no tiene (lesiones y alineaciones
    antes de que se publiquen; ESPN no publica lesionados de fútbol).
- Versión anterior: variables de Poisson en la logística (−0.0004),
  decaimiento temporal (−0.0006), historial de la división inferior
  (−0.0006), acercamiento del total (Over/Under 0.6851 → 0.6802).

Desde la terminal: `python3 modelo_prediccion.py "Arsenal" "Leeds" --backtest`.

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
pip install -r requirements-dev.txt
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
