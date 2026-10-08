"""Fuente football-data.co.uk: resultados con cuotas de cierre (para el backtest).

- Ligas principales: `https://football-data.co.uk/mmz4281/<aaaa>/<div>.csv`
  (una temporada por archivo; p. ej. `2526/E0.csv` = Premier League 2025/26).
- Ligas "extra" (Liga MX, Argentina, Brasil, MLS...): `https://football-data.co.uk/new/<PAIS>.csv`
  (todas las temporadas en un archivo).

Cuotas: se toma la de cierre más precisa disponible (Pinnacle; si no, la media
del mercado; si no, Bet365) y, en las ligas principales, también Over/Under
2.5. Caché en `data/football_data_cache/`: temporadas pasadas sin caducidad; la
actual y los archivos "extra", a las 6 horas.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.match_model import PredictionError

FOOTBALL_DATA = "https://football-data.co.uk"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) football-predictor-macho"}
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "football_data_cache"
CURRENT_CACHE_TTL_S = 6 * 3600

# Código de competición (slug de ESPN) → división de football-data.
MAIN_DIVISIONS = {
    "eng.1": "E0", "eng.2": "E1", "esp.1": "SP1", "esp.2": "SP2", "ger.1": "D1", "ger.2": "D2",
    "ita.1": "I1", "ita.2": "I2", "fra.1": "F1", "fra.2": "F2", "ned.1": "N1", "por.1": "P1",
    "bel.1": "B1", "tur.1": "T1", "sco.1": "SC0",
}
EXTRA_LEAGUES = {"mex.1": "MEX", "arg.1": "ARG", "bra.1": "BRA", "usa.1": "USA", "jpn.1": "JPN", "rus.1": "RUS"}

# Columnas de cuotas por orden de preferencia: (nombre, local, empate, visitante).
ODDS_1X2 = [("Pinnacle (cierre)", "PSCH", "PSCD", "PSCA"), ("media del mercado (cierre)", "AvgCH", "AvgCD", "AvgCA"),
            ("Bet365 (cierre)", "B365CH", "B365CD", "B365CA"), ("media del mercado", "AvgH", "AvgD", "AvgA")]
MAX_OVERROUND = 1.25  # suma de probabilidades implícitas máxima creíble (margen del 25%)
ODDS_OU = [("PC>2.5", "PC<2.5"), ("AvgC>2.5", "AvgC<2.5"), ("B365C>2.5", "B365C<2.5"), ("Avg>2.5", "Avg<2.5")]


def has_odds(code: str) -> bool:
    return code in MAIN_DIVISIONS or code in EXTRA_LEAGUES


def _season_code(season: int) -> str:
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def _download(url: str, cache_file: Path, permanent: bool) -> pd.DataFrame | None:
    if cache_file.exists() and (permanent or time.time() - cache_file.stat().st_mtime < CURRENT_CACHE_TTL_S):
        text = cache_file.read_text(encoding="utf-8")
    else:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=HTTP_HEADERS, timeout=60)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                text = resp.content.decode("utf-8-sig", errors="replace")
                break
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2**attempt)
        else:
            raise PredictionError(f"No se pudo descargar {url}: {last_error}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="utf-8")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip")


def _pick_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Cuotas 1X2 fila a fila según la preferencia de ODDS_1X2 (y la fuente usada)."""
    out = pd.DataFrame(np.nan, index=df.index, columns=["odds_h", "odds_d", "odds_a"])
    source = pd.Series("", index=df.index, dtype=object)
    for name, h, d, a in ODDS_1X2:
        if not {h, d, a} <= set(df.columns):
            continue
        cand = df[[h, d, a]].apply(pd.to_numeric, errors="coerce")
        overround = (1 / cand).sum(axis=1)  # 1 + margen de la casa: fuera de [1, 1.25] son cuotas corruptas
        fill = (out["odds_h"].isna() & cand.notna().all(axis=1) & (cand > 1).all(axis=1)
                & overround.between(1.0, MAX_OVERROUND))
        out.loc[fill, ["odds_h", "odds_d", "odds_a"]] = cand[fill].to_numpy()
        source[fill] = name
    return out, source


def _pick_over_under(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=df.index, columns=["odds_over25", "odds_under25"])
    for over, under in ODDS_OU:
        if not {over, under} <= set(df.columns):
            continue
        cand = df[[over, under]].apply(pd.to_numeric, errors="coerce")
        fill = (out["odds_over25"].isna() & cand.notna().all(axis=1) & (cand > 1).all(axis=1)
                & (1 / cand).sum(axis=1).between(1.0, MAX_OVERROUND))
        out.loc[fill, ["odds_over25", "odds_under25"]] = cand[fill].to_numpy()
    return out


def _standardize(df: pd.DataFrame, home: str, away: str, hg: str, ag: str) -> pd.DataFrame:
    df = df.dropna(subset=[home, away, hg, ag]).reset_index(drop=True)
    odds, source = _pick_odds(df)
    return pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.normalize(),
        "home": df[home].astype(str).str.strip(),
        "away": df[away].astype(str).str.strip(),
        "hg": pd.to_numeric(df[hg], errors="coerce"),
        "ag": pd.to_numeric(df[ag], errors="coerce"),
        **{c: odds[c] for c in odds.columns},
        **{c: v for c, v in _pick_over_under(df).items()},
        "odds_source": source,
    }).dropna(subset=["date", "hg", "ag"])


def load(code: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Partidos con cuotas de cierre entre `start` y `end` (fechas). Devuelve (partidos, URLs)."""
    now = pd.Timestamp.now()
    current_season = now.year if now.month >= 7 else now.year - 1
    frames, urls = [], []
    if code in MAIN_DIVISIONS:
        div = MAIN_DIVISIONS[code]
        for season in range(start.year - 1, min(end.year, current_season) + 1):
            url = f"{FOOTBALL_DATA}/mmz4281/{_season_code(season)}/{div}.csv"
            raw = _download(url, CACHE_DIR / f"{div}_{_season_code(season)}.csv", season < current_season)
            if raw is not None and len(raw):
                frames.append(_standardize(raw, "HomeTeam", "AwayTeam", "FTHG", "FTAG"))
                urls.append(url)
    elif code in EXTRA_LEAGUES:
        url = f"{FOOTBALL_DATA}/new/{EXTRA_LEAGUES[code]}.csv"
        raw = _download(url, CACHE_DIR / f"extra_{EXTRA_LEAGUES[code]}.csv", False)
        if raw is not None and len(raw):
            frames.append(_standardize(raw, "Home", "Away", "HG", "AG"))
            urls.append(url)
    else:
        raise PredictionError(f"football-data.co.uk no tiene cuotas de {code}.")
    if not frames:
        raise PredictionError(f"football-data.co.uk no devolvió partidos de {code}.")
    df = pd.concat(frames, ignore_index=True)
    window = (df["date"] >= start.normalize() - pd.Timedelta(days=1)) & (df["date"] <= end.normalize() + pd.Timedelta(days=1))
    return df[window].reset_index(drop=True), urls
