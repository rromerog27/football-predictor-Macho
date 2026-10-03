"""Descarga y lectura de datos de mercado de EA SPORTS FC 27 Ultimate Team.

Fuente: FUT.GG. Sus páginas llegan renderizadas desde el servidor con los
datos embebidos en el HTML (objetos `{clave:valor,...}` de su framework), así
que basta una petición HTTP normal: no hace falta navegador ni su API privada
(que está protegida por Cloudflare).

Se leen tres páginas:
- `/players/momentum/?page=N`: las cartas con más movimiento en 24h (precio
  actual + variación porcentual de 24h según FUT.GG).
- `/cheapest-by-rating/`: las cartas más baratas por rating (fodder de SBC).
- `/sbc/`: los SBC activos (puntuación exigida, coste estimado, fecha de fin y
  carta de premio).

Las funciones `parse_*` son puras (reciben el HTML como texto) para poder
probarlas sin red. Ninguna función inventa datos: si un campo no está en la
página, queda como None.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

BASE_URL = "https://www.fut.gg"
MOMENTUM_PAGES = 10
REQUEST_TIMEOUT_S = 20
REQUEST_PAUSE_S = 0.3
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)

# Puntos de Item Score que aporta cada rating oro en un SBC "Streamlined".
# 84, 85, 86, 87, 88 y 90 se comprobaron resolviendo las rutas de SBC que
# publica FUT.GG (p. ej. 1×85 + 106×84 = 90.000 puntos exactos); 83 y 89
# vienen de tablas de terceros.
ITEM_SCORE = {83: 410, 84: 830, 85: 2100, 86: 4100, 87: 5500, 88: 8300, 89: 11000, 90: 14000}
ITEM_SCORE_VERIFIED = {84, 85, 86, 87, 88, 90}

MOVER_COLUMNS = ["ea_id", "name", "overall", "rarity", "price", "pct_24h", "url", "created_at"]
# Columnas solo para mostrar (no se guardan en el historial): la imagen de la carta tal como la
# pinta FUT.GG (la variante de 300 px que usa su propia web) y el nombre corto impreso en la carta.
DISPLAY_COLUMNS = ["card_image", "card_name"]
CARD_IMAGE_URL = "https://game-assets.fut.gg/cdn-cgi/image/quality=85,format=auto,width=300/"
CHEAPEST_COLUMNS = ["ea_id", "name", "overall", "price"]
SBC_COLUMNS = [
    "slug", "name", "category", "url", "end_time", "created_at", "repeatable",
    "score_requirement", "cost", "cost_pc", "award_ea_id", "award_name",
    "award_overall", "award_rarity", "award_untradeable",
]


class FetchError(RuntimeError):
    """No se pudo descargar una página de FUT.GG (red, bloqueo o error HTTP)."""


@dataclass
class MarketSnapshot:
    """Una foto del mercado en un instante dado."""

    fetched_at: datetime
    movers: pd.DataFrame
    cheapest: pd.DataFrame
    sbcs: pd.DataFrame
    errors: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.movers.empty and self.cheapest.empty and self.sbcs.empty


# ---------------------------------------------------------------------------
# Lectura del HTML
# ---------------------------------------------------------------------------

_STR = r'"((?:[^"\\]|\\.)*)"'


def _unescape(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except ValueError:
        return raw


def _first(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _to_int(value: str | None) -> int | None:
    if value is None or value == "null":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _chunks(text: str, start_pattern: str) -> list[str]:
    """Divide el texto en trozos que empiezan en cada coincidencia del patrón."""
    starts = [m.start() for m in re.finditer(start_pattern, text)]
    return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]


def parse_momentum(html: str) -> pd.DataFrame:
    """Cartas de la página de momentum.

    FUT.GG guarda en `momentumPercentage` la variación de 24h con el signo
    invertido (en su web, 34,84 se muestra como "−34,84%"), así que aquí se
    cambia el signo: `pct_24h` positivo = el precio subió.
    """
    rows = []
    for chunk in _chunks(html, r"momentumPercentage:"):
        pct = _first(r"^momentumPercentage:(-?[\d.]+|null)", chunk)
        price = _to_int(_first(r"currentDbPrice:(\d+|null)", chunk))
        ea_id = _to_int(_first(r"eaId:(\d+),overall:", chunk))
        # Precio 0 = FUT.GG no tiene precio de mercado para la carta: se trata como dato que falta.
        if pct in (None, "null") or not price or ea_id is None:
            continue
        card_name = _first(r"cardName:" + _STR, chunk)
        name = _first(r"commonName:" + _STR, chunk) or card_name
        image = _first(r'cardImagePath:"([^"]+)"', chunk)
        rows.append({
            "ea_id": ea_id,
            "name": _unescape(name) if name else str(ea_id),
            "overall": _to_int(_first(r"eaId:\d+,overall:(\d+)", chunk)),
            "rarity": _unescape(_first(r"rarityName:" + _STR, chunk) or ""),
            "price": price,
            "pct_24h": round(-float(pct), 2),
            "url": _first(r'url:"(/players/[^"]+)"', chunk),
            "created_at": _first(r'createdAt:"([^"]+)"', chunk),
            # Solo si la ruta es de esta misma carta (el nombre del archivo lleva su eaId).
            "card_image": CARD_IMAGE_URL + image if image and f"-{ea_id}." in image else None,
            "card_name": _unescape(card_name) if card_name else None,
        })
    df = pd.DataFrame(rows, columns=[*MOVER_COLUMNS, *DISPLAY_COLUMNS])
    return df.drop_duplicates("ea_id").reset_index(drop=True)


def parse_cheapest(html: str) -> pd.DataFrame:
    """Cartas más baratas por rating (página "Cheapest by Rating")."""
    pattern = r"\{price:(\d+),eaId:(\d+),name:" + _STR + r",overall:(\d+)"
    rows = [
        {"ea_id": int(ea), "name": _unescape(name), "overall": int(ovr), "price": int(price)}
        for price, ea, name, ovr in re.findall(pattern, html)
    ]
    df = pd.DataFrame(rows, columns=CHEAPEST_COLUMNS)
    return df.drop_duplicates("ea_id").sort_values(["overall", "price"]).reset_index(drop=True)


def parse_sbcs(html: str) -> pd.DataFrame:
    """SBC activos de la página `/sbc/`."""
    start = r'\{id:\d+,game:"27",eaId:\d+,slug:"'
    rows = []
    for chunk in _chunks(html, start):
        slug = _first(r'slug:"([^"]+)"', chunk)
        name = _first(r"categoryEaId:\d+,name:" + _STR, chunk)
        if not slug or name is None:
            continue
        url = _first(r'url:"(/sbc/[^"]+)"', chunk)
        category = url.split("/")[2] if url and url.count("/") >= 3 else None
        award = re.search(
            r"isUntradeable:(!0|!1)[^{}]*?playerEaId:(\d+)[^{}]*?player:\$R\[\d+\]=\{.*?"
            r"eaId:(\d+),overall:(\d+),commonName:" + _STR + r".*?rarityName:" + _STR,
            chunk,
        )
        rows.append({
            "slug": slug,
            "name": _unescape(name),
            "category": category,
            "url": url,
            "end_time": _first(r'endTime:"([^"]+)"', chunk),
            "created_at": _first(r'createdAt:"([^"]+)"', chunk),
            "repeatable": _first(r"isRepeatable:(!0|!1)", chunk) == "!0",
            "score_requirement": _to_int(_first(r"scoreRequirement:(\d+|null)", chunk)),
            "cost": _to_int(_first(r"cost:(\d+|null)", chunk)),
            "cost_pc": _to_int(_first(r"costPc:(\d+|null)", chunk)),
            "award_ea_id": int(award.group(3)) if award else None,
            "award_name": _unescape(award.group(5)) if award else None,
            "award_overall": int(award.group(4)) if award else None,
            "award_rarity": _unescape(award.group(6)) if award else None,
            "award_untradeable": (award.group(1) == "!0") if award else None,
        })
    df = pd.DataFrame(rows, columns=SBC_COLUMNS)
    return df.drop_duplicates("slug").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------


def fetch_html(path: str) -> str:
    """Descarga una página de FUT.GG. Lanza FetchError si falla."""
    url = path if path.startswith("http") else BASE_URL + path
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return response.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        hint = " (FUT.GG pidió una verificación anti-bots)" if exc.code in (403, 503) else ""
        raise FetchError(f"{url}: HTTP {exc.code}{hint}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"{url}: {exc}") from exc


def fetch_market_snapshot(momentum_pages: int = MOMENTUM_PAGES) -> MarketSnapshot:
    """Descarga momentum, más baratos por rating y SBC.

    Si una página falla, se anota en `errors` y se sigue con el resto: una
    instantánea parcial es más útil que ninguna.
    """
    errors: list[str] = []
    mover_frames = []
    for page in range(1, momentum_pages + 1):
        try:
            mover_frames.append(parse_momentum(fetch_html(f"/players/momentum/?page={page}")))
        except FetchError as exc:
            errors.append(str(exc))
            if page == 1:
                break  # si falla la primera, las demás también fallarán
        time.sleep(REQUEST_PAUSE_S)
    movers = (
        pd.concat(mover_frames, ignore_index=True).drop_duplicates("ea_id").reset_index(drop=True)
        if mover_frames else pd.DataFrame(columns=MOVER_COLUMNS)
    )

    try:
        cheapest = parse_cheapest(fetch_html("/cheapest-by-rating/"))
    except FetchError as exc:
        errors.append(str(exc))
        cheapest = pd.DataFrame(columns=CHEAPEST_COLUMNS)

    try:
        sbcs = parse_sbcs(fetch_html("/sbc/"))
    except FetchError as exc:
        errors.append(str(exc))
        sbcs = pd.DataFrame(columns=SBC_COLUMNS)

    for e in errors:
        logger.warning(e)
    return MarketSnapshot(datetime.now(timezone.utc), movers, cheapest, sbcs, errors)


def fodder_table(cheapest: pd.DataFrame, sample: int = 5) -> pd.DataFrame:
    """Coste por punto de Item Score de cada rating.

    El precio de referencia es la mediana de las `sample` cartas más baratas
    del rating, no la más barata sola: un SBC necesita varias cartas y una
    única oferta suelta (p. ej. un 86 a 2.000 cuando el resto está a 3.800)
    no representa lo que pagarías de verdad.

    Columnas: overall, name (la más barata), min_price, price (referencia),
    item_score, coins_per_point, score_verified, is_best. Solo incluye
    ratings con Item Score conocido.
    """
    cols = ["overall", "name", "min_price", "price", "item_score", "coins_per_point", "score_verified", "is_best"]
    if cheapest.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for ovr, group in cheapest.sort_values("price").groupby("overall"):
        if ovr not in ITEM_SCORE:
            continue
        top = group.head(sample)
        rows.append({"overall": int(ovr), "name": top.iloc[0]["name"], "min_price": int(top.iloc[0]["price"]),
                     "price": int(top["price"].median())})
    best = pd.DataFrame(rows, columns=["overall", "name", "min_price", "price"])
    if best.empty:
        return pd.DataFrame(columns=cols)
    best["item_score"] = best["overall"].map(ITEM_SCORE)
    best["coins_per_point"] = (best["price"] / best["item_score"]).round(3)
    best["score_verified"] = best["overall"].isin(ITEM_SCORE_VERIFIED)
    best["is_best"] = best["coins_per_point"] == best["coins_per_point"].min()
    return best[cols].sort_values("overall").reset_index(drop=True)


def fodder_floor_price(cheapest: pd.DataFrame) -> int | None:
    """Precio mínimo observado entre las cartas oro de 81-84.

    En FC 27 todas suelen estar clavadas en el mismo valor, que coincide con
    el mínimo del price range de EA. Es una estimación, no un dato oficial.
    """
    low = cheapest[cheapest["overall"].between(81, 84)]
    return int(low["price"].min()) if not low.empty else None
