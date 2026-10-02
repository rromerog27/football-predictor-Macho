"""Motor de señales del mercado de FC 27: Market Score, Risk Score y alertas.

Todo es determinista y está escrito como reglas legibles, para que cada
puntuación se pueda explicar ("¿por qué esta carta tiene 72?"). Las reglas
son heurísticas del analista, no un modelo entrenado: la calibración real
llega con el historial de señales (`fc27_history.evaluate_signals`).

Market Score (0-100) = Momentum 20 + Supply 15 + Demand 15 + Upcoming Content 15
+ SBC/Evo Utility 15 + News 10 + Risk/Downside 10. Es un indicador
comparativo, no una probabilidad.

Clasificación: 85-100 Señal muy fuerte · 70-84 Interesante · 55-69 Watchlist ·
40-54 Neutral · 0-39 Riesgo elevado.

Señal: COMPRAR si Market Score >= 70 y Risk Score <= 45; RIESGO si Risk
Score >= 70 o Market Score < 40; VIGILAR si Market Score >= 55; si no, nada.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.fc27_market import ITEM_SCORE, MarketSnapshot, fodder_floor_price, fodder_table

ANALYST_FILE = Path(__file__).resolve().parent.parent / "data" / "fc27_analyst.json"

PACK_CYCLE_DAYS = 7          # una carta TOTW/promo sale de packs ~7 días después
NEW_CARD_DAYS = 2
PROMO_WINDOW_H = 48          # una promo que sale en <48h afecta a las cartas especiales
ILLIQUID_PRICE = 1_000_000
VERY_ILLIQUID_PRICE = 5_000_000
PARABOLIC_PCT = 20.0
SUBSTITUTE_DISCOUNT = 0.85   # SBC sustituto si cuesta <85% del precio de la carta
NEWS_NEUTRAL = 5             # no hay noticias automáticas: valor neutro

BASE_RARITIES = {"Base Icon", "Base Hero"}

SCORE_COMPONENTS = [
    ("momentum", "Momentum", 20),
    ("supply", "Supply", 15),
    ("demand", "Demand", 15),
    ("upcoming", "Upcoming Content", 15),
    ("utility", "SBC/Evo Utility", 15),
    ("news", "News", 10),
    ("downside", "Risk/Downside", 10),
]

SIGNAL_COLUMNS = [
    "key", "ea_id", "name", "overall", "rarity", "price", "pct_24h", "pct_1h", "pct_6h", "pct_72h", "pct_168h",
    *[c for c, _, _ in SCORE_COMPONENTS], "market_score", "risk_score", "label", "signal",
    "horizon", "horizon_h", "reasons", "risks", "url",
]


@dataclass
class Alert:
    level: str      # "crítica", "aviso" o "info"
    status: str     # CONFIRMADO / ALTA / MEDIA / BAJA / RUMOR
    title: str
    detail: str
    category: str = "otra"   # sustituto, movimiento_1h, subida_24h, sbc, nuevas, evento
    ea_id: int | None = None  # carta afectada, si la hay (para filtrar por Mi lista)


# ---------------------------------------------------------------------------
# Archivo del analista (calendario y notas manuales)
# ---------------------------------------------------------------------------


def load_analyst_file(path: Path | str = ANALYST_FILE) -> dict:
    """Calendario y notas que el analista mantiene a mano en JSON."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {"events": [], "notes": []}
    data.setdefault("events", [])
    data.setdefault("notes", [])
    return data


def upcoming_events(analyst: dict, now: datetime, days: int = 30) -> pd.DataFrame:
    rows = []
    for e in analyst.get("events", []):
        try:
            when = datetime.fromisoformat(e["when"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if now - timedelta(hours=12) <= when <= now + timedelta(days=days):
            rows.append({**e, "when": when})
    df = pd.DataFrame(rows, columns=["when", "type", "label", "status", "source"])
    return df.sort_values("when").reset_index(drop=True)


def active_notes(analyst: dict, now: datetime) -> list[dict]:
    notes = []
    for n in analyst.get("notes", []):
        try:
            expires = datetime.fromisoformat(n["expires"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires >= now:
            notes.append(n)
    return notes


_DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
_MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def format_when(dt: datetime) -> str:
    """'vie 2 oct · 17:00 UTC'."""
    dt = dt.astimezone(timezone.utc)
    return f"{_DIAS[dt.weekday()]} {dt.day} {_MESES[dt.month - 1]} · {dt:%H:%M} UTC"


def _promo_soon(events: pd.DataFrame, now: datetime) -> bool:
    if events.empty:
        return False
    soon = events[(events["type"] == "promo") & (events["when"] >= now) & (events["when"] <= now + timedelta(hours=PROMO_WINDOW_H))]
    return not soon.empty


# ---------------------------------------------------------------------------
# Puntuación
# ---------------------------------------------------------------------------


def classify(market_score: float) -> str:
    if market_score >= 85:
        return "Señal muy fuerte"
    if market_score >= 70:
        return "Interesante"
    if market_score >= 55:
        return "Watchlist"
    if market_score >= 40:
        return "Neutral"
    return "Riesgo elevado"


def decide_signal(market_score: float, risk_score: float, has_reason: bool = True) -> str | None:
    """`has_reason`: la carta tiene un motivo concreto para vigilarla (fuera de packs,
    caída sin causa, suelo de fodder…). Sin él, una puntuación media no basta para
    VIGILAR: así la lista no se llena de cartas que solo suben con el mercado."""
    if market_score >= 70 and risk_score <= 45:
        return "COMPRAR"
    if risk_score >= 70 or market_score < 40:
        return "RIESGO"
    if market_score >= 55 and has_reason:
        return "VIGILAR"
    return None


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def _age_days(created_at, now: datetime) -> float | None:
    if created_at is None or (isinstance(created_at, float) and pd.isna(created_at)):
        return None
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (now - dt).total_seconds() / 86400


def _sbc_substitutes(sbcs: pd.DataFrame) -> dict[str, tuple[int, int, str]]:
    """{nombre normalizado: (media, coste, nombre del SBC)} de los SBC que
    regalan una carta de jugador."""
    out: dict[str, tuple[int, int, str]] = {}
    if sbcs.empty:
        return out
    for r in sbcs.dropna(subset=["award_name", "cost"]).itertuples(index=False):
        key = _norm(r.award_name)
        if key not in out or r.cost < out[key][1]:
            out[key] = (int(r.award_overall), int(r.cost), r.name)
    return out


def _int_or_none(v) -> int | None:
    return None if v is None or pd.isna(v) else int(v)


def headline_alerts(alerts: list[Alert], followed: set[int]) -> list[Alert]:
    """Alertas que merecen un aviso destacado arriba de la página: sustitutos más
    baratos y movimientos fuertes de cartas que el usuario sigue. El resto se
    queda en la pestaña Alertas para no meter ruido."""
    return [a for a in alerts
            if a.category == "sustituto" or (a.ea_id is not None and a.ea_id in followed and a.level != "info")]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def score_movers(
    snap: MarketSnapshot,
    changes: pd.DataFrame | None,
    events: pd.DataFrame,
    now: datetime,
) -> pd.DataFrame:
    """Puntúa cada carta del listado de momentum."""
    movers = snap.movers.copy()
    if movers.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    if changes is not None and not changes.empty:
        # La variación de 24h ya la da FUT.GG; del historial propio se toman el resto de ventanas.
        movers = movers.merge(changes.drop(columns=["price_now", "pct_24h"], errors="ignore"), on="ea_id", how="left")
    for col in ("pct_1h", "pct_6h", "pct_72h", "pct_168h"):
        if col not in movers:
            movers[col] = float("nan")

    fodder = fodder_table(snap.cheapest)
    best_cpp = float(fodder["coins_per_point"].min()) if not fodder.empty else None
    substitutes = _sbc_substitutes(snap.sbcs)
    promo_soon = _promo_soon(events, now)
    group_trend = movers.groupby("rarity")["pct_24h"].median().to_dict()

    rows = []
    for r in movers.itertuples(index=False):
        p24 = float(r.pct_24h)
        p1 = None if pd.isna(r.pct_1h) else float(r.pct_1h)
        age = _age_days(r.created_at, now)
        is_base = r.rarity in BASE_RARITIES
        in_packs = is_base or (age is not None and age < PACK_CYCLE_DAYS)
        is_new = age is not None and age < NEW_CARD_DAYS
        fodder_value = ITEM_SCORE[r.overall] * best_cpp if best_cpp and r.overall in ITEM_SCORE else None
        fodder_supported = fodder_value is not None and r.price <= 1.3 * fodder_value
        sub = substitutes.get(_norm(r.name))
        has_substitute = (
            sub is not None and not (r.rarity or "").startswith("POTM")
            and sub[0] >= r.overall and sub[1] < SUBSTITUTE_DISCOUNT * r.price
        )
        reasons: list[str] = []
        risks: list[str] = []

        # Momentum (0-20)
        if is_new and p24 < 0:
            momentum = 5
            reasons.append("Carta recién salida: la caída es la normal mientras se abren packs.")
        elif p24 >= PARABOLIC_PCT:
            momentum = 8
        elif p24 >= 10:
            momentum = 13  # subida fuerte: ya no es "tendencia sana", el riesgo sube abajo
        elif p24 >= 2:
            momentum = 12 + (p24 - 2)
            reasons.append(f"Tendencia alcista moderada en 24h ({p24:+.1f}%).")
        elif p24 > -2:
            momentum = 10
        elif p24 > -12:
            momentum = 6 if in_packs else 9
            if not in_packs:
                reasons.append(f"Cae {p24:.1f}% en 24h estando fuera de packs: posible sobrerreacción.")
        else:
            momentum = 3 if in_packs else 6
        if p1 is not None and p1 <= -10:
            momentum -= 3
        momentum = _clamp(momentum, 0, 20)

        # Supply (0-15)
        if is_base:
            supply = 9
        elif is_new:
            supply = 4
        elif in_packs:
            supply = 7
        else:
            supply = 13
            reasons.append("Fuera de packs: no entra oferta nueva.")

        # Demand (0-15)
        demand = 8
        trend = group_trend.get(r.rarity, 0) or 0
        demand += 2 if trend > 1 else (-2 if trend < -1 else 0)
        if fodder_supported:
            demand += 5
            reasons.append("Su precio está cerca de su valor como fodder de SBC: suelo de precio.")
        demand = _clamp(demand, 0, 15)

        # Upcoming content (0-15)
        upcoming = 8
        if promo_soon and not is_base:
            upcoming = 4
            risks.append("Sale una promo nueva en menos de 48h: compite por las monedas.")

        # SBC/Evo utility (0-15)
        utility = round(_clamp(15 * fodder_value / r.price, 0, 15)) if fodder_value else 2
        if has_substitute:
            utility = 0
            # El sustituto es el riesgo más fuerte: va primero para que lo vea quien lea solo una línea.
            risks.insert(0, f"Sustituto: '{sub[2]}' da {r.name} {sub[0]} por ~{sub[1]:,} monedas.".replace(",", "."))

        # Risk score (0-100)
        risk = 25
        if p24 >= PARABOLIC_PCT:
            risk += 30
            risks.append(f"Subida vertical ({p24:+.1f}% en 24h) sin catalizador detectado.")
        elif p24 >= 10:
            risk += 15
        if has_substitute:
            risk += 45  # un sustituto casi idéntico y más barato basta por sí solo para RIESGO
        if r.price >= VERY_ILLIQUID_PRICE:
            risk += 25
            risks.append("Precio muy alto: pocos compradores, difícil de vender rápido.")
        elif r.price >= ILLIQUID_PRICE:
            risk += 15
            risks.append("Precio alto: liquidez limitada.")
        if is_new and in_packs:
            risk += 15
            risks.append("Sigue en packs: entra oferta nueva cada día.")
        if promo_soon and not is_base:
            risk += 15
        if fodder_supported:
            risk -= 15
        if p1 is not None and p1 <= -10:
            risk += 10
            risks.append(f"Cae {p1:.1f}% en la última hora.")
        risk = int(_clamp(risk, 0, 100))

        downside = round(10 - risk / 10)
        ms = int(round(momentum + supply + demand + upcoming + utility + NEWS_NEUTRAL + downside))
        # Motivo concreto para vigilar: oferta congelada, posible sobrerreacción o suelo de fodder.
        has_reason = (not in_packs) or fodder_supported or (p24 <= -2 and not is_new)
        signal = decide_signal(ms, risk, has_reason)
        horizon, horizon_h = ("1-3 días", 72) if not in_packs or is_base else ("3-7 días", 168)
        rows.append({
            "key": f"card:{r.ea_id}", "ea_id": r.ea_id, "name": r.name, "overall": r.overall, "rarity": r.rarity,
            "price": r.price, "pct_24h": p24, "pct_1h": p1, "pct_6h": r.pct_6h, "pct_72h": r.pct_72h,
            "pct_168h": r.pct_168h, "momentum": round(momentum), "supply": supply, "demand": round(demand),
            "upcoming": upcoming, "utility": utility, "news": NEWS_NEUTRAL, "downside": downside,
            "market_score": ms, "risk_score": risk, "label": classify(ms), "signal": signal,
            "horizon": horizon, "horizon_h": horizon_h, "reasons": reasons, "risks": risks, "url": r.url,
        })
    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)


def score_fodder(snap: MarketSnapshot, changes: pd.DataFrame | None, events: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Puntúa el rating de fodder más eficiente (menos monedas por punto)."""
    fodder = fodder_table(snap.cheapest)
    if fodder.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    best = fodder[fodder["is_best"]].iloc[0]
    floor = fodder_floor_price(snap.cheapest)
    at_floor = floor is not None and best["price"] <= floor * 1.08  # mediana pegada al suelo

    card = snap.cheapest[snap.cheapest["overall"] == best["overall"]].sort_values("price").iloc[0]
    p24 = None
    if changes is not None and not changes.empty:
        hit = changes[changes["ea_id"] == card["ea_id"]]
        if not hit.empty and not pd.isna(hit.iloc[0].get("pct_24h")):
            p24 = float(hit.iloc[0]["pct_24h"])

    active_points = 0
    if not snap.sbcs.empty:
        recent = snap.sbcs[(snap.sbcs["category"] == "players") & (~snap.sbcs["repeatable"].astype(bool))]
        active_points = int(recent["score_requirement"].fillna(0).sum())
    promo_soon = _promo_soon(events, now)

    momentum = 8 if p24 is None else _clamp(10 + p24, 4, 18)
    supply = 6
    demand = int(_clamp(6 + active_points / 100_000, 6, 15))
    upcoming = 12 if promo_soon else 9
    utility = 15
    downside = 9 if at_floor else 6
    risk = 20 if at_floor else 40
    ms = int(round(momentum + supply + demand + upcoming + utility + NEWS_NEUTRAL + downside))

    reasons = [
        f"El {int(best['overall'])} es el fodder más barato por punto: {best['coins_per_point']:.2f} monedas.".replace(".", ",", 1),
        f"Los SBC de jugador activos piden {active_points:,} puntos en total.".replace(",", "."),
    ]
    if at_floor:
        reasons.append(f"Está pegado al precio mínimo observado ({floor}): poco margen de caída.")
    if promo_soon:
        reasons.append("Sale una promo en <48h: suele traer SBC nuevos que consumen fodder.")
    risks = ["Impuesto del 5%: necesita subir >5,3% para ganar.",
             "Si sube, otros ratings pasan a ser más baratos por punto y frenan la subida."]

    return pd.DataFrame([{
        "key": f"fodder:{int(best['overall'])}", "ea_id": int(card["ea_id"]),
        "name": f"Fodder de {int(best['overall'])}", "overall": int(best["overall"]), "rarity": "Oro rara (fodder)",
        "price": int(best["price"]), "pct_24h": p24, "pct_1h": None, "pct_6h": None, "pct_72h": None, "pct_168h": None,
        "momentum": round(momentum), "supply": supply, "demand": demand, "upcoming": upcoming, "utility": utility,
        "news": NEWS_NEUTRAL, "downside": downside, "market_score": ms, "risk_score": risk, "label": classify(ms),
        "signal": decide_signal(ms, risk), "horizon": "1-7 días", "horizon_h": 168,
        "reasons": reasons, "risks": risks, "url": None,
    }], columns=SIGNAL_COLUMNS)


def build_signals(snap: MarketSnapshot, changes: pd.DataFrame | None, analyst: dict, now: datetime) -> pd.DataFrame:
    """Fodder + cartas del momentum, ordenadas por Market Score."""
    events = upcoming_events(analyst, now)
    frames = [f for f in (score_fodder(snap, changes, events, now), score_movers(snap, changes, events, now)) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values("market_score", ascending=False).reset_index(drop=True)


def top_by_signal(signals: pd.DataFrame, kind: str, n: int = 5) -> pd.DataFrame:
    sub = signals[signals["signal"] == kind]
    if kind == "RIESGO":
        return sub.sort_values(["risk_score", "market_score"], ascending=[False, True]).head(n)
    return sub.sort_values(["market_score", "risk_score"], ascending=[False, True]).head(n)


# ---------------------------------------------------------------------------
# Estado del mercado y alertas
# ---------------------------------------------------------------------------


def market_breadth(movers: pd.DataFrame) -> pd.DataFrame:
    """Por rareza: nº de cartas, mediana 24h, cuántas suben y bajan, y tono."""
    cols = ["rarity", "cards", "median_24h", "up", "down", "tone"]
    if movers.empty:
        return pd.DataFrame(columns=cols)
    g = movers.groupby("rarity")["pct_24h"]
    df = pd.DataFrame({
        "cards": g.size(), "median_24h": g.median().round(1),
        "up": g.apply(lambda s: int((s > 0).sum())), "down": g.apply(lambda s: int((s < 0).sum())),
    }).reset_index()
    share = df["up"] / (df["up"] + df["down"]).where(lambda s: s > 0, 1)
    df["tone"] = share.map(lambda x: "Alcista" if x >= 0.6 else ("Bajista" if x <= 0.4 else "Neutral"))
    return df.sort_values("cards", ascending=False)[cols].reset_index(drop=True)


def overall_tone(movers: pd.DataFrame) -> str:
    if movers.empty:
        return "Sin datos"
    up, down = int((movers["pct_24h"] > 0).sum()), int((movers["pct_24h"] < 0).sum())
    share = up / max(1, up + down)
    if share >= 0.6:
        return "Alcista"
    if share <= 0.4:
        return "Bajista"
    return "Neutral"


def build_alerts(
    snap: MarketSnapshot,
    signals: pd.DataFrame,
    changes: pd.DataFrame | None,
    analyst: dict,
    now: datetime,
) -> list[Alert]:
    alerts: list[Alert] = []
    if not snap.sbcs.empty:
        for r in snap.sbcs.itertuples(index=False):
            created = _age_days(r.created_at, now)
            ends = -_age_days(r.end_time, now) if r.end_time else None
            pts = f"{int(r.score_requirement):,} pts · " if pd.notna(r.score_requirement) else ""
            cost = f"~{int(r.cost):,} monedas" if pd.notna(r.cost) else ""
            detail = (pts + cost).replace(",", ".")
            if created is not None and created <= 1:
                alerts.append(Alert("aviso", "CONFIRMADO", f"Nuevo SBC: {r.name}", detail, "sbc"))
            if ends is not None and 0 <= ends <= 1:
                alerts.append(Alert("info", "CONFIRMADO", f"Expira en {ends * 24:.0f}h: {r.name}", detail, "sbc"))
    if changes is not None and not changes.empty and "pct_1h" in changes:
        names = dict(zip(snap.movers["ea_id"], snap.movers["name"])) if not snap.movers.empty else {}
        for r in changes.dropna(subset=["pct_1h"]).itertuples(index=False):
            label = names.get(r.ea_id, str(r.ea_id))
            if r.pct_1h <= -10:
                # En cartas caras con pocas ventas, una sola oferta mueve mucho el precio: solo aviso.
                level = "crítica" if r.price_now < ILLIQUID_PRICE else "aviso"
                alerts.append(Alert(level, "CONFIRMADO", f"{label} cae {r.pct_1h:.1f}% en 1h",
                                    f"Precio {int(r.price_now):,}".replace(",", "."), "movimiento_1h", int(r.ea_id)))
            elif r.pct_1h >= 15:
                alerts.append(Alert("aviso", "CONFIRMADO", f"{label} sube {r.pct_1h:+.1f}% en 1h",
                                    f"Precio {int(r.price_now):,}".replace(",", "."), "movimiento_1h", int(r.ea_id)))
    if not signals.empty:
        for r in signals[signals["pct_24h"].fillna(0) >= 50].itertuples(index=False):
            alerts.append(Alert("aviso", "CONFIRMADO", f"{r.name} {r.overall} sube {r.pct_24h:+.0f}% en 24h",
                                "Subida vertical: revisa antes de comprar.", "subida_24h", _int_or_none(r.ea_id)))
        for r in signals.itertuples(index=False):
            for risk in r.risks:
                if risk.startswith("Sustituto"):
                    alerts.append(Alert("crítica", "CONFIRMADO", f"Sustituto más barato para {r.name} {r.overall}", risk,
                                        "sustituto", _int_or_none(r.ea_id)))
    if not snap.movers.empty:
        fresh = snap.movers[snap.movers["created_at"].map(lambda c: (_age_days(c, now) or 99) <= 1)]
        if not fresh.empty:
            alerts.append(Alert("info", "CONFIRMADO", f"{len(fresh)} cartas nuevas en las últimas 24h", category="nuevas",
                                detail=
                                ", ".join(sorted(fresh["rarity"].dropna().unique()))))
    for e in upcoming_events(analyst, now, days=2).itertuples(index=False):
        if e.when >= now:
            alerts.append(Alert("aviso", e.status, f"Próximo: {e.label}", format_when(e.when), "evento"))
    order = {"crítica": 0, "aviso": 1, "info": 2}
    return sorted(alerts, key=lambda a: order.get(a.level, 3))


def trade_plan(row) -> dict[str, str]:
    """Zona de compra, objetivo e invalidación en monedas, a partir de la señal.

    Reglas fijas: el objetivo de compra exige +15% bruto (≈ +9% neto tras el
    5% de EA); la invalidación de una compra es −10% desde la entrada.
    """
    price = int(row["price"])
    fmt = lambda v: f"{int(round(v, -1 if v < 10_000 else -2)):,}".replace(",", ".")
    if row["signal"] == "COMPRAR":
        if str(row["key"]).startswith("fodder:"):
            return {"zona": f"{fmt(price)} (no pagar más de {fmt(price * 1.08)})",
                    "objetivo": f"{fmt(price * 1.15)}-{fmt(price * 1.25)}",
                    "invalidacion": "Sigue en el suelo 3 días seguidos sin SBC nuevos de alta puntuación."}
        return {"zona": f"{fmt(price * 0.97)}-{fmt(price)}", "objetivo": fmt(price * 1.15),
                "invalidacion": f"Cierra por debajo de {fmt(price * 0.9)}."}
    if row["signal"] == "VIGILAR":
        return {"zona": f"Comprar solo si baja a ~{fmt(price * 0.93)} y se estabiliza",
                "objetivo": fmt(price * 1.08), "invalidacion": f"Rompe {fmt(price * 0.85)}."}
    if row["signal"] == "RIESGO":
        return {"zona": "No comprar. Si la tienes, valorar vender.", "objetivo": f"Posible caída hacia {fmt(price * 0.85)}",
                "invalidacion": f"Se mantiene por encima de {fmt(price * 1.03)} durante 48h."}
    return {"zona": "—", "objetivo": "—", "invalidacion": "—"}
