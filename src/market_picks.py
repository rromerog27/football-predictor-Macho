"""Picks de goles, córners y tarjetas por niveles, su prueba con partidos ya jugados y los parlays del día.

Regla (`TIERS`): una selección de `src.markets` es pick si el modelo le da al menos el mínimo del nivel y
al menos `edge` puntos más que en un partido medio de la competición (así no entran "más de 0.5 goles" ni
"menos de 12.5 córners", que salen casi siempre). De cada partido, familia y nivel queda una sola: la que
más se separa de lo normal.

Prueba histórica: la misma regla sobre los partidos de validación (los que el modelo no usó para
ajustarse, cada uno predicho solo con los anteriores): goles desde `TrainedModels.val_predictions`,
córners y tarjetas desde `corners_cards.validation_forecasts`.

Parlays: una pata por partido, del nivel que corresponde (fija → fijos, intermedia → intermedios,
arriesgada → arriesgados), las más probables primero; la probabilidad del parlay es el producto (los
partidos son independientes). `simulate_parlays` arma el parlay de cada día de la validación con la
misma regla y cuenta cuántos salieron.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import corners_cards as cc
from src import markets
from src import match_model as mm


@dataclass(frozen=True)
class Tier:
    name: str
    lo: float  # probabilidad mínima
    hi: float  # probabilidad máxima (excluida)
    edge: float  # puntos mínimos sobre un partido medio de la competición


TIERS = (Tier("Fijo", 0.90, 1.01, 0.10), Tier("Intermedio", 0.80, 0.90, 0.10), Tier("Arriesgado", 0.65, 0.80, 0.15))
TIER_NAMES = tuple(t.name for t in TIERS)
MARKET_GAP = 0.15  # partido en el que el favorito del modelo se separa de la casa en más de esto: sin picks


@dataclass(frozen=True)
class MarketPick:
    key: str  # partido
    kickoff: pd.Timestamp  # UTC
    competition: str
    home: str
    away: str
    tier: str
    sel: markets.Selection
    hit: bool | None = None  # None: todavía no se jugó

    @property
    def edge(self) -> float:
        return self.sel.p - (self.sel.base if self.sel.base is not None else self.sel.p)


def tier_of(sel: markets.Selection) -> Tier | None:
    if sel.base is None:
        return None
    for t in TIERS:
        if t.lo <= sel.p < t.hi and sel.p - sel.base >= t.edge:
            return t
    return None


def match_picks(selections: list[markets.Selection]) -> list[tuple[str, markets.Selection]]:
    """(nivel, selección) de un partido: una por familia y nivel, la que más se separa de lo normal."""
    best: dict[tuple[str, str], markets.Selection] = {}
    for s in selections:
        t = tier_of(s)
        if t is None:
            continue
        k = (s.family, t.name)
        if k not in best or s.p - s.base > best[k].p - best[k].base:
            best[k] = s
    return [(tier, s) for (_, tier), s in best.items()]


def market_disagrees(p_model: np.ndarray, p_market: np.ndarray | None, gap: float = MARKET_GAP) -> bool:
    """El favorito del modelo y el de la casa no coinciden, o el modelo se separa más de `gap` en él."""
    if p_market is None:
        return False
    k = int(np.argmax(p_model))
    return int(np.argmax(p_market)) != k or abs(float(p_model[k]) - float(p_market[k])) > gap


def selection_hit(sel: markets.Selection, hv: float, av: float, home: str, away: str) -> bool:
    """Si la selección salió con `hv`-`av` (goles, córners o tarjetas del local y del visitante)."""
    m, label = sel.market, sel.label

    def over_under(x: float) -> bool:
        line = float(label.split()[-1])
        return x > line if label.startswith("Más") else x < line

    if m.startswith("Total de"):
        return over_under(hv + av)
    if m == f"{sel.family} de {home}":
        return over_under(hv)
    if m == f"{sel.family} de {away}":
        return over_under(av)
    if m == "Ambos anotan":
        return (hv > 0 and av > 0) == (label == "Sí")
    if m == "Doble oportunidad":
        r = 0 if hv > av else (1 if hv == av else 2)
        return {"1X": r in (0, 1), "X2": r in (1, 2), "12": r in (0, 2)}[label]
    if m == "Ganar por 2 o más":
        return hv - av >= 2 if label == home else av - hv >= 2
    if m.startswith("Quién"):
        return {home: hv > av, "Iguales": hv == av, away: av > hv}[label]
    raise ValueError(f"Mercado desconocido: {m}")


def _market_1x2_ou(r) -> tuple[np.ndarray | None, float | None]:
    """1X2 y más de 2.5 sin margen de las cuotas de una fila (None si no hay)."""
    odds = np.array([getattr(r, c, np.nan) for c in ("odds_h", "odds_d", "odds_a")], float)
    if not np.isfinite(odds).all() or (odds <= 1).any():
        return None, None
    inv = 1 / odds
    ou = np.array([getattr(r, c, np.nan) for c in ("odds_over25", "odds_under25")], float)
    p_over = float((1 / ou[0]) / (1 / ou).sum()) if np.isfinite(ou).all() and (ou > 1).all() else None
    return inv / inv.sum(), p_over


def goal_history(val: pd.DataFrame, rho: float, odds: pd.DataFrame | None = None) -> list[MarketPick]:
    """Picks de goles que la regla habría dado en los partidos de validación, con su resultado. El partido
    medio usa los goles esperados medios de la propia validación. Con `odds` (columnas id, odds_h/d/a y, si hay,
    odds_over25/odds_under25), en los partidos con cuotas cada selección vale lo mínimo entre el modelo y la casa
    (1X2 de la casa y goles esperados de sus cuotas), como en la app (`markets.consensus`)."""
    val = val.dropna(subset=["hg", "ag"])
    if val.empty:
        return []
    cols = ["odds_h", "odds_d", "odds_a", "odds_over25", "odds_under25"]
    val = val.drop(columns=[c for c in cols if c in val])
    if odds is not None and len(odds):
        val = val.merge(odds[["id", *[c for c in cols if c in odds]]].drop_duplicates("id"), on="id", how="left")
    base_h, base_a = float(val["lam_h"].mean()), float(val["lam_a"].mean())

    def selections(lam_h: float, lam_a: float, p: np.ndarray, r) -> list[markets.Selection]:
        return markets.goal_selections(SimpleNamespace(
            home=r.home, away=r.away, lam_home=lam_h, lam_away=lam_a, p_final=p,
            matrix=mm.reweight_matrix(mm.score_matrix(lam_h, lam_a, rho), p), base_home=base_h, base_away=base_a,
            trained=SimpleNamespace(rho=rho)))

    out = []
    for r in val.itertuples(index=False):
        sels = selections(r.lam_h, r.lam_a, np.array([r.p_model_H, r.p_model_D, r.p_model_A], float), r)
        pm, p_over = _market_1x2_ou(r) if mm.MARKET_ANCHOR else (None, None)
        if pm is not None:
            sels = markets.consensus(selections(*mm.market_lambdas(pm, p_over, rho), pm, r), sels)
        for tier, sel in match_picks(sels):
            out.append(MarketPick(str(r.id), r.datetime, r.competition, r.home, r.away, tier, sel,
                                  selection_hit(sel, r.hg, r.ag, r.home, r.away)))
    return out


def count_history(model: cc.CountModel) -> list[MarketPick]:
    """Picks de córners o tarjetas que la regla habría dado en los partidos de validación del modelo."""
    out = []
    for r, fc in cc.validation_forecasts(model):
        for tier, sel in match_picks(markets.count_selections(fc, r["home"], r["away"])):
            out.append(MarketPick(str(r["id"]), r["datetime"], model.competition, r["home"], r["away"], tier, sel,
                                  selection_hit(sel, r["hv"], r["av"], r["home"], r["away"])))
    return out


def summary(picks: list[MarketPick]) -> pd.DataFrame:
    """Aciertos por nivel y familia: columnas tier, family, n, hits, mean_p."""
    rows = [(p.tier, p.sel.family, bool(p.hit), p.sel.p) for p in picks if p.hit is not None]
    if not rows:
        return pd.DataFrame(columns=["tier", "family", "n", "hits", "mean_p"])
    df = pd.DataFrame(rows, columns=["tier", "family", "hit", "p"])
    out = df.groupby(["tier", "family"]).agg(n=("hit", "size"), hits=("hit", "sum"), mean_p=("p", "mean")).reset_index()
    out["tier"] = pd.Categorical(out["tier"], TIER_NAMES, ordered=True)
    out["family"] = pd.Categorical(out["family"], markets.FAMILIES, ordered=True)
    return out.sort_values(["tier", "family"]).reset_index(drop=True)


@dataclass
class Parlay:
    name: str  # "Fija", "Intermedia" o "Arriesgada"
    legs: list[MarketPick] = field(default_factory=list)

    @property
    def p(self) -> float:
        return float(np.prod([leg.sel.p for leg in self.legs])) if self.legs else 0.0

    @property
    def fair_odds(self) -> float:
        return 1 / self.p if self.p > 0 else float("inf")

    @property
    def hit(self) -> bool | None:
        if any(leg.hit is None for leg in self.legs):
            return None
        return all(leg.hit for leg in self.legs)


PARLAYS = (("Fija", "Fijo", 4), ("Intermedia", "Intermedio", 5), ("Arriesgada", "Arriesgado", 5))
MIN_LEGS = 2


def build_parlays(picks: list[MarketPick]) -> dict[str, Parlay]:
    """Parlay de cada nivel: una pata por partido (la más probable del nivel), hasta el máximo de patas;
    sin parlay si no llega a MIN_LEGS."""
    out = {}
    for name, tier, max_legs in PARLAYS:
        legs, used = [], set()
        for p in sorted((p for p in picks if p.tier == tier), key=lambda p: (-p.sel.p, p.kickoff)):
            if p.key in used:
                continue
            legs.append(p)
            used.add(p.key)
            if len(legs) == max_legs:
                break
        if len(legs) >= MIN_LEGS:
            out[name] = Parlay(name, sorted(legs, key=lambda p: p.kickoff))
    return out


def simulate_parlays(picks: list[MarketPick]) -> pd.DataFrame:
    """Parlays que la regla habría armado cada día de la validación y si salieron. Columnas: day, name,
    legs, p, hit."""
    rows = []
    by_day: dict = {}
    for p in picks:
        if p.hit is not None:
            by_day.setdefault(pd.Timestamp(p.kickoff).normalize(), []).append(p)
    for day, day_picks in sorted(by_day.items()):
        for name, parlay in build_parlays(day_picks).items():
            rows.append({"day": day, "name": name, "legs": len(parlay.legs), "p": parlay.p, "hit": parlay.hit})
    return pd.DataFrame(rows, columns=["day", "name", "legs", "p", "hit"])
