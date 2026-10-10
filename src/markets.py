"""Mercados de un partido con su probabilidad, su cuota justa y lo que pasa en un partido medio.

- Goles: de la matriz de marcadores del modelo (reescalada al 1X2 final): más/menos de 0.5 a 4.5,
  ambos anotan, goles de cada equipo, doble oportunidad y ganar por 2 o más.
- Córners y tarjetas: de `src.corners_cards` (total, de cada equipo y quién tiene más).
- "Partido medio": la misma selección con las medias de la competición, la referencia para ver qué
  tiene de especial el partido.
- Cuota justa = 1 / probabilidad: si una casa paga más, el modelo ve valor. En goles el modelo queda
  cerca de las cuotas de cierre (ver src/backtest.py); en córners y tarjetas no hay cuotas pasadas para
  comprobarlo, solo los aciertos (`count_checks`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import corners_cards as cc
from src import match_model as mm

GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
TEAM_GOAL_LINES = (0.5, 1.5, 2.5)
HIGHLIGHT_MIN = 0.60
# Destacados: solo selecciones que en un partido medio de la competición están cerca de una moneda al
# aire (sin "más de 0.5 goles" ni "menos de 12.5 córners", que salen casi siempre y pagan casi nada).
HIGHLIGHT_BASE = (0.35, 0.65)
HIGHLIGHT_SKIP = {"Doble oportunidad"}  # es el 1X2 de otra forma (ya está en Partidos y Picks)
CONFIDENT = 0.65
FAMILIES = ("Goles", "Córners", "Tarjetas")


@dataclass(frozen=True)
class Selection:
    family: str  # "Goles", "Córners" o "Tarjetas"
    market: str  # p. ej. "Total de goles", "Ambos anotan", "Córners de Arsenal"
    label: str  # p. ej. "Más de 2.5", "Sí", "Arsenal"
    p: float
    base: float | None = None  # en un partido medio de la competición
    text: str = ""  # la selección en una frase ("Más de 2.5 goles", "Arsenal saca más córners")

    @property
    def fair_odds(self) -> float:
        return 1 / self.p if self.p > 0 else float("inf")


def _over_under(family: str, market: str, p_over: Callable[[float], float],
                base_over: Callable[[float], float] | None, lines: tuple[float, ...]) -> list[Selection]:
    unit = family.lower()
    out = []
    for line in lines:
        po, bo = p_over(line), (base_over(line) if base_over else None)
        out.append(Selection(family, market, f"Más de {line:g}", po, bo, f"Más de {line:g} {unit}"))
        out.append(Selection(family, market, f"Menos de {line:g}", 1 - po, None if bo is None else 1 - bo,
                             f"Menos de {line:g} {unit}"))
    return out


def matrix_over(matrix: np.ndarray, line: float, side: str = "total") -> float:
    """P(más de `line` goles) en el total ("total") o de un equipo ("home"/"away")."""
    i, j = np.indices(matrix.shape)
    count = {"total": i + j, "home": i, "away": j}[side]
    return float(matrix[count > line].sum())


def goal_selections(pred: mm.MatchPrediction) -> list[Selection]:
    matrix = pred.matrix if pred.matrix is not None else mm.reweight_matrix(
        mm.score_matrix(pred.lam_home, pred.lam_away, pred.trained.rho), pred.p_final)
    base = (mm.score_matrix(pred.base_home, pred.base_away, pred.trained.rho)
            if np.isfinite(pred.base_home) and np.isfinite(pred.base_away) else None)

    def base_of(fn: Callable[[np.ndarray], float]) -> float | None:
        return fn(base) if base is not None else None

    f = "Goles"
    out = _over_under(f, "Total de goles", lambda x: matrix_over(matrix, x),
                      (lambda x: matrix_over(base, x)) if base is not None else None, GOAL_LINES)
    btts, b_btts = float(matrix[1:, 1:].sum()), base_of(lambda m: float(m[1:, 1:].sum()))
    out += [Selection(f, "Ambos anotan", "Sí", btts, b_btts, "Ambos anotan"),
            Selection(f, "Ambos anotan", "No", 1 - btts, None if b_btts is None else 1 - b_btts,
                      "No anotan los dos")]
    for side, name in (("home", pred.home), ("away", pred.away)):
        for line in TEAM_GOAL_LINES:
            out.append(Selection(f, f"Goles de {name}", f"Más de {line:g}", matrix_over(matrix, line, side),
                                 base_of(lambda m, line=line, side=side: matrix_over(m, line, side)),
                                 f"{name} marca" if line == 0.5 else f"{name}: más de {line:g} goles"))
    p, pb = pred.p_final, (mm.outcome_probs(base) if base is not None else None)
    for code, text, idx in (("1X", f"{pred.home} o empate", (0, 1)), ("X2", f"Empate o {pred.away}", (1, 2)),
                            ("12", "No hay empate", (0, 2))):
        out.append(Selection(f, "Doble oportunidad", code, float(p[list(idx)].sum()),
                             float(pb[list(idx)].sum()) if pb is not None else None, f"{text} ({code})"))
    i, j = np.indices(matrix.shape)
    out.append(Selection(f, "Ganar por 2 o más", pred.home, float(matrix[i - j >= 2].sum()),
                         base_of(lambda m: float(m[i - j >= 2].sum())), f"{pred.home} gana por 2 o más"))
    out.append(Selection(f, "Ganar por 2 o más", pred.away, float(matrix[j - i >= 2].sum()),
                         base_of(lambda m: float(m[j - i >= 2].sum())), f"{pred.away} gana por 2 o más"))
    return out


def count_selections(fc: cc.CountForecast, home: str, away: str) -> list[Selection]:
    f = fc.label
    spec = cc.STATS[fc.stat]
    out = _over_under(f, f"Total de {f.lower()}", fc.p_over, lambda x: fc.p_over(x, league=True),
                      spec["total_lines"])
    unit = f.lower()
    for side, name in (("home", home), ("away", away)):
        for line in spec["team_lines"]:
            out.append(Selection(f, f"{f} de {name}", f"Más de {line:g}", fc.p_over(line, side),
                                 fc.p_over(line, side, league=True), f"{name}: más de {line:g} {unit}"))
    more, base = fc.more(), fc.more(league=True)
    corners = fc.stat == "corners"
    market = "Quién saca más córners" if corners else "Quién recibe más tarjetas"
    verb = "saca más córners" if corners else "recibe más tarjetas"
    texts = (f"{home} {verb}", f"Igual cantidad de {unit}", f"{away} {verb}")
    out += [Selection(f, market, label, float(more[k]), float(base[k]), texts[k])
            for k, label in ((0, home), (1, "Iguales"), (2, away))]
    return out


def highlight(selections: list[Selection], min_p: float = HIGHLIGHT_MIN,
              base_range: tuple[float, float] = HIGHLIGHT_BASE) -> Selection | None:
    """La selección más probable del partido entre las que en un partido medio están cerca del 50%
    (`base_range`), si llega a `min_p`: donde el modelo ve el partido más claro que lo normal."""
    lo, hi = base_range
    candidates = [s for s in selections if s.base is not None and lo <= s.base <= hi and s.p >= min_p
                  and s.market not in HIGHLIGHT_SKIP]
    return max(candidates, key=lambda s: s.p) if candidates else None


@dataclass(frozen=True)
class Check:
    """Aciertos de un mercado en partidos que el modelo no había visto."""
    market: str
    n: int
    hits: int  # el lado más probable salió
    confident_n: int  # partidos con un lado de CONFIDENT o más
    confident_hits: int
    confident_mean: float  # probabilidad media que daba el modelo en esos partidos


def _check(market: str, p_yes: np.ndarray, yes: np.ndarray) -> Check:
    p_yes, yes = np.asarray(p_yes, float), np.asarray(yes, bool)
    side_p = np.maximum(p_yes, 1 - p_yes)
    hit = np.where(p_yes >= 0.5, yes, ~yes)
    conf = side_p >= CONFIDENT
    return Check(market, len(yes), int(hit.sum()), int(conf.sum()), int(hit[conf].sum()),
                 float(side_p[conf].mean()) if conf.any() else float("nan"))


def goal_checks(val: pd.DataFrame, rho: float) -> list[Check]:
    """Más/menos de 2.5 y ambos anotan en las predicciones de validación del modelo de goles."""
    val = val.dropna(subset=["hg", "ag"])
    if val.empty:
        return []
    p_model = val[[f"p_model_{c}" for c in mm.CLASSES]].to_numpy(float)
    btts = np.array([float(mm.reweight_matrix(mm.score_matrix(lh, la, rho), p)[1:, 1:].sum())
                     for lh, la, p in zip(val["lam_h"], val["lam_a"], p_model)])
    return [_check("Más/menos de 2.5 goles", val["over25"].to_numpy(float), (val["hg"] + val["ag"]) > 2.5),
            _check("Ambos anotan", btts, (val["hg"] > 0) & (val["ag"] > 0))]


def count_checks(model: cc.CountModel) -> list[Check]:
    """Total en la línea principal y quién tiene más, de la validación de córners o tarjetas."""
    v = model.validation
    if v is None:
        return []
    unit = model.label.lower()
    more = "Quién saca más córners" if model.stat == "corners" else "Quién recibe más tarjetas"
    return [Check(f"Más/menos de {v.line:g} {unit}", v.n, v.hits, v.confident_n, v.confident_hits, v.confident_mean),
            Check(more, v.more_n, v.more_hits, v.more_confident_n, v.more_confident_hits, float("nan"))]


def _market_key(market: str) -> str:
    """"Más/menos de 9.5 córners" y "Más/menos de 10.5 córners" son el mismo mercado en ligas distintas."""
    return "Más/menos · " + market.split()[-1] if market.startswith("Más/menos de ") else market


def merge_checks(groups: list[list[Check]]) -> list[Check]:
    """Suma los aciertos del mismo mercado de varias competiciones (p. ej. las del día)."""
    by_key: dict[str, list[Check]] = {}
    for checks in groups:
        for c in checks:
            by_key.setdefault(_market_key(c.market), []).append(c)
    out = []
    for key, cs in by_key.items():
        market = cs[0].market if len({c.market for c in cs}) == 1 else \
            f"Más/menos de la línea de cada liga ({key.split(' · ')[1]})"
        means = [(c.confident_mean, c.confident_n) for c in cs if c.confident_n and np.isfinite(c.confident_mean)]
        mean = sum(m * n for m, n in means) / sum(n for _, n in means) if means else float("nan")
        out.append(Check(market, sum(c.n for c in cs), sum(c.hits for c in cs), sum(c.confident_n for c in cs),
                         sum(c.confident_hits for c in cs), mean))
    return out
