"""Córners y tarjetas: cuántos se esperan en un partido y la probabilidad de sus mercados.

Modelo por equipo, aparte del de goles (usa las columnas `STAT_COLUMNS` de los partidos, de ESPN):

- Cada equipo tiene una tasa a favor y otra en contra, relativas a la media de la competición en la
  que jugó cada partido (la del local y la del visitante por separado, así un partido de copa o de
  otra liga del pool cuenta en su escala). Las tasas se suavizan exponencialmente (vida media de
  `HALF_LIFE` partidos) y se encogen hacia la media con `PRIOR_MATCHES` partidos "de media".
- Esperado del local = media del local en la competición × su tasa a favor × la tasa en contra del
  visitante (y al revés para el visitante).
- Lo esperado se acerca a la media de la competición en la proporción que mejor predijo los partidos
  de entrenamiento (`BETA_GRID`; por separado para lo de cada equipo y para el total del partido): las
  tasas por equipo exageran las diferencias en los totales.
- Las cantidades siguen una binomial negativa (algo más dispersa que Poisson); la dispersión se ajusta
  con las predicciones hechas antes de cada partido.
- Validación como la del modelo de goles: el `VAL_FRACTION` más reciente de los partidos de la
  competición, cada uno predicho solo con los anteriores, contra la media de la liga como referencia.

En 6.065 partidos de 18 ligas europeas (2025/26, football-data) este método predice bien quién saca
más córners (64% de aciertos; 77% en el cuarto de partidos más desparejos) y aporta poco sobre la media
de la liga en los totales de córners y tarjetas (el árbitro, que pesa en las tarjetas, no está en los
datos de casi ninguna liga).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.match_model import LeagueData

HALF_LIFE = 40.0  # partidos
PRIOR_MATCHES = 20.0
LEAGUE_ALPHA = 0.01  # suavizado de la media de cada competición (por partido)
LEAGUE_INIT_MATCHES = 100  # partidos con los que arranca la media de cada competición
MIN_TEAM_WEIGHT = 3.0  # partidos (ponderados) de un equipo para evaluar o fiarse de su predicción
MIN_FOCUS_MATCHES = 30  # partidos de la competición con el dato para ajustar el modelo
MIN_COVERAGE = 0.5  # partidos jugados de una competición con el dato (si no, se descarta la competición)
MIN_CARD_MATCHES = 0.7  # partidos con al menos una tarjeta (menos: la liga no publica las tarjetas)
VAL_FRACTION = 0.3
CONFIDENT = 0.65
MAX_COUNT = 40
K_GRID = np.geomspace(2.0, 1000.0, 50)
BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)  # peso de lo propio de los equipos frente a la media de la liga

STATS = {
    "corners": {"cols": ("h_corners", "a_corners"), "label": "Córners",
                "total_lines": (7.5, 8.5, 9.5, 10.5, 11.5, 12.5), "team_lines": (3.5, 4.5, 5.5)},
    "cards": {"cols": ("h_cards", "a_cards"), "label": "Tarjetas",
              "total_lines": (2.5, 3.5, 4.5, 5.5, 6.5), "team_lines": (0.5, 1.5, 2.5)},
}


@dataclass(frozen=True)
class Validation:
    """Cómo le fue al modelo con partidos que no había visto (los más recientes de la competición)."""
    n: int
    line: float  # línea principal del total (la media de la liga redondeada a x.5)
    over_rate: float  # cuántas veces salió el "más de" de esa línea
    ll_model: float  # log loss del "más / menos" de esa línea
    ll_league: float  # lo mismo prediciendo con la media de la liga
    hits: int  # aciertos del lado más probable de la línea
    confident_n: int  # partidos con un lado de CONFIDENT o más
    confident_hits: int
    confident_mean: float  # probabilidad media que daba el modelo en esos partidos
    more_n: int  # "quién saca más": partidos evaluados
    more_hits: int  # el equipo al que el modelo le daba más salió con más (los empates cuentan como fallo)
    more_confident_n: int  # partidos con un equipo de CONFIDENT o más en "quién saca más"
    more_confident_hits: int


@dataclass
class CountModel:
    stat: str  # "corners" o "cards"
    competition: str
    team_state: dict[str, list[float]] = field(repr=False)  # equipo -> [a favor, en contra, peso]
    league: dict[str, tuple[float, float]]  # competición -> (media del local, media del visitante)
    k_team: float  # dispersión de lo de cada equipo
    k_total: float  # dispersión del total del partido
    beta_team: float  # peso de las tasas de los equipos en lo de cada equipo (0 = media de la liga)
    beta_total: float  # lo mismo en el total del partido
    pre: dict[str, tuple[float, float, float, float]] = field(repr=False)  # id -> (esp. local, esp. visit., pesos)
    validation: Validation | None

    @property
    def label(self) -> str:
        return STATS[self.stat]["label"]


def _alpha(half_life: float) -> float:
    return 1 - 0.5 ** (1 / half_life)


def nb_pmf(mu: float, k: float, n: int = MAX_COUNT) -> np.ndarray:
    """Probabilidad de 0..n con media `mu` y dispersión `k` (varianza mu + mu²/k); suma 1."""
    x = np.arange(n + 1)
    p = stats.nbinom.pmf(x, k, k / (k + max(mu, 1e-6)))
    return p / p.sum()


def _nb_loglik(y: np.ndarray, mu: np.ndarray, k: float) -> float:
    return float(stats.nbinom.logpmf(y, k, k / (k + np.maximum(mu, 1e-6))).mean())


def fit_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """k de la binomial negativa que mejor explica `y` con medias `mu` (grilla; casi Poisson arriba)."""
    if len(y) < 30:
        return float(K_GRID[-1])
    ll = [_nb_loglik(y, mu, k) for k in K_GRID]
    return float(K_GRID[int(np.argmax(ll))])


def shrink(mu, base, beta: float):
    """Lo esperado acercado a la media de la liga: base + beta · (mu − base)."""
    return base + beta * (np.asarray(mu, float) - base)


def fit_shrink(y: np.ndarray, mu: np.ndarray, base: np.ndarray) -> tuple[float, float]:
    """(beta, k) que mejor explican `y` (ver BETA_GRID y K_GRID)."""
    best = (-np.inf, 1.0, float(K_GRID[-1]))
    for beta in BETA_GRID:
        m = shrink(mu, base, beta)
        k = fit_dispersion(y, m)
        ll = _nb_loglik(y, m, k)
        if ll > best[0]:
            best = (ll, beta, k)
    return best[1], best[2]


def _p_over(mu: np.ndarray, k: float, line: float) -> np.ndarray:
    return 1 - stats.nbinom.cdf(np.floor(line), k, k / (k + np.maximum(mu, 1e-6)))


def usable_rows(matches: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Partidos jugados en 90 minutos con el dato de los dos equipos, sin las competiciones que casi
    no lo publican (o que publican las incidencias sin tarjetas)."""
    hc, ac = STATS[stat]["cols"]
    if hc not in matches:
        return matches.iloc[0:0]
    played = matches[matches["played"] & ~matches["extra_time"].astype(bool)]
    has = played[hc].notna() & played[ac].notna()
    keep = []
    for comp, g in played.groupby("competition"):
        with_stat = has[g.index]
        if not with_stat.any():
            continue
        since = g.loc[with_stat, "datetime"].min()  # desde que la fuente publica el dato
        coverage = with_stat[g["datetime"] >= since].mean()
        rows = g[with_stat]
        if stat == "cards" and ((rows[hc] + rows[ac]) > 0).mean() < MIN_CARD_MATCHES:
            continue
        if coverage >= MIN_COVERAGE:
            keep.append(comp)
    out = played[has & played["competition"].isin(keep)]
    return out.sort_values(["datetime", "id"])


def _rates(state: list[float] | None) -> tuple[float, float, float]:
    """(a favor, en contra, peso) relativos a la media (1.0 = media), encogidos hacia 1."""
    s = state or [0.0, 0.0, 0.0]
    return (s[0] + PRIOR_MATCHES) / (s[2] + PRIOR_MATCHES), (s[1] + PRIOR_MATCHES) / (s[2] + PRIOR_MATCHES), s[2]


def _expected(state: dict, league: tuple[float, float], home: str, away: str, neutral: bool = False
              ) -> tuple[float, float, float, float]:
    lh, la = league
    if neutral:
        lh = la = (lh + la) / 2
    att_h, def_h, wh = _rates(state.get(home))
    att_a, def_a, wa = _rates(state.get(away))
    return lh * att_h * def_a, la * att_a * def_h, wh, wa


def _sequential(rows: pd.DataFrame, stat: str) -> tuple[pd.DataFrame, dict, dict]:
    """Recorre los partidos en orden: predicción previa de cada uno y estado final (equipos y ligas)."""
    hc, ac = STATS[stat]["cols"]
    alpha = _alpha(HALF_LIFE)
    league: dict[str, list[float]] = {}
    for comp, g in rows.groupby("competition"):
        first = g.head(LEAGUE_INIT_MATCHES)
        league[comp] = [float(first[hc].mean()), float(first[ac].mean())]
    state: dict[str, list[float]] = {}
    out = []
    for r in rows[["id", "competition", "home", "away", "neutral", hc, ac]].itertuples(index=False):
        mid, comp, home, away, neutral, hv, av = r
        lg = league[comp]
        mh, ma, wh, wa = _expected(state, (lg[0], lg[1]), home, away, bool(neutral))
        out.append((mid, comp, mh, ma, wh, wa, lg[0], lg[1], hv, av))
        # Lo de cada equipo relativo a la media de su condición en esta competición.
        for team, f, a, mf, ma_ in ((home, hv, av, lg[0], lg[1]), (away, av, hv, lg[1], lg[0])):
            s = state.setdefault(team, [0.0, 0.0, 0.0])
            s[0] = (1 - alpha) * s[0] + f / max(mf, 1e-6)
            s[1] = (1 - alpha) * s[1] + a / max(ma_, 1e-6)
            s[2] = (1 - alpha) * s[2] + 1.0
        lg[0] += LEAGUE_ALPHA * (hv - lg[0])
        lg[1] += LEAGUE_ALPHA * (av - lg[1])
    pre = pd.DataFrame(out, columns=["id", "competition", "mh", "ma", "wh", "wa", "lh", "la", "hv", "av"])
    return pre, state, {c: (v[0], v[1]) for c, v in league.items()}


def more_probs(home_pmf: np.ndarray, away_pmf: np.ndarray) -> np.ndarray:
    """[local más, iguales, visitante más] con las dos cantidades independientes."""
    joint = np.outer(home_pmf, away_pmf)
    return np.array([np.tril(joint, -1).sum(), np.trace(joint), np.triu(joint, 1).sum()])


def _more_probs_rows(mh: np.ndarray, ma: np.ndarray, k: float) -> np.ndarray:
    return np.array([more_probs(nb_pmf(h, k), nb_pmf(a, k)) for h, a in zip(mh, ma)])


def _validate(pre: pd.DataFrame, focus: str, cutoff_idx: int, fitted: dict, line: float) -> Validation | None:
    val = pre.iloc[cutoff_idx:]
    val = val[(val["competition"] == focus) & (val["wh"] >= MIN_TEAM_WEIGHT) & (val["wa"] >= MIN_TEAM_WEIGHT)]
    if len(val) < 20:
        return None
    y = (val["hv"] + val["av"]).to_numpy(float)
    over = (y > line).astype(float)
    league_total = (val["lh"] + val["la"]).to_numpy(float)
    mu_total = shrink((val["mh"] + val["ma"]).to_numpy(float), league_total, fitted["beta_total"])
    p = np.clip(_p_over(mu_total, fitted["k_total"], line), 1e-6, 1 - 1e-6)
    pl = np.clip(_p_over(league_total, fitted["k_league"], line), 1e-6, 1 - 1e-6)

    def ll(q):
        return float(-np.mean(over * np.log(q) + (1 - over) * np.log(1 - q)))

    side_hit = np.where(p >= 0.5, over, 1 - over)
    conf = np.maximum(p, 1 - p) >= CONFIDENT
    more = _more_probs_rows(shrink(val["mh"], val["lh"].to_numpy(float), fitted["beta_team"]),
                            shrink(val["ma"], val["la"].to_numpy(float), fitted["beta_team"]), fitted["k_team"])
    pick = np.where(more[:, 0] >= more[:, 2], 0, 2)
    actual = np.where(val["hv"] > val["av"], 0, np.where(val["hv"] == val["av"], 1, 2))
    more_hit = pick == actual
    more_conf = more[np.arange(len(pick)), pick] >= CONFIDENT
    return Validation(
        n=len(val), line=line, over_rate=float(over.mean()), ll_model=ll(p), ll_league=ll(pl),
        hits=int(side_hit.sum()), confident_n=int(conf.sum()), confident_hits=int(side_hit[conf].sum()),
        confident_mean=float(np.maximum(p, 1 - p)[conf].mean()) if conf.any() else float("nan"),
        more_n=len(val), more_hits=int(more_hit.sum()),
        more_confident_n=int(more_conf.sum()), more_confident_hits=int(more_hit[more_conf].sum()),
    )


def main_line(mean_total: float) -> float:
    """Línea x.5 más cercana a la media (córners 9.74 → 9.5; tarjetas 4.34 → 4.5)."""
    return float(np.floor(mean_total) + 0.5)


def fit(data: LeagueData, stat: str) -> CountModel | None:
    """Ajusta el modelo de `stat` ("corners" o "cards") con los partidos de la competición (y su pool).
    None si la competición no tiene el dato (p. ej. ESPN no lo publica)."""
    rows = usable_rows(data.matches, stat)
    focus = rows[rows["competition"] == data.competition]
    if len(focus) < MIN_FOCUS_MATCHES:
        return None
    pre, state, league = _sequential(rows, stat)
    reliable = (pre["wh"] >= MIN_TEAM_WEIGHT) & (pre["wa"] >= MIN_TEAM_WEIGHT)
    focus_pos = np.flatnonzero(pre["competition"].to_numpy() == data.competition)
    cutoff_idx = int(focus_pos[int(len(focus_pos) * (1 - VAL_FRACTION))])
    train = pre.iloc[:cutoff_idx][reliable.iloc[:cutoff_idx]]

    def fit_params(df: pd.DataFrame) -> dict:
        total = (df["hv"] + df["av"]).to_numpy(float)
        league_total = (df["lh"] + df["la"]).to_numpy(float)
        beta_team, k_team = fit_shrink(np.concatenate([df["hv"], df["av"]]).astype(float),
                                       np.concatenate([df["mh"], df["ma"]]).astype(float),
                                       np.concatenate([df["lh"], df["la"]]).astype(float))
        beta_total, k_total = fit_shrink(total, (df["mh"] + df["ma"]).to_numpy(float), league_total)
        return {"beta_team": beta_team, "k_team": k_team, "beta_total": beta_total, "k_total": k_total,
                "k_league": fit_dispersion(total, league_total)}

    validation = None
    if len(train) >= 100:
        line = main_line(float((train["hv"] + train["av"]).mean()))
        validation = _validate(pre, data.competition, cutoff_idx, fit_params(train), line)
    final = fit_params(pre[reliable]) if reliable.sum() >= 100 else {
        "beta_team": 0.5, "k_team": float(K_GRID[-1]), "beta_total": 0.5, "k_total": float(K_GRID[-1])}
    pre_map = {r.id: (r.mh, r.ma, r.wh, r.wa) for r in pre.itertuples(index=False)}
    return CountModel(stat, data.competition, state, league, final["k_team"], final["k_total"],
                      final["beta_team"], final["beta_total"], pre_map, validation)


def fit_all(data: LeagueData) -> dict[str, CountModel]:
    """Modelos de córners y tarjetas de la competición (solo los que tienen datos)."""
    return {stat: model for stat in STATS if (model := fit(data, stat)) is not None}


@dataclass(frozen=True)
class CountForecast:
    """Lo que se espera de un partido para una estadística (córners o tarjetas)."""
    stat: str
    mu_home: float
    mu_away: float
    league_home: float  # media de la competición (referencia: un partido "normal")
    league_away: float
    k_team: float
    k_total: float
    low_data: bool  # algún equipo con menos de MIN_TEAM_WEIGHT partidos con el dato
    mu_total_own: float = float("nan")  # total esperado (con su propio encogimiento hacia la liga)

    @property
    def label(self) -> str:
        return STATS[self.stat]["label"]

    @property
    def mu_total(self) -> float:
        return self.mu_total_own if np.isfinite(self.mu_total_own) else self.mu_home + self.mu_away

    def p_over(self, line: float, side: str = "total", league: bool = False) -> float:
        """P(más de `line`) del total o de un equipo ("home"/"away"); con `league`, la de un partido medio."""
        if side == "total":
            mu = (self.league_home + self.league_away) if league else self.mu_total
            return float(_p_over(np.array([mu]), self.k_total, line)[0])
        mu = (self.league_home if side == "home" else self.league_away) if league else (
            self.mu_home if side == "home" else self.mu_away)
        return float(_p_over(np.array([mu]), self.k_team, line)[0])

    def more(self, league: bool = False) -> np.ndarray:
        """[local más, iguales, visitante más]."""
        h, a = (self.league_home, self.league_away) if league else (self.mu_home, self.mu_away)
        return more_probs(nb_pmf(h, self.k_team), nb_pmf(a, self.k_team))


def forecast(model: CountModel, home: str, away: str, competition: str | None = None,
             match_id: str | None = None, neutral: bool = False) -> CountForecast:
    """Esperado de `home`-`away`. Si el partido ya se jugó y está en los datos (`match_id`), la
    predicción que se hacía antes de jugarlo (sin mirar su resultado)."""
    comp = competition or model.competition
    league = model.league.get(comp) or model.league.get(model.competition) or (
        float(np.mean([v[0] for v in model.league.values()])), float(np.mean([v[1] for v in model.league.values()])))
    if match_id is not None and match_id in model.pre:
        mh, ma, wh, wa = model.pre[match_id]
    else:
        mh, ma, wh, wa = _expected(model.team_state, league, home, away, neutral)
    if neutral:
        league = ((league[0] + league[1]) / 2,) * 2
    total = float(shrink(mh + ma, league[0] + league[1], model.beta_total))
    mh, ma = (float(shrink(mh, league[0], model.beta_team)), float(shrink(ma, league[1], model.beta_team)))
    return CountForecast(model.stat, mh, ma, float(league[0]), float(league[1]), model.k_team, model.k_total,
                         min(wh, wa) < MIN_TEAM_WEIGHT, total)
