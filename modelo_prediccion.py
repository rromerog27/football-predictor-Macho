#!/usr/bin/env python3
"""Predicción de un partido con datos reales de Understat (xG): Poisson + regresión logística.

Script de terminal sobre `src/understat_model.py` (el mismo modelo que usa la
página "Partidos del día" de la app). Ver la descripción del modelo en ese
módulo y en el README.

Uso:
    python3 modelo_prediccion.py "Arsenal" "Leeds" --liga "Premier League"
    python3 modelo_prediccion.py "Arsenal" "Leeds" --detalle
"""

from __future__ import annotations

import argparse
import sys

from src import understat_model as um

SEP = "=" * 50


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def team_metrics_lines(name: str, role: str, snap: dict) -> list[str]:
    recent = snap["recent_rows"]
    venue_txt = "en casa" if role == "local" else "fuera"
    first, last = recent["datetime"].iloc[0], recent["datetime"].iloc[-1]
    seasons = " + ".join(um.season_label(s) for s in sorted(recent["season"].unique()))
    return [
        f"    {name} ({role}) — últimos {len(recent)} partidos de liga "
        f"[{first:%Y-%m-%d} → {last:%Y-%m-%d}, temporadas {seasons}]",
        f"      xG a favor {recent['xgf'].mean():.2f} | xG en contra {recent['xga'].mean():.2f} | "
        f"Goles a favor {recent['gf'].mean():.2f} | Goles en contra {recent['ga'].mean():.2f} (por partido)",
        f"      Forma últ. {len(snap['form_rows'])}: {um.form_string(snap['form_rows'])} "
        f"= {int(snap['form_rows']['pts'].sum())} pts | Últ. {len(snap['venue_rows'])} {venue_txt}: "
        f"{um.form_string(snap['venue_rows'])} = {int(snap['venue_rows']['pts'].sum())} pts | "
        f"Descanso: {snap['rest_days']:.0f} días ({um.rest_category(snap['rest_days'])})",
    ]


def detail_lines(name: str, snap: dict) -> list[str]:
    lines = [f"    {name}: fecha | sede | rival | resultado | xG"]
    for r in snap["recent_rows"].itertuples(index=False):
        lines.append(
            f"      {r.datetime:%Y-%m-%d} | {'L' if r.venue == 'h' else 'V'} | {r.opp:<24} | "
            f"{int(r.gf)}-{int(r.ga)} | {r.xgf:.2f}-{r.xga:.2f}"
        )
    return lines


def report(pred: um.MatchPrediction, sources: list[um.SourceInfo], detail: bool) -> str:
    t, h, a = pred.trained, pred.home_snap, pred.away_snap
    att_h, def_h = pred.attack_defense("home")
    att_a, def_a = pred.attack_defense("away")
    w_recent = h["n_recent"] / (h["n_recent"] + t.shrink)
    league = um.league_name(pred.league)

    out = [SEP, "🌐 FUENTES DE DATOS LOCALIZADAS EN LA WEB", "- URLs analizadas para el scraping:"]
    for s in sources:
        cache_txt = " (caché local)" if s.from_cache else ""
        out.append(f"    {s.url}  [{um.season_label(s.season)}: {s.played} partidos con xG]{cache_txt}")
    if pred.match_id is not None:
        out.append(f"- Partido: {pred.home} vs {pred.away} | {league} | {pred.kickoff:%Y-%m-%d %H:%M} UTC | "
                   f"{pred.understat_url}")
    else:
        out.append(f"- Partido: {pred.home} vs {pred.away} | {league} | sin partido en el calendario de "
                   f"Understat en los próximos {um.FIXTURE_HORIZON_DAYS} días: se usan los datos "
                   f"disponibles a {pred.cutoff:%Y-%m-%d}")
    out.append("- Métricas extraídas:")
    out += team_metrics_lines(pred.home, "local", h)
    out += team_metrics_lines(pred.away, "visitante", a)
    if detail:
        out += detail_lines(pred.home, h) + detail_lines(pred.away, a)
    out += [
        f"- Fuerza ajustada por rival (1.00 = media liga; {w_recent:.0%} últimos {um.N_RECENT} / "
        f"{1 - w_recent:.0%} últimos 12 meses):",
        f"    {pred.home}: ataque {att_h:.2f} | defensa {def_h:.2f}"
        f"   ·   {pred.away}: ataque {att_a:.2f} | defensa {def_a:.2f}",
        f"- Media de la liga (12 meses): xG local {pred.ratings.mu_home:.2f} / visitante "
        f"{pred.ratings.mu_away:.2f} | goles local {pred.ratings.goals_home:.2f} / visitante "
        f"{pred.ratings.goals_away:.2f}",
        f"- Goles esperados del partido: λ local {pred.lam_home:.2f} | λ visitante {pred.lam_away:.2f}",
        "- Bajas/lesiones: Understat no las publica → no incluidas en el modelo (no se infieren).",
        "- Descanso: calculado solo con partidos de liga (no ve copas ni competiciones europeas).",
        SEP,
        "🧮 RESULTADOS DE LOS MODELOS (POISSON & LOGÍSTICO)",
        f"- Poisson Puro -> 1: {pct(pred.p_poisson[0])} | X: {pct(pred.p_poisson[1])} | "
        f"2: {pct(pred.p_poisson[2])}",
        f"- Regresión Logística -> 1: {pct(pred.p_logistic[0])} | X: {pct(pred.p_logistic[1])} | "
        f"2: {pct(pred.p_logistic[2])}",
        "- Marcadores exactos más probables: "
        + " | ".join(f"{i}-{j} ({pct(p)})" for i, j, p in pred.top_scores),
        f"- Poisson Puro goles -> Over 2.5: {pct(pred.over25_poisson)} | Ambos anotan: {pct(pred.btts_poisson)}",
        f"- Validación ({t.val_seasons}, {t.n_val} partidos no vistos; entrenamiento {t.n_train}) "
        f"log loss 1X2 -> Poisson {t.ll_poisson:.4f} | Logística {t.ll_logistic:.4f} | "
        f"Ensemble {t.ll_ensemble:.4f}",
        SEP,
        "🎯 PREDICCIÓN FINAL COMBINADA (ENSEMBLE MODEL)",
        f"- Pesos: Poisson {t.w_poisson:.0%} | Logística {1 - t.w_poisson:.0%} (mínimo log loss de validación)",
        f"- Mercado 1X2: Local {pct(pred.p_final[0])} | Empate {pct(pred.p_final[1])} | "
        f"Visitante {pct(pred.p_final[2])}",
        f"- Línea de Goles: Over 2.5 {pct(pred.over25)} | Under 2.5 {pct(1 - pred.over25)}",
        f"- Ambos Anotan: SÍ {pct(pred.btts)} | NO {pct(1 - pred.btts)}",
        SEP,
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predicción Poisson + regresión logística con xG de Understat.")
    parser.add_argument("local", help="Equipo local (p. ej. 'Arsenal')")
    parser.add_argument("visitante", help="Equipo visitante (p. ej. 'Leeds')")
    parser.add_argument("--liga", default="Premier League", help="Liga (Premier League, LaLiga, Serie A, ...)")
    parser.add_argument("--temporadas", type=int, default=um.DEFAULT_SEASONS_BACK,
                        help="Temporadas anteriores usadas para entrenar/validar (por defecto 4)")
    parser.add_argument("--refrescar", action="store_true", help="Ignora la caché y vuelve a descargar")
    parser.add_argument("--detalle", action="store_true", help="Muestra los últimos partidos usados de cada equipo")
    args = parser.parse_args(argv)

    try:
        league = um.resolve_league(args.liga)
        data = um.load_league(league, args.temporadas, args.refrescar, progress=log)
        log("Construyendo histórico pre-partido y entrenando modelos...")
        model = um.train_league(data, args.temporadas)
        pred = um.predict_match(model, args.local, args.visitante)
    except um.PredictionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(report(pred, data.sources, args.detalle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
