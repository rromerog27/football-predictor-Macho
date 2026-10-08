#!/usr/bin/env python3
"""Predicción de un partido: Poisson (Dixon-Coles) + regresión logística.

Script de terminal sobre `src/match_model.py` y `src/competitions.py` (el mismo
modelo que usa la página "Partidos del día" de la app). Usa xG real de
Understat en las 6 ligas que cubre y, en el resto de ligas y copas, el xG
aproximado con tiros de ESPN; donde hay cuotas de cierre de partidos anteriores
(football-data.co.uk o ESPN), también la fuerza que les da el mercado. Ver el
README para la descripción del modelo.

Uso:
    python3 modelo_prediccion.py "Arsenal" "Leeds" --liga "Premier League"
    python3 modelo_prediccion.py "América" "Monterrey" --liga "Liga MX" --detalle
    python3 modelo_prediccion.py "Arsenal" "Leeds" --backtest     # + comparación con las cuotas de cierre
    python3 modelo_prediccion.py --listar
"""

from __future__ import annotations

import argparse
import sys

from src import backtest
from src import competitions as comps
from src import match_model as mm

SEP = "=" * 50


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def team_metrics_lines(name: str, role: str, snap: dict, signal: str) -> list[str]:
    recent = snap["recent_rows"]
    venue_txt = "en casa" if role == "local" else "fuera"
    first, last = recent["datetime"].iloc[0], recent["datetime"].iloc[-1]
    with_odds = recent[recent["mkt_real"]] if "mkt_real" in recent else recent.iloc[0:0]
    market = ([f"      Mercado (goles esperados según las cuotas de cierre, {len(with_odds)} de esos partidos): "
               f"a favor {with_odds['mf'].mean():.2f} | en contra {with_odds['ma'].mean():.2f}"]
              if len(with_odds) else [])
    return [
        f"    {name} ({role}) — últimos {len(recent)} partidos [{first:%Y-%m-%d} → {last:%Y-%m-%d}]",
        f"      {signal} a favor {recent['sf'].mean():.2f} | en contra {recent['sa'].mean():.2f} | "
        f"Goles a favor {recent['gf'].mean():.2f} | Goles en contra {recent['ga'].mean():.2f} (por partido)",
        f"      Forma últ. {len(snap['form_rows'])}: {mm.form_string(snap['form_rows'])} "
        f"= {int(snap['form_rows']['pts'].sum())} pts | Últ. {len(snap['venue_rows'])} {venue_txt}: "
        f"{mm.form_string(snap['venue_rows'])} = {int(snap['venue_rows']['pts'].sum())} pts | "
        f"Descanso: {snap['rest_days']:.0f} días ({mm.rest_category(snap['rest_days'])})",
        *market,
    ]


def detail_lines(name: str, snap: dict) -> list[str]:
    lines = [f"    {name}: fecha | sede | rival | resultado | señal (a favor-en contra)"]
    for r in snap["recent_rows"].itertuples(index=False):
        lines.append(
            f"      {r.datetime:%Y-%m-%d} | {'L' if r.venue == 'h' else 'V'} | {r.opp:<26} | "
            f"{int(r.gf)}-{int(r.ga)} | {r.sf:.2f}-{r.sa:.2f}"
        )
    return lines


def report(pred: mm.MatchPrediction, data: mm.LeagueData, detail: bool) -> str:
    t, h, a = pred.trained, pred.home_snap, pred.away_snap
    att_h, def_h = pred.attack_defense("home")
    att_a, def_a = pred.attack_defense("away")
    w_recent = h["n_recent"] / (h["n_recent"] + t.shrink)
    signal = pred.signal_short

    out = [SEP, "🌐 FUENTES DE DATOS LOCALIZADAS EN LA WEB", "- URLs analizadas para el scraping:"]
    for s in data.sources:
        cache_txt = " (caché local)" if s.from_cache else ""
        out.append(f"    {s.url}  [{s.label}: {s.played} partidos jugados]{cache_txt}")
    where = " (campo neutral)" if pred.neutral else ""
    if pred.match_id is not None:
        out.append(f"- Partido: {pred.home} vs {pred.away}{where} | {data.name} | {pred.kickoff:%Y-%m-%d %H:%M} UTC"
                   + (f" | {comps.match_url(pred.match_id)}" if comps.match_url(pred.match_id) else ""))
    else:
        out.append(f"- Partido: {pred.home} vs {pred.away} | {data.name} | sin partido en el calendario en los "
                   f"próximos {mm.FIXTURE_HORIZON_DAYS} días: se usan los datos disponibles a {pred.cutoff:%Y-%m-%d}")
    out.append(f"- Señal de calidad de ocasiones: {data.signal_name}")
    if t.market_weight:
        how = "elegido con datos" if data.market_mode == "learn" else "típico: solo hay cuotas recientes"
        out.append(f"- Señal de mercado: goles esperados implícitos en las cuotas de cierre de los partidos "
                   f"anteriores ({data.market_coverage:.0%} de los partidos de los últimos 12 meses; peso "
                   f"{t.market_weight:.0%}, {how})")
    else:
        out.append("- Señal de mercado: sin cuotas de cierre de partidos anteriores suficientes → no se usa")
    out.append("- Métricas extraídas:")
    out += team_metrics_lines(pred.home, "local", h, signal)
    out += team_metrics_lines(pred.away, "visitante", a, signal)
    if detail:
        out += detail_lines(pred.home, h) + detail_lines(pred.away, a)
    if pred.low_data:
        out.append(f"- ⚠ Pocos datos: algún equipo tiene menos de {mm.MIN_MATCHES} partidos en los últimos 12 meses.")
    if pred.newcomers:
        out.append(f"- Recién llegado a la competición: {', '.join(pred.newcomers)} (se aplica la calibración de "
                   f"ascendidos: ×{t.newcomer[0]:.2f} a sus goles, ×{t.newcomer[1]:.2f} a los del rival).")
    decay = f"vida media {mm.DECAY_HALF_LIFE_DAYS:.0f} días" if mm.DECAY_HALF_LIFE_DAYS else "sin decaimiento"
    mix = f"{t.signal_weight:.0%} {signal} / {1 - t.signal_weight:.0%} goles"
    if t.market_weight:
        mix = f"{t.market_weight:.0%} mercado + {1 - t.market_weight:.0%} ({mix})"
    out += [
        f"- Fuerza ajustada por rival (1.00 = media; 12 meses ponderados por antigüedad, {decay}; "
        f"{w_recent:.0%} forma de los últimos {mm.N_RECENT}; {mix}):",
        f"    {pred.home}: ataque {att_h:.2f} | defensa {def_h:.2f}"
        f"   ·   {pred.away}: ataque {att_a:.2f} | defensa {def_a:.2f}",
        f"- Goles medios de la competición (12 meses): local {pred.base_home:.2f} / visitante {pred.base_away:.2f}",
        f"- Goles esperados del partido: λ local {pred.lam_home:.2f} | λ visitante {pred.lam_away:.2f}",
        "- Bajas/lesiones del partido: no incluidas (solo de forma indirecta, vía las cuotas de partidos anteriores).",
        "- Descanso: calculado solo con los partidos de las competiciones descargadas.",
        SEP,
        "🧮 RESULTADOS DE LOS MODELOS (POISSON & LOGÍSTICO)",
        f"- Poisson (Dixon-Coles ρ={t.rho:+.3f}) -> 1: {pct(pred.p_poisson[0])} | X: {pct(pred.p_poisson[1])} | "
        f"2: {pct(pred.p_poisson[2])}",
        f"- Regresión Logística -> 1: {pct(pred.p_logistic[0])} | X: {pct(pred.p_logistic[1])} | "
        f"2: {pct(pred.p_logistic[2])}",
        "- Marcadores exactos más probables: "
        + " | ".join(f"{i}-{j} ({pct(p)})" for i, j, p in pred.top_scores),
        f"- Poisson goles -> Over 2.5: {pct(pred.over25_poisson)} | Ambos anotan: {pct(pred.btts_poisson)}",
        f"- Validación ({t.val_period}, {t.n_val} partidos no vistos; entrenamiento {t.n_train}) "
        f"log loss 1X2 -> Poisson {t.ll_poisson:.4f} | Logística {t.ll_logistic:.4f} | "
        f"Ensemble {t.ll_ensemble:.4f}"
        + (f" | solo {data.name}: {t.ll_ensemble_focus:.4f} ({t.n_val_focus} partidos)"
           if t.ll_ensemble_focus is not None else ""),
        SEP,
        "🎯 PREDICCIÓN FINAL COMBINADA (ENSEMBLE MODEL)",
        f"- Pesos: Poisson {t.w_poisson:.0%} | Logística {1 - t.w_poisson:.0%} (mínimo log loss de validación)",
        f"- Mercado 1X2: Local {pct(pred.p_final[0])} | Empate {pct(pred.p_final[1])} | "
        f"Visitante {pct(pred.p_final[2])}",
        f"- Línea de Goles: Over 2.5 {pct(pred.over25)} | Under 2.5 {pct(1 - pred.over25)}",
        f"- Ambos Anotan: SÍ {pct(pred.btts)} | NO {pct(1 - pred.btts)}",
    ]
    if pred.market:
        pm = pred.market["p_1x2"]
        diff = pred.p_final - pm
        out += [
            SEP,
            "💱 MERCADO (cuotas DraftKings vía ESPN, sin margen de la casa)",
            f"- 1X2 mercado: Local {pct(pm[0])} | Empate {pct(pm[1])} | Visitante {pct(pm[2])} "
            f"(margen {pred.market['margin'] * 100:.1f}%)",
            f"- Modelo − mercado: Local {diff[0] * 100:+.1f} | Empate {diff[1] * 100:+.1f} | "
            f"Visitante {diff[2] * 100:+.1f} puntos",
        ]
        if pred.market["over25"] is not None:
            out.append(f"- Over 2.5 mercado: {pct(pred.market['over25'])} (modelo {pct(pred.over25)})")
    out.append(SEP)
    return "\n".join(out)


def backtest_report(result) -> str:
    r = result
    lines = [
        SEP,
        "📊 BACKTEST CONTRA EL MERCADO (cuotas de cierre)",
        f"- Partidos: {r.n_matched} de {r.n_val} de validación ({r.period}) · cuotas: {r.odds_source}",
        f"- Log loss 1X2 -> Modelo {r.ll_model:.4f} (Poisson {r.ll_poisson:.4f}, Logística {r.ll_logistic:.4f}) | "
        f"Mercado {r.ll_market:.4f} | Frecuencias {r.ll_baseline:.4f}",
        f"- Distancia al mercado: {r.gap:+.4f} (negativo = el modelo es mejor que el cierre)",
        f"- Mezcla modelo + mercado: peso del modelo {r.alpha:.0%} → log loss {r.ll_blend_cv:.4f} "
        "(validación cruzada; si el peso es ~0, el modelo no añade información al mercado)",
    ]
    if r.ll_ou_model is not None:
        lines.append(f"- Over/Under 2.5 ({r.n_ou} partidos): modelo {r.ll_ou_model:.4f} | mercado {r.ll_ou_market:.4f}")
    roi = " | ".join(f"{row.umbral}: {row.apuestas} apuestas, ROI {row.ROI * 100:+.1f}%" if row.apuestas
                     else f"{row.umbral}: sin apuestas" for row in r.roi.itertuples())
    lines += [f"- ROI simulado (1 unidad a la cuota de cierre cuando el modelo supera al mercado): {roi}", SEP]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predicción Poisson + regresión logística (xG / tiros / goles).")
    parser.add_argument("local", nargs="?", help="Equipo local (p. ej. 'Arsenal')")
    parser.add_argument("visitante", nargs="?", help="Equipo visitante (p. ej. 'Leeds')")
    parser.add_argument("--liga", default="Premier League", help="Liga o copa (Premier League, Liga MX, Champions...)")
    parser.add_argument("--refrescar", action="store_true", help="Ignora la caché y vuelve a descargar")
    parser.add_argument("--detalle", action="store_true", help="Muestra los últimos partidos usados de cada equipo")
    parser.add_argument("--listar", action="store_true", help="Lista las ligas y copas disponibles")
    parser.add_argument("--backtest", action="store_true",
                        help="Compara el modelo con las cuotas de cierre (football-data.co.uk o ESPN)")
    args = parser.parse_args(argv)

    if args.listar:
        for region in comps.REGIONS:
            print(f"{region}:")
            for c in comps.COMPETITIONS:
                if c.region == region:
                    print(f"  {c.name:<32} ({c.code}; {comps.signal_label(c)})")
        return 0
    if not args.local or not args.visitante:
        parser.error("indica local y visitante (o usa --listar)")

    try:
        comp = comps.resolve_competition(args.liga)
        data = comps.load_competition(comp, args.refrescar, progress=log)
        log("Construyendo histórico pre-partido y entrenando modelos...")
        model = mm.train_league(data)
        pred = mm.predict_match(model, args.local, args.visitante)
        if comp.understat and pred.match_id is not None:  # cuotas de ESPN para el partido de Understat
            fixture = comps.understat_fixture_with_odds(comp, data, pred.match_id)
            if fixture is not None:
                pred = mm.predict_match(model, pred.home, pred.away, fixture=fixture)
    except mm.PredictionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(report(pred, data, args.detalle))
    if args.backtest:
        try:
            print(backtest_report(backtest.run(model)))
        except mm.PredictionError as exc:
            print(f"Backtest no disponible: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
