"""Optimiza pesos para M3: clientes degradados + enriquecimiento + pesos ajustados.

A diferencia de M1 (que optimiza sobre clientes completos), M3 optimiza
sobre clientes degradados enriquecidos, buscando los pesos que mejor
compensan la informacion parcial recuperada.

Uso:
    PYTHONPATH=src python3 scripts/optimize_m3_weights.py

Salida:
    reports/results/m3_weights.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imperia_matching_tfm.evaluation.ranking import ndcg_at_k, precision_at_k, recall_at_k
from imperia_matching_tfm.lead_extraction.enrichment import enrich_preference
from imperia_matching_tfm.matching.baseline import DEFAULT_WEIGHTS, score_property_for_client
from imperia_matching_tfm.models import Client, Property

K = 5
OPTIMIZE_METRIC = "ndcg"
ITERATIONS = 3
CANDIDATES = [1, 3, 5, 7, 10, 13, 15, 18, 20, 25, 30]


def load_data():
    properties_data = json.loads((ROOT / "data/synthetic/properties.json").read_text(encoding="utf-8"))
    degraded_data = json.loads((ROOT / "data/synthetic/clients_degraded.json").read_text(encoding="utf-8"))
    labels_data = json.loads((ROOT / "data/labels/relevance_synthetic.json").read_text(encoding="utf-8"))

    properties = [Property.from_dict(p) for p in properties_data]
    relevance = {(item["client_id"], item["property_id"]): int(item["relevance"]) for item in labels_data}

    clients = []
    for item in degraded_data:
        clean = {k: v for k, v in item.items() if k != "dropped_fields"}
        client = Client.from_dict(clean)
        client.preference = enrich_preference(client.preference)
        clients.append(client)

    return clients, properties, relevance


def evaluate(clients, properties, relevance, weights):
    metrics = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
    n = len(clients)

    for client in clients:
        scored = sorted(
            (score_property_for_client(client, p, weights) for p in properties),
            key=lambda s: s.score,
            reverse=True,
        )
        ranked_rel = [relevance.get((client.id, s.property_id), 0) for s in scored]
        total_relevant = sum(1 for v in ranked_rel if v >= 2)

        metrics["precision"] += precision_at_k([v > 0 for v in ranked_rel], K)
        metrics["recall"] += recall_at_k([v >= 2 for v in ranked_rel], total_relevant, K)
        metrics["ndcg"] += ndcg_at_k(ranked_rel, K)

    return {k: v / n for k, v in metrics.items()}


def optimize(clients, properties, relevance):
    best_weights = dict(DEFAULT_WEIGHTS)
    best_score = evaluate(clients, properties, relevance, best_weights)[OPTIMIZE_METRIC]

    criteria_to_tune = [k for k in best_weights if k != "operation"]

    for iteration in range(1, ITERATIONS + 1):
        improved = False
        for criterion in criteria_to_tune:
            current_best_value = best_weights[criterion]
            for candidate in CANDIDATES:
                trial = dict(best_weights)
                trial[criterion] = candidate
                result = evaluate(clients, properties, relevance, trial)[OPTIMIZE_METRIC]
                if result > best_score:
                    best_score = result
                    current_best_value = candidate
                    improved = True
            best_weights[criterion] = current_best_value

        print(f"  Iteracion {iteration}: {OPTIMIZE_METRIC}@{K} = {best_score:.4f}")
        if not improved:
            print(f"  Sin mejora en iteracion {iteration}, terminando.")
            break

    return best_weights


def main():
    clients, properties, relevance = load_data()

    print("Metricas M2 base (pesos originales + enriquecimiento):")
    m2_base = evaluate(clients, properties, relevance, DEFAULT_WEIGHTS)
    for k, v in m2_base.items():
        print(f"  {k}@{K} = {v:.4f}")

    print(f"\nOptimizando pesos M3 (metrica objetivo: {OPTIMIZE_METRIC}@{K})...")
    optimized = optimize(clients, properties, relevance)

    print(f"\nMetricas M3 (pesos optimizados + enriquecimiento):")
    m3 = evaluate(clients, properties, relevance, optimized)
    for k, v in m3.items():
        print(f"  {k}@{K} = {v:.4f}")

    # Tabla de pesos
    print(f"\n{'Criterio':<16} {'Default':>8} {'M3':>8} {'Cambio':>8}")
    print("-" * 42)
    for criterion in DEFAULT_WEIGHTS:
        w0 = DEFAULT_WEIGHTS[criterion]
        w3 = optimized[criterion]
        delta = w3 - w0
        sign = "+" if delta > 0 else ""
        print(f"{criterion:<16} {w0:>8} {w3:>8} {sign}{delta:>7}")

    # Guardar
    output_path = ROOT / "reports/results/m3_weights.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"weights": optimized, "metrics": m3, "k": K}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nPesos guardados en {output_path}")


if __name__ == "__main__":
    main()
