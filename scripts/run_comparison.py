"""Ejecuta M0, M1, M2, M3 y M4 y muestra tabla comparativa completa.

Uso:
    PYTHONPATH=src python3 scripts/run_comparison.py
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
from imperia_matching_tfm.matching.hybrid_m4 import DEFAULT_ALPHA, score_property_for_client_m4
from imperia_matching_tfm.matching.semantic import load_embeddings_cache
from imperia_matching_tfm.models import Client, Property

K_VALUES = [3, 5, 10]


def load_data():
    clients_data = json.loads((ROOT / "data/synthetic/clients.json").read_text(encoding="utf-8"))
    properties_data = json.loads((ROOT / "data/synthetic/properties.json").read_text(encoding="utf-8"))
    labels_path = ROOT / "data/labels/relevance_synthetic.json"
    if not labels_path.exists():
        labels_path = ROOT / "data/labels/example_relevance.json"
    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))

    properties = [Property.from_dict(p) for p in properties_data]
    relevance = {(item["client_id"], item["property_id"]): int(item["relevance"]) for item in labels_data}

    clients_degraded_path = ROOT / "data/synthetic/clients_degraded.json"
    clients_degraded_data = None
    if clients_degraded_path.exists():
        clients_degraded_data = json.loads(clients_degraded_path.read_text(encoding="utf-8"))

    return clients_data, clients_degraded_data, properties, relevance


def load_weights(filename: str) -> dict[str, float] | None:
    path = ROOT / "reports/results" / filename
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["weights"]


def build_clients(clients_data: list[dict], enrich: bool = False) -> list[Client]:
    clients = []
    for item in clients_data:
        clean = {k: v for k, v in item.items() if k != "dropped_fields"}
        client = Client.from_dict(clean)
        if enrich:
            client.preference = enrich_preference(client.preference)
        clients.append(client)
    return clients


def evaluate_model(clients, properties, relevance, weights):
    results = {}
    for k in K_VALUES:
        results[f"precision@{k}"] = 0.0
        results[f"recall@{k}"] = 0.0
        results[f"ndcg@{k}"] = 0.0

    n = len(clients)
    for client in clients:
        scored = sorted(
            (score_property_for_client(client, p, weights) for p in properties),
            key=lambda s: s.score,
            reverse=True,
        )
        ranked_rel = [relevance.get((client.id, s.property_id), 0) for s in scored]
        total_relevant = sum(1 for v in ranked_rel if v >= 2)

        for k in K_VALUES:
            results[f"precision@{k}"] += precision_at_k([v > 0 for v in ranked_rel], k)
            results[f"recall@{k}"] += recall_at_k([v >= 2 for v in ranked_rel], total_relevant, k)
            results[f"ndcg@{k}"] += ndcg_at_k(ranked_rel, k)

    return {key: val / n for key, val in results.items()}


def evaluate_m4(clients, properties, relevance, weights, embeddings_cache, alpha):
    results = {}
    for k in K_VALUES:
        results[f"precision@{k}"] = 0.0
        results[f"recall@{k}"] = 0.0
        results[f"ndcg@{k}"] = 0.0

    n = len(clients)
    for client in clients:
        scored = sorted(
            (
                score_property_for_client_m4(client, p, embeddings_cache, alpha=alpha, weights=weights)
                for p in properties
            ),
            key=lambda s: s.score,
            reverse=True,
        )
        ranked_rel = [relevance.get((client.id, s.property_id), 0) for s in scored]
        total_relevant = sum(1 for v in ranked_rel if v >= 2)

        for k in K_VALUES:
            results[f"precision@{k}"] += precision_at_k([v > 0 for v in ranked_rel], k)
            results[f"recall@{k}"] += recall_at_k([v >= 2 for v in ranked_rel], total_relevant, k)
            results[f"ndcg@{k}"] += ndcg_at_k(ranked_rel, k)

    return {key: val / n for key, val in results.items()}


def main():
    clients_data, clients_degraded_data, properties, relevance = load_data()
    clients_full = build_clients(clients_data)

    all_results = {}

    # M0 Baseline
    all_results["M0 Baseline"] = evaluate_model(clients_full, properties, relevance, DEFAULT_WEIGHTS)

    # M1 Ajustado (pesos optimizados, clientes completos)
    m1_weights = load_weights("m1_weights.json")
    if m1_weights:
        all_results["M1 Ajustado"] = evaluate_model(clients_full, properties, relevance, m1_weights)

    # M2 Enriquecido (clientes degradados + enriquecimiento, pesos default)
    if clients_degraded_data:
        clients_enriched = build_clients(clients_degraded_data, enrich=True)
        all_results["M2 Enriquecido"] = evaluate_model(clients_enriched, properties, relevance, DEFAULT_WEIGHTS)

    # M3 Hibrido (clientes degradados + enriquecimiento + pesos optimizados)
    m3_weights = load_weights("m3_weights.json")
    if clients_degraded_data and m3_weights:
        clients_enriched_m3 = build_clients(clients_degraded_data, enrich=True)
        all_results["M3 Hibrido"] = evaluate_model(clients_enriched_m3, properties, relevance, m3_weights)

    # M4 Semantico (estructurado + embeddings, pesos baseline, clientes completos)
    embeddings_path = ROOT / "data/processed/embeddings_cache.json"
    if embeddings_path.exists():
        embeddings_cache = load_embeddings_cache(embeddings_path)
        all_results["M4 Semantico"] = evaluate_m4(
            clients_full, properties, relevance, DEFAULT_WEIGHTS, embeddings_cache, alpha=DEFAULT_ALPHA
        )

        # M4d: misma capa semantica sobre clientes degradados sin enriquecimiento.
        # Demuestra si la afinidad textual rescata informacion perdida.
        if clients_degraded_data:
            clients_degraded = build_clients(clients_degraded_data, enrich=False)
            all_results["M4d Semantico+Degradado"] = evaluate_m4(
                clients_degraded, properties, relevance, DEFAULT_WEIGHTS, embeddings_cache, alpha=DEFAULT_ALPHA
            )
            # Referencia: estructurado puro sobre clientes degradados (sin enriquecer)
            all_results["M0d Degradado"] = evaluate_model(
                clients_degraded, properties, relevance, DEFAULT_WEIGHTS
            )

    # Tabla comparativa
    metric_keys = list(next(iter(all_results.values())).keys())
    model_names = list(all_results.keys())
    col_width = max(len(n) for n in model_names) + 2
    header = f"{'Metrica':<16}" + "".join(f"{name:>{col_width}}" for name in model_names)

    print("\n" + "=" * len(header))
    print("COMPARATIVA DE MODELOS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for metric in metric_keys:
        row = f"{metric:<16}"
        for name in model_names:
            row += f"{all_results[name][metric]:>{col_width}.4f}"
        print(row)

    print("-" * len(header))
    print(f"Clientes: {len(clients_full)} | Propiedades: {len(properties)} | Etiquetas: {len(relevance)}")

    # Guardar resultados
    output_path = ROOT / "reports/results/comparison_all.json"
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nResultados guardados en {output_path}")


if __name__ == "__main__":
    main()
