"""M2: Matching con preferencias enriquecidas desde texto de leads.

Compara tres escenarios:
  - M0: Baseline con clientes completos (referencia ideal)
  - M0-degraded: Baseline con clientes degradados (sin enriquecimiento)
  - M2: Baseline con clientes degradados + enriquecimiento desde notas

Esto mide el impacto de la extraccion automatica de preferencias.

Uso:
    PYTHONPATH=src python3 scripts/run_m2_enrichment.py
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
from imperia_matching_tfm.models import BuyerPreference, Client, Property

K_VALUES = [3, 5, 10]


def load_data():
    properties_data = json.loads((ROOT / "data/synthetic/properties.json").read_text(encoding="utf-8"))
    clients_full = json.loads((ROOT / "data/synthetic/clients.json").read_text(encoding="utf-8"))
    clients_degraded = json.loads((ROOT / "data/synthetic/clients_degraded.json").read_text(encoding="utf-8"))
    labels_data = json.loads((ROOT / "data/labels/relevance_synthetic.json").read_text(encoding="utf-8"))

    properties = [Property.from_dict(p) for p in properties_data]
    relevance = {(item["client_id"], item["property_id"]): int(item["relevance"]) for item in labels_data}
    return properties, clients_full, clients_degraded, relevance


def build_clients(clients_data: list[dict], enrich: bool = False) -> list[Client]:
    clients = []
    for item in clients_data:
        # Excluir dropped_fields del dict antes de parsear
        clean = {k: v for k, v in item.items() if k != "dropped_fields"}
        client = Client.from_dict(clean)
        if enrich:
            client.preference = enrich_preference(client.preference)
        clients.append(client)
    return clients


def evaluate(clients, properties, relevance, weights=None):
    w = weights or DEFAULT_WEIGHTS
    results = {f"{m}@{k}": 0.0 for m in ("precision", "recall", "ndcg") for k in K_VALUES}
    n = len(clients)

    for client in clients:
        scored = sorted(
            (score_property_for_client(client, p, w) for p in properties),
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


def measure_enrichment_recovery(clients_full, clients_degraded):
    """Mide cuantos campos degradados se recuperan con el enriquecimiento."""
    total_dropped = 0
    total_recovered = 0

    for full_data, deg_data in zip(clients_full, clients_degraded):
        dropped_fields = deg_data.get("dropped_fields", [])
        if not dropped_fields:
            continue

        full_pref = BuyerPreference.from_dict(full_data["preference"])
        deg_clean = {k: v for k, v in deg_data.items() if k != "dropped_fields"}
        deg_client = Client.from_dict(deg_clean)
        enriched_pref = enrich_preference(deg_client.preference)

        for field in dropped_fields:
            total_dropped += 1
            original_val = getattr(full_pref, field)
            enriched_val = getattr(enriched_pref, field)

            if isinstance(original_val, list):
                if enriched_val:  # se recupero algo
                    total_recovered += 1
            else:
                if enriched_val is not None:
                    total_recovered += 1

    return total_dropped, total_recovered


def main():
    properties, clients_full_data, clients_degraded_data, relevance = load_data()

    # Tres escenarios
    clients_full = build_clients(clients_full_data)
    clients_degraded = build_clients(clients_degraded_data, enrich=False)
    clients_enriched = build_clients(clients_degraded_data, enrich=True)

    m0_full = evaluate(clients_full, properties, relevance)
    m0_degraded = evaluate(clients_degraded, properties, relevance)
    m2_enriched = evaluate(clients_enriched, properties, relevance)

    models = {
        "M0 Completo": m0_full,
        "M0 Degradado": m0_degraded,
        "M2 Enriquecido": m2_enriched,
    }

    # Tabla comparativa
    metric_keys = list(m0_full.keys())
    print("\n" + "=" * 72)
    print("COMPARATIVA M2 - Enriquecimiento desde leads")
    print("=" * 72)
    header = f"{'Metrica':<16}" + "".join(f"{name:>18}" for name in models)
    print(header)
    print("-" * len(header))

    for metric in metric_keys:
        row = f"{metric:<16}"
        for name in models:
            row += f"{models[name][metric]:>18.4f}"
        print(row)

    # Deltas
    print("\n--- Impacto de la degradacion (M0 Degradado vs M0 Completo) ---")
    for metric in metric_keys:
        delta = m0_degraded[metric] - m0_full[metric]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<16} {sign}{delta:.4f}")

    print("\n--- Recuperacion por enriquecimiento (M2 vs M0 Degradado) ---")
    for metric in metric_keys:
        delta = m2_enriched[metric] - m0_degraded[metric]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<16} {sign}{delta:.4f}")

    print("\n--- Gap restante (M2 vs M0 Completo) ---")
    for metric in metric_keys:
        delta = m2_enriched[metric] - m0_full[metric]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<16} {sign}{delta:.4f}")

    # Estadisticas de recuperacion de campos
    total_dropped, total_recovered = measure_enrichment_recovery(
        clients_full_data, clients_degraded_data,
    )
    recovery_pct = (total_recovered / total_dropped * 100) if total_dropped else 0
    print(f"\n--- Recuperacion de campos ---")
    print(f"  Campos eliminados:   {total_dropped}")
    print(f"  Campos recuperados:  {total_recovered}")
    print(f"  Tasa de recuperacion: {recovery_pct:.1f}%")

    # Guardar
    output = {
        "models": {name: metrics for name, metrics in models.items()},
        "field_recovery": {
            "dropped": total_dropped,
            "recovered": total_recovered,
            "recovery_rate": round(recovery_pct, 1),
        },
    }
    output_path = ROOT / "reports/results/m2_enrichment.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nResultados guardados en {output_path}")


if __name__ == "__main__":
    main()
