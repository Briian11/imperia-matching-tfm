from __future__ import annotations

import json
from pathlib import Path

from imperia_matching_tfm.evaluation.ranking import ndcg_at_k, precision_at_k, recall_at_k
from imperia_matching_tfm.matching.baseline import score_property_for_client
from imperia_matching_tfm.models import Client, Property


ROOT = Path(__file__).resolve().parents[1]

K_VALUES = [3, 5, 10]


def main() -> None:
    labels_path = ROOT / "data/labels/relevance_synthetic.json"
    if not labels_path.exists():
        labels_path = ROOT / "data/labels/example_relevance.json"

    clients = _load_models(ROOT / "data/synthetic/clients.json", Client)
    properties = _load_models(ROOT / "data/synthetic/properties.json", Property)
    relevance = _load_relevance(labels_path)

    totals = {f"precision@{k}": 0.0 for k in K_VALUES}
    totals.update({f"recall@{k}": 0.0 for k in K_VALUES})
    totals.update({f"ndcg@{k}": 0.0 for k in K_VALUES})
    n_clients = len(clients)

    for client in clients:
        scored = sorted(
            (score_property_for_client(client, property_) for property_ in properties),
            key=lambda item: item.score,
            reverse=True,
        )
        ranked_relevance = [relevance.get((client.id, item.property_id), 0) for item in scored]
        total_relevant = sum(1 for v in ranked_relevance if v >= 2)

        print(f"\nCliente: {client.name} ({client.id})")
        for k in K_VALUES:
            p = precision_at_k([v > 0 for v in ranked_relevance], k)
            r = recall_at_k([v >= 2 for v in ranked_relevance], total_relevant, k)
            n = ndcg_at_k(ranked_relevance, k)
            totals[f"precision@{k}"] += p
            totals[f"recall@{k}"] += r
            totals[f"ndcg@{k}"] += n
            print(f"  Precision@{k}={p:.2f}  Recall@{k}={r:.2f}  NDCG@{k}={n:.3f}")

        for position, item in enumerate(scored[:5], start=1):
            rel = relevance.get((client.id, item.property_id), 0)
            print(f"  {position}. {item.property_id} score={item.score:.1f} rel={rel}")

    # Tabla resumen
    print("\n" + "=" * 60)
    print("RESUMEN M0 - Baseline estructurado")
    print("=" * 60)
    print(f"{'Metrica':<20} {'Valor':>10}")
    print("-" * 30)
    for key, total in totals.items():
        avg = total / n_clients
        print(f"{key:<20} {avg:>10.3f}")
    print("-" * 30)
    print(f"{'Clientes':<20} {n_clients:>10}")
    print(f"{'Propiedades':<20} {len(properties):>10}")
    print(f"{'Etiquetas':<20} {len(relevance):>10}")


def _load_models(path: Path, model_class):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [model_class.from_dict(item) for item in payload]


def _load_relevance(path: Path) -> dict[tuple[str, str], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(item["client_id"], item["property_id"]): int(item["relevance"]) for item in payload}


if __name__ == "__main__":
    main()
