from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from imperia_matching_tfm.evaluation.ranking import ndcg_at_k, precision_at_k  # noqa: E402
from imperia_matching_tfm.explainability.text import summarize_match  # noqa: E402
from imperia_matching_tfm.matching.baseline import score_property_for_client  # noqa: E402
from imperia_matching_tfm.models import Client, Property  # noqa: E402


def main() -> None:
    args = parse_args()
    clients_path = resolve_path(args.clients)
    properties_path = resolve_path(args.properties)
    labels_path = resolve_path(args.labels) if args.labels else None

    clients = _load_models(clients_path, Client)
    properties = _load_models(properties_path, Property)
    relevance = _load_relevance(labels_path) if labels_path and labels_path.exists() else {}

    print("Imperia Matching TFM - Demo reproducible")
    print("=" * 45)
    print("Modo: baseline estructurado")
    print("Dependencias externas: ninguna")
    print(f"Clientes: {clients_path}")
    print(f"Propiedades: {properties_path}")
    if labels_path:
        print(f"Etiquetas: {labels_path}")
    else:
        print("Etiquetas: no indicadas")

    if not properties:
        print("\nNo hay propiedades cargadas. Revisa el archivo de entrada.")
        print("Demo finalizada sin rankings.")
        return

    for client in clients:
        scored = sorted(
            (score_property_for_client(client, property_) for property_ in properties),
            key=lambda item: item.score,
            reverse=True,
        )
        ranked_relevance = [relevance.get((client.id, item.property_id), 0) for item in scored]
        has_client_labels = any((client.id, property_.id) in relevance for property_ in properties)

        print("\n" + "-" * 45)
        print(f"Cliente: {client.name}")
        print(f"ID: {client.id}")
        if has_client_labels:
            print(f"Precision@{args.metric_k}: {precision_at_k([value > 0 for value in ranked_relevance], args.metric_k):.2f}")
            print(f"NDCG@{args.metric_k}: {ndcg_at_k(ranked_relevance, args.metric_k):.2f}")
        else:
            print("Metricas: no disponibles para este dataset porque no hay etiquetas.")
        print("\nTop recomendaciones:")

        for position, item in enumerate(scored[: args.top_k], start=1):
            property_ = next(candidate for candidate in properties if candidate.id == item.property_id)
            print(f"{position}. {property_.title}")
            print(f"   Propiedad: {item.property_id}")
            print(f"   Score: {item.score:.1f}/100")
            if (client.id, item.property_id) in relevance:
                print(f"   Relevancia esperada: {relevance[(client.id, item.property_id)]}")
            for explanation in summarize_match(item):
                print(f"   - {explanation}")

    print("\n" + "=" * 45)
    print("Demo finalizada correctamente.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo reproducible del TFM de matching inmobiliario.")
    parser.add_argument(
        "--clients",
        default="data/synthetic/clients.json",
        help="JSON de clientes o leads normalizados.",
    )
    parser.add_argument(
        "--properties",
        default="data/synthetic/properties.json",
        help="JSON de propiedades normalizadas.",
    )
    parser.add_argument(
        "--labels",
        default="data/labels/example_relevance.json",
        help="JSON de etiquetas de relevancia. Usa --labels '' para omitir.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Numero de recomendaciones a mostrar.")
    parser.add_argument("--metric-k", type=int, default=3, help="K usado para Precision@K y NDCG@K.")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_models(path: Path, model_class):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [model_class.from_dict(item) for item in payload]


def _load_relevance(path: Path | None) -> dict[tuple[str, str], int]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(item["client_id"], item["property_id"]): int(item["relevance"]) for item in payload}


if __name__ == "__main__":
    main()
