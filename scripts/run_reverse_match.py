from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from imperia_matching_tfm.explainability.text import summarize_match  # noqa: E402
from imperia_matching_tfm.matching.baseline import score_property_for_client  # noqa: E402
from imperia_matching_tfm.matching.hybrid_m4 import DEFAULT_ALPHA, score_property_for_client_m4  # noqa: E402
from imperia_matching_tfm.matching.semantic import load_embeddings_cache  # noqa: E402
from imperia_matching_tfm.models import Client, Property  # noqa: E402


def main() -> None:
    args = parse_args()
    clients = _load_models(resolve_path(args.clients), Client)
    properties = _load_models(resolve_path(args.properties), Property)

    target = _resolve_property(properties, args)
    if target is None:
        print("No se encontro la propiedad indicada.")
        sys.exit(1)

    embeddings_cache = None
    if args.model == "m4":
        cache_path = resolve_path(args.embeddings)
        if not cache_path.exists():
            print(f"No se encontro la cache de embeddings en {cache_path}.")
            print("Ejecuta antes: PYTHONPATH=src python3 scripts/precompute_embeddings.py")
            sys.exit(1)
        embeddings_cache = load_embeddings_cache(cache_path)

    label = "M4 Semantico" if args.model == "m4" else "M0 Baseline"
    print(f"Imperia Matching TFM - Matching inverso ({label})")
    print("=" * 70)
    print(f"Propiedad objetivo: {target.id} - {target.title}")
    print(f"  Operacion: {target.operation} | Tipo: {target.property_type}")
    print(f"  Ciudad: {target.city} | Zona: {target.zone} | Precio: {target.price}")
    print(f"Clientes evaluados: {len(clients)} | alpha={args.alpha if args.model == 'm4' else 'N/A'}")
    print("-" * 70)

    if args.model == "m4":
        scored = sorted(
            (
                score_property_for_client_m4(client, target, embeddings_cache, alpha=args.alpha)
                for client in clients
            ),
            key=lambda item: item.score,
            reverse=True,
        )
    else:
        scored = sorted(
            (score_property_for_client(client, target) for client in clients),
            key=lambda item: item.score,
            reverse=True,
        )

    top = scored[: args.top_k]
    for position, item in enumerate(top, start=1):
        client = next(c for c in clients if c.id == item.client_id)
        print(f"\n{position}. {client.name} ({client.id}) - Score: {item.score:.1f}/100")
        pref = client.preference
        print(f"   Busca: {pref.operation or '-'} | {pref.property_types or '-'}")
        print(f"   Zonas: {pref.zones or '-'} | Ciudades: {pref.cities or '-'}")
        print(f"   Presupuesto: {pref.price_min or '-'} - {pref.price_max or '-'}")
        for explanation in summarize_match(item):
            print(f"   - {explanation}")

    if args.output:
        out_path = resolve_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "client_id": item.client_id,
                "client_name": next(c.name for c in clients if c.id == item.client_id),
                "score": item.score,
                "criteria": [
                    {"name": c.name, "earned": c.earned, "max": c.max, "matched": c.matched, "note": c.note}
                    for c in item.criteria
                ],
            }
            for item in scored
        ]
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRanking completo guardado en: {out_path}")

    print("\n" + "=" * 70)
    print("Matching inverso finalizado.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rankea todos los clientes del dataset frente a una propiedad concreta."
    )
    parser.add_argument(
        "--clients",
        default="data/synthetic/clients.json",
        help="JSON de clientes/leads normalizados.",
    )
    parser.add_argument(
        "--properties",
        default="data/synthetic/properties.json",
        help="JSON de propiedades normalizadas.",
    )
    parser.add_argument(
        "--property-id",
        help="ID de la propiedad dentro del archivo de propiedades.",
    )
    parser.add_argument(
        "--property-file",
        help="Ruta a un JSON con una unica propiedad (objeto o lista de un elemento).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Numero de clientes a mostrar.")
    parser.add_argument("--output", help="Opcional: ruta JSON para guardar el ranking completo.")
    parser.add_argument(
        "--model",
        choices=["m0", "m4"],
        default="m0",
        help="m0=baseline estructurado, m4=hibrido estructurado + semantico.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Peso del scoring estructurado en M4 (1-alpha = peso semantico). Default 0.7.",
    )
    parser.add_argument(
        "--embeddings",
        default="data/processed/embeddings_cache.json",
        help="Ruta al cache de embeddings para M4.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_models(path: Path, model_class):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [model_class.from_dict(item) for item in payload]


def _resolve_property(properties: list[Property], args: argparse.Namespace) -> Property | None:
    if args.property_file:
        payload = json.loads(resolve_path(args.property_file).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            payload = payload[0]
        return Property.from_dict(payload)
    target_id = args.property_id or (properties[0].id if properties else None)
    return next((p for p in properties if p.id == target_id), None)


if __name__ == "__main__":
    main()
