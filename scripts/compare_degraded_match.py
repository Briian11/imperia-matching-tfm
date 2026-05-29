"""Compara el ranking M0 vs M4 sobre clientes degradados para una propiedad.

Para una propiedad concreta, evalua todos los clientes (en su version degradada,
con campos eliminados) usando dos estrategias:

  - M0: scoring estructurado puro. No tiene acceso a las notas del cliente.
  - M4: scoring estructurado + embeddings. Las notas si entran via similitud.

El objetivo es visualizar que clientes "rescata" la capa semantica cuando el
estructurado pierde campos.

Uso:
    PYTHONPATH=src python3 scripts/compare_degraded_match.py --property-id p_003 --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imperia_matching_tfm.matching.baseline import score_property_for_client  # noqa: E402
from imperia_matching_tfm.matching.hybrid_m4 import DEFAULT_ALPHA, score_property_for_client_m4  # noqa: E402
from imperia_matching_tfm.matching.semantic import load_embeddings_cache  # noqa: E402
from imperia_matching_tfm.models import Client, Property  # noqa: E402


def main() -> None:
    args = parse_args()

    properties = [Property.from_dict(p) for p in json.loads((ROOT / args.properties).read_text(encoding="utf-8"))]
    degraded = json.loads((ROOT / args.clients_degraded).read_text(encoding="utf-8"))
    clients = []
    dropped_by_id = {}
    for item in degraded:
        dropped_by_id[item["id"]] = item.get("dropped_fields", [])
        clean = {k: v for k, v in item.items() if k != "dropped_fields"}
        clients.append(Client.from_dict(clean))

    cache = load_embeddings_cache(ROOT / args.embeddings)

    target = next((p for p in properties if p.id == args.property_id), None)
    if target is None:
        print(f"No se encontro la propiedad {args.property_id}")
        sys.exit(1)

    print("=" * 80)
    print(f"Propiedad objetivo: {target.id} - {target.title}")
    print(f"  Tipo: {target.property_type.value} | Operacion: {target.operation.value}")
    print(f"  Ciudad: {target.city} | Zona: {target.zone} | Precio: {target.price}")
    print(f"  Features: {', '.join(target.features) if target.features else '-'}")
    print(f"  Descripcion: {target.description or '-'}")
    print("=" * 80)

    m0_ranking = sorted(
        ((c, score_property_for_client(c, target)) for c in clients),
        key=lambda pair: pair[1].score,
        reverse=True,
    )
    m4_ranking = sorted(
        ((c, score_property_for_client_m4(c, target, cache, alpha=args.alpha)) for c in clients),
        key=lambda pair: pair[1].score,
        reverse=True,
    )

    m0_pos = {c.id: i for i, (c, _) in enumerate(m0_ranking, start=1)}
    m4_pos = {c.id: i for i, (c, _) in enumerate(m4_ranking, start=1)}

    print(f"\nTop {args.top_k} en M4 (clientes degradados, alpha={args.alpha}):\n")
    print(f"{'#':<3} {'Cliente':<22} {'Score M4':>9} {'Pos M0':>7} {'Δ pos':>7}  Campos perdidos")
    print("-" * 100)
    for rank_m4, (client, m4_match) in enumerate(m4_ranking[: args.top_k], start=1):
        pos_m0 = m0_pos[client.id]
        delta = pos_m0 - rank_m4
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        dropped = ", ".join(dropped_by_id.get(client.id, [])) or "-"
        print(f"{rank_m4:<3} {client.name:<22} {m4_match.score:>9.2f} {pos_m0:>7} {delta_str:>7}  {dropped}")

    print("\nClientes que M4 RESCATA (suben mas posiciones vs M0):")
    rescued = sorted(
        [
            (c, m4_pos[c.id], m0_pos[c.id], m0_pos[c.id] - m4_pos[c.id])
            for c in clients
            if m0_pos[c.id] - m4_pos[c.id] > 0
        ],
        key=lambda t: t[3],
        reverse=True,
    )[:5]

    if not rescued:
        print("  (ninguno: el ranking no cambia)")
    else:
        for client, pos_m4, pos_m0, delta in rescued:
            print(f"\n  {client.name} ({client.id}): pos {pos_m0} -> {pos_m4} (+{delta})")
            print(f"    Notas: {client.preference.notes}")
            print(f"    Campos perdidos: {', '.join(dropped_by_id.get(client.id, []))}")
            m4_match = next(m for c, m in m4_ranking if c.id == client.id)
            sem = next((c for c in m4_match.criteria if c.name == "semantic"), None)
            if sem:
                print(f"    Afinidad semantica: {sem.note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--property-id", default="p_003")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--properties", default="data/synthetic/properties.json")
    parser.add_argument("--clients-degraded", default="data/synthetic/clients_degraded.json")
    parser.add_argument("--embeddings", default="data/processed/embeddings_cache_degraded.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
