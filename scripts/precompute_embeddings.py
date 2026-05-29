"""Precomputa embeddings de clientes y propiedades y los guarda en cache.

Uso:
    PYTHONPATH=src python3 scripts/precompute_embeddings.py

Salida:
    data/processed/embeddings_cache.json

Tras ejecutarlo, M4 se puede evaluar sin volver a cargar el modelo,
manteniendo la reproducibilidad del flujo offline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imperia_matching_tfm.matching.semantic import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    build_client_text,
    build_property_text,
    encode_texts,
    save_embeddings_cache,
)
from imperia_matching_tfm.models import Client, Property  # noqa: E402


CLIENTS_PATH = ROOT / "data/synthetic/clients.json"
CLIENTS_DEGRADED_PATH = ROOT / "data/synthetic/clients_degraded.json"
PROPERTIES_PATH = ROOT / "data/synthetic/properties.json"
CACHE_PATH = ROOT / "data/processed/embeddings_cache.json"
CACHE_DEGRADED_PATH = ROOT / "data/processed/embeddings_cache_degraded.json"


def _build_client_from_raw(payload: dict) -> Client:
    clean = {k: v for k, v in payload.items() if k != "dropped_fields"}
    return Client.from_dict(clean)


def main() -> None:
    clients_data = json.loads(CLIENTS_PATH.read_text(encoding="utf-8"))
    properties_data = json.loads(PROPERTIES_PATH.read_text(encoding="utf-8"))

    clients = [Client.from_dict(c) for c in clients_data]
    properties = [Property.from_dict(p) for p in properties_data]

    print(f"Modelo: {DEFAULT_MODEL_NAME}")
    print(f"Codificando {len(properties)} propiedades...")
    property_texts = [build_property_text(p) for p in properties]
    property_vectors = encode_texts(property_texts)

    print(f"Codificando {len(clients)} clientes completos...")
    client_texts = [build_client_text(c) for c in clients]
    client_vectors = encode_texts(client_texts)

    cache = {
        "properties": {p.id: vec for p, vec in zip(properties, property_vectors)},
        "clients": {c.id: vec for c, vec in zip(clients, client_vectors)},
    }
    save_embeddings_cache(CACHE_PATH, cache, DEFAULT_MODEL_NAME)
    dim = len(property_vectors[0]) if property_vectors else 0
    print(f"Cache (completos) guardada en {CACHE_PATH} (dim={dim}).")

    if CLIENTS_DEGRADED_PATH.exists():
        degraded_data = json.loads(CLIENTS_DEGRADED_PATH.read_text(encoding="utf-8"))
        degraded_clients = [_build_client_from_raw(c) for c in degraded_data]
        print(f"Codificando {len(degraded_clients)} clientes degradados...")
        degraded_texts = [build_client_text(c) for c in degraded_clients]
        degraded_vectors = encode_texts(degraded_texts)
        cache_degraded = {
            "properties": cache["properties"],
            "clients": {c.id: vec for c, vec in zip(degraded_clients, degraded_vectors)},
        }
        save_embeddings_cache(CACHE_DEGRADED_PATH, cache_degraded, DEFAULT_MODEL_NAME)
        print(f"Cache (degradados) guardada en {CACHE_DEGRADED_PATH}.")


if __name__ == "__main__":
    main()
