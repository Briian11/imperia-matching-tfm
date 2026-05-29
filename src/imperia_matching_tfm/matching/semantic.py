"""Capa semantica del matching (M4).

Construye representaciones textuales de propiedades y clientes,
calcula embeddings con sentence-transformers y mide afinidad por
similitud coseno.

El embedding se calcula una vez y se cachea en JSON para que la
evaluacion sea reproducible sin volver a cargar el modelo.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from imperia_matching_tfm.models import Client, Property


DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def build_property_text(property_: Property) -> str:
    parts: list[str] = [property_.title or ""]
    parts.append(f"Tipo: {property_.property_type.value.lower()}")
    parts.append(f"Operacion: {property_.operation.value.lower()}")
    if property_.zone:
        parts.append(f"Zona: {property_.zone}")
    if property_.city:
        parts.append(f"Ciudad: {property_.city}")
    if property_.features:
        parts.append("Caracteristicas: " + ", ".join(property_.features))
    if property_.description:
        parts.append(property_.description)
    return " . ".join(p for p in parts if p)


def build_client_text(client: Client) -> str:
    pref = client.preference
    parts: list[str] = []
    if pref.notes:
        parts.append(pref.notes)
    if pref.property_types:
        parts.append("Busca: " + ", ".join(pt.value.lower() for pt in pref.property_types))
    if pref.operation:
        parts.append(f"Operacion: {pref.operation.value.lower()}")
    if pref.zones:
        parts.append("Zonas: " + ", ".join(pref.zones))
    if pref.cities:
        parts.append("Ciudades: " + ", ".join(pref.cities))
    if pref.features:
        parts.append("Preferencias: " + ", ".join(pref.features))
    return " . ".join(p for p in parts if p)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def encode_texts(texts: Iterable[str], model_name: str = DEFAULT_MODEL_NAME) -> list[list[float]]:
    """Carga el modelo y codifica una lista de textos.

    Importacion perezosa de sentence-transformers para que el resto del
    proyecto no requiera la dependencia.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return [vec.tolist() for vec in vectors]


def save_embeddings_cache(path: Path, embeddings: dict[str, dict[str, list[float]]], model_name: str) -> None:
    payload = {
        "model": model_name,
        "dim": len(next(iter(embeddings["properties"].values()))) if embeddings["properties"] else 0,
        "properties": embeddings["properties"],
        "clients": embeddings["clients"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_embeddings_cache(path: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "model": payload.get("model"),
        "properties": payload["properties"],
        "clients": payload["clients"],
    }


def semantic_score(client_id: str, property_id: str, cache: dict[str, dict[str, list[float]]]) -> float:
    client_vec = cache["clients"].get(client_id)
    property_vec = cache["properties"].get(property_id)
    if client_vec is None or property_vec is None:
        return 0.0
    sim = cosine_similarity(client_vec, property_vec)
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))
