"""M5: RAG aplicado a matching inmobiliario.

Flujo:
  1. Retrieval (M4): para cada cliente, recupera top-K propiedades
     por scoring hibrido estructurado + semantico.
  2. Augmented Generation (Claude): el LLM razona sobre los candidatos
     y genera una explicacion en lenguaje natural del mejor match,
     incluyendo preferencias implicitas que el scoring estructurado no captura.

Modos de ejecucion:
  - Con ANTHROPIC_API_KEY (o --api-key): llama a Claude y guarda resultados.
  - Sin key: carga resultados pre-computados de reports/results/m5_rag_results.json.

Uso:
    PYTHONPATH=src python3 scripts/run_m5_rag.py --api-key sk-ant-...
    PYTHONPATH=src python3 scripts/run_m5_rag.py  # modo offline con cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imperia_matching_tfm.matching.hybrid_m4 import score_property_for_client_m4  # noqa: E402
from imperia_matching_tfm.matching.semantic import load_embeddings_cache  # noqa: E402
from imperia_matching_tfm.models import Client, Property  # noqa: E402

CACHE_PATH = ROOT / "data/processed/embeddings_cache.json"
OUTPUT_PATH = ROOT / "reports/results/m5_rag_results.json"
MODEL = "claude-haiku-4-5-20251001"
RETRIEVAL_K = 5


def build_rag_prompt(client: Client, candidates: list[Property]) -> str:
    pref = client.preference
    lines = [
        "Eres un agente inmobiliario experto. Analiza el perfil del cliente y las propiedades candidatas.",
        "",
        "## Perfil del cliente",
        f"Nombre: {client.name}",
        f"Operacion: {pref.operation.value if pref.operation else 'no especificada'}",
        f"Tipos buscados: {', '.join(pt.value for pt in pref.property_types) if pref.property_types else 'sin preferencia'}",
        f"Zonas: {', '.join(pref.zones) if pref.zones else '-'}",
        f"Ciudades: {', '.join(pref.cities) if pref.cities else '-'}",
        f"Presupuesto: {pref.price_min or 'sin minimo'} - {pref.price_max or 'sin maximo'} EUR",
        f"Habitaciones minimas: {pref.bedrooms_min or 'sin preferencia'}",
        f"Superficie minima: {pref.surface_min_m2 or 'sin preferencia'} m2",
        f"Caracteristicas deseadas: {', '.join(pref.features) if pref.features else 'sin especificar'}",
        f"Notas del cliente: {pref.notes or 'sin notas'}",
        "",
        "## Propiedades candidatas (ya filtradas por scoring estructurado + semantico)",
    ]
    for i, prop in enumerate(candidates, 1):
        lines += [
            f"\n### Propiedad {i}: {prop.id}",
            f"Titulo: {prop.title}",
            f"Tipo: {prop.property_type.value} | Operacion: {prop.operation.value}",
            f"Precio: {prop.price} EUR",
            f"Ubicacion: {prop.zone or '-'}, {prop.city}, {prop.province or '-'}",
            f"Habitaciones: {prop.bedrooms or 'N/D'} | Banos: {prop.bathrooms or 'N/D'} | Superficie: {prop.surface_m2 or 'N/D'} m2",
            f"Caracteristicas: {', '.join(prop.features) if prop.features else 'ninguna'}",
            f"Descripcion: {prop.description or 'sin descripcion'}",
        ]
    lines += [
        "",
        "## Tu tarea",
        "1. Indica cual es la MEJOR propiedad para este cliente y por que.",
        "2. Destaca al menos una preferencia IMPLICITA del cliente (algo que se intuye de sus notas pero no esta en los campos estructurados).",
        "3. Si hay algun punto debil del mejor match, mencionalo brevemente.",
        "4. Responde en JSON con este formato exacto:",
        '{"mejor_match": "p_XXX", "razonamiento": "...", "preferencia_implicita": "...", "punto_debil": "..."}',
    ]
    return "\n".join(lines)


def call_claude(prompt: str, api_key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end]) if start != -1 else {"razonamiento": text}


def run_live(clients, properties, cache, api_key, top_k, limit) -> list[dict]:
    results = []
    target_clients = clients[:limit] if limit else clients
    for i, client in enumerate(target_clients, 1):
        print(f"  [{i}/{len(target_clients)}] {client.name}...", end=" ", flush=True)
        scored = sorted(
            (score_property_for_client_m4(client, p, cache) for p in properties),
            key=lambda s: s.score,
            reverse=True,
        )
        candidate_ids = [s.property_id for s in scored[:top_k]]
        candidates = [p for pid in candidate_ids for p in properties if p.id == pid]
        prompt = build_rag_prompt(client, candidates)
        try:
            response = call_claude(prompt, api_key)
            print("ok")
        except Exception as e:
            print(f"error: {e}")
            response = {"error": str(e)}
        results.append({
            "client_id": client.id,
            "client_name": client.name,
            "retrieval_top_k": [
                {"property_id": s.property_id, "score_m4": s.score}
                for s in scored[:top_k]
            ],
            "llm_response": response,
        })
    return results


def display_results(results: list[dict]) -> None:
    for item in results:
        print(f"\n{'='*70}")
        print(f"Cliente: {item['client_name']} ({item['client_id']})")
        print(f"Top {len(item['retrieval_top_k'])} por M4:")
        for r in item["retrieval_top_k"]:
            print(f"  {r['property_id']}  score={r['score_m4']:.1f}")
        llm = item.get("llm_response", {})
        if "error" in llm:
            print(f"  Error LLM: {llm['error']}")
        else:
            print(f"\nMejor match segun Claude: {llm.get('mejor_match', '?')}")
            print(f"Razonamiento: {llm.get('razonamiento', '-')}")
            print(f"Preferencia implicita detectada: {llm.get('preferencia_implicita', '-')}")
            print(f"Punto debil: {llm.get('punto_debil', '-')}")


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        cached = OUTPUT_PATH
        if cached.exists():
            print(f"Sin API key. Cargando resultados pre-computados de {cached}")
            results = json.loads(cached.read_text(encoding="utf-8"))
            display_results(results)
        else:
            print("Sin API key y sin resultados pre-computados.")
            print("Ejecuta con: ANTHROPIC_API_KEY=sk-ant-... python3 scripts/run_m5_rag.py")
        return

    clients = [Client.from_dict(c) for c in json.loads((ROOT / args.clients).read_text(encoding="utf-8"))]
    properties = [Property.from_dict(p) for p in json.loads((ROOT / args.properties).read_text(encoding="utf-8"))]
    cache = load_embeddings_cache(ROOT / args.embeddings)

    print(f"M5 RAG — modelo: {MODEL}")
    print(f"Retrieval: M4 top-{args.top_k} | Clientes: {args.limit or len(clients)}")
    print("-" * 70)

    results = run_live(clients, properties, cache, api_key, args.top_k, args.limit)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados guardados en {OUTPUT_PATH}")
    display_results(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M5 RAG: retrieval M4 + razonamiento Claude.")
    parser.add_argument("--api-key", help="Anthropic API key (o usar ANTHROPIC_API_KEY).")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_K, help="Candidatas por cliente para el LLM.")
    parser.add_argument("--limit", type=int, default=None, help="Limitar numero de clientes (para pruebas).")
    parser.add_argument("--clients", default="data/synthetic/clients.json")
    parser.add_argument("--properties", default="data/synthetic/properties.json")
    parser.add_argument("--embeddings", default="data/processed/embeddings_cache.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
