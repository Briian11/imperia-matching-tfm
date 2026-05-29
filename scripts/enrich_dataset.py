"""Enriquece el dataset con descripciones realistas y propiedades nuevas.

Usa Claude para:
  1. Generar descripciones estilo Idealista para las 100 propiedades existentes.
  2. Generar 20 propiedades nuevas muy detalladas que parezcan listings reales.

Uso:
    ANTHROPIC_API_KEY=sk-ant-... PYTHONPATH=src python3 scripts/enrich_dataset.py

Salida:
    data/synthetic/properties.json        (actualizado con descripciones)
    data/synthetic/properties_extra.json  (20 propiedades nuevas)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anthropic

PROPERTIES_PATH = ROOT / "data/synthetic/properties.json"
EXTRA_PATH = ROOT / "data/synthetic/properties_extra.json"
MODEL = "claude-haiku-4-5-20251001"


def build_description_prompt(prop: dict) -> str:
    features_str = ", ".join(prop.get("features", [])) or "sin extras destacados"
    return f"""Eres un redactor de anuncios inmobiliarios para Idealista.es.
Escribe una descripcion realista de 3-4 frases para este inmueble.
El tono debe ser profesional pero cercano, como un anuncio real de Idealista.
Incluye detalles concretos sobre la distribucion, la luz, el estado y el entorno.
NO incluyas el precio ni el titulo. Solo la descripcion del inmueble.

Datos del inmueble:
- Tipo: {prop['property_type']}
- Operacion: {prop['operation']}
- Ciudad: {prop['city']} | Zona: {prop.get('zone', '')}
- Superficie: {prop.get('surface_m2', '?')} m2
- Habitaciones: {prop.get('bedrooms', '?')} | Banos: {prop.get('bathrooms', '?')}
- Planta: {prop.get('floor', 0)}
- Extras: {features_str}
- Ano construccion: {prop.get('built_year') or 'no indicado'}

Responde SOLO con la descripcion, sin encabezados ni formato extra."""


def build_new_properties_prompt() -> str:
    return """Genera exactamente 20 propiedades inmobiliarias realistas para el mercado espanol.
Deben parecer listings reales de Idealista con datos detallados y variados.

Usa estas ciudades y zonas reales:
- Madrid: Chamberi, Salamanca, Retiro, Malasana, Lavapies, Chamartin
- Barcelona: Eixample, Gracia, Born, Poble Nou, Sarria
- Valencia: Ruzafa, Benimaclet, Campanar, El Cabanyal, Patraix
- Sevilla: Triana, Nervion, Santa Cruz, Macarena
- Malaga: Centro, Teatinos, Pedregalejo, La Malagueta

Tipos de operacion: 80% BUY, 20% RENT
Tipos de inmueble: FLAT, HOUSE, CHALET, PENTHOUSE, STUDIO

Responde con un array JSON con exactamente 20 objetos. Cada objeto debe tener:
{
  "id": "p_extra_001",  (incrementar hasta p_extra_020)
  "title": "titulo realista estilo Idealista",
  "operation": "BUY" o "RENT",
  "property_type": "FLAT|HOUSE|CHALET|PENTHOUSE|STUDIO",
  "price": precio en euros (entero, realista para la zona),
  "city": "ciudad",
  "zone": "barrio o zona",
  "province": "provincia",
  "bedrooms": numero (entero),
  "bathrooms": numero (entero),
  "surface_m2": superficie util en m2 (entero),
  "built_year": ano de construccion (entero, entre 1950 y 2024, o null),
  "floor": planta (entero, 0 para chalets/casas),
  "features": lista de strings de: terrace, garage, elevator, pool, garden, furnished, air_conditioning, storage, concierge, gym,
  "description": "descripcion realista de 3-4 frases estilo Idealista"
}

Precios orientativos:
- Madrid Salamanca/Chamberi BUY: 400.000-900.000€
- Barcelona Eixample BUY: 350.000-700.000€
- Valencia Ruzafa BUY: 180.000-350.000€
- Sevilla Triana BUY: 150.000-280.000€
- Malaga Centro BUY: 200.000-400.000€
- Alquiler cualquier ciudad: 700-2.500€/mes

Responde SOLO con el array JSON valido, sin markdown ni explicaciones."""


def enrich_descriptions(client: anthropic.Anthropic, properties: list[dict]) -> list[dict]:
    enriched = []
    for i, prop in enumerate(properties, 1):
        print(f"  [{i}/{len(properties)}] {prop['id']} {prop['title'][:40]}...", end=" ", flush=True)
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": build_description_prompt(prop)}],
            )
            prop["description"] = msg.content[0].text.strip()
            print("ok")
        except Exception as e:
            print(f"error: {e}")
        enriched.append(prop)
        if i % 10 == 0:
            time.sleep(1)
    return enriched


def generate_extra_properties(client: anthropic.Anthropic) -> list[dict]:
    print("  Generando 20 propiedades nuevas...", end=" ", flush=True)
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": build_new_properties_prompt()}],
        )
        text = msg.content[0].text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        properties = json.loads(text[start:end])
        print(f"ok ({len(properties)} propiedades)")
        return properties
    except Exception as e:
        print(f"error: {e}")
        return []


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Falta ANTHROPIC_API_KEY.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    properties = json.loads(PROPERTIES_PATH.read_text(encoding="utf-8"))

    print(f"Enriqueciendo descripciones de {len(properties)} propiedades...")
    properties = enrich_descriptions(client, properties)
    PROPERTIES_PATH.write_text(json.dumps(properties, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Guardado en {PROPERTIES_PATH}")

    print("\nGenerando propiedades extra realistas...")
    extra = generate_extra_properties(client)
    if extra:
        EXTRA_PATH.write_text(json.dumps(extra, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Guardado en {EXTRA_PATH}")

    print("\nDataset enriquecido correctamente.")
    print(f"  Propiedades con descripcion: {sum(1 for p in properties if p.get('description'))}/{len(properties)}")
    print(f"  Propiedades extra generadas: {len(extra)}")


if __name__ == "__main__":
    main()
