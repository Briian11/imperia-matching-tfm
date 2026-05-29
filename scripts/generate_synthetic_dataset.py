"""Genera un dataset sintetico ampliado de clientes, propiedades y etiquetas.

Uso:
    python3 scripts/generate_synthetic_dataset.py

Salida:
    data/synthetic/clients.json
    data/synthetic/properties.json
    data/labels/relevance_synthetic.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imperia_matching_tfm.matching.baseline import score_property_for_client
from imperia_matching_tfm.models import Client, Property

SEED = 42
DEGRADED_DROP_RATE = 0.5  # probabilidad de borrar cada campo estructurado en version degradada

# ---------------------------------------------------------------------------
# Datos de referencia para generar variedad realista
# ---------------------------------------------------------------------------

LOCATIONS = [
    {"city": "Madrid", "province": "Madrid", "zones": ["Chamberi", "Salamanca", "Retiro", "Malasana", "Tetuan", "Chamartin", "Moncloa", "Lavapies", "Chueca", "Arganzuela"]},
    {"city": "Barcelona", "province": "Barcelona", "zones": ["Eixample", "Gracia", "Sarria", "Born", "Poble Nou", "Sant Andreu", "Les Corts", "Sants"]},
    {"city": "Valencia", "province": "Valencia", "zones": ["La Eliana", "Ruzafa", "Benimaclet", "Campanar", "Patraix", "Mestalla", "El Cabanyal"]},
    {"city": "Sevilla", "province": "Sevilla", "zones": ["Triana", "Nervion", "Santa Cruz", "Macarena", "Los Remedios"]},
    {"city": "Malaga", "province": "Malaga", "zones": ["Centro", "El Palo", "Teatinos", "Pedregalejo", "La Malagueta"]},
    {"city": "Alicante", "province": "Alicante", "zones": ["Playa San Juan", "Centro", "Cabo Huertas", "San Blas"]},
    {"city": "Bilbao", "province": "Vizcaya", "zones": ["Casco Viejo", "Deusto", "Indautxu", "Abando"]},
]

BUY_PRICE_RANGES = [
    (80_000, 150_000),
    (120_000, 220_000),
    (180_000, 300_000),
    (250_000, 400_000),
    (350_000, 550_000),
    (500_000, 800_000),
]

RENT_PRICE_RANGES = [
    (500, 800),
    (700, 1_100),
    (900, 1_400),
    (1_200, 1_800),
    (1_500, 2_500),
]

PROPERTY_TYPES_BUY = ["FLAT", "HOUSE", "CHALET", "PENTHOUSE"]
PROPERTY_TYPES_RENT = ["FLAT", "STUDIO", "PENTHOUSE"]

FEATURES_POOL = [
    "terrace", "garage", "elevator", "pool", "garden",
    "furnished", "air_conditioning", "storage", "concierge", "gym",
]

FEATURE_ALIASES_ES = {
    "terrace": "terraza", "garage": "garaje", "elevator": "ascensor",
    "pool": "piscina", "garden": "jardin", "furnished": "amueblado",
    "air_conditioning": "aire acondicionado", "storage": "trastero",
    "concierge": "portero", "gym": "gimnasio",
}

CLIENT_NAMES = [
    "Ana Garcia", "Carlos Lopez", "Maria Fernandez", "Pedro Martinez",
    "Laura Sanchez", "Jorge Ruiz", "Elena Diaz", "Miguel Torres",
    "Sofia Moreno", "David Jimenez", "Carmen Alvarez", "Raul Romero",
    "Isabel Navarro", "Pablo Gil", "Lucia Molina", "Andres Serrano",
    "Marta Dominguez", "Fernando Gutierrez", "Natalia Munoz", "Alberto Ortega",
    "Patricia Castillo", "Ricardo Santos", "Beatriz Herrera", "Javier Medina",
    "Rosa Iglesias", "Daniel Cortes", "Cristina Garrido", "Hugo Delgado",
    "Eva Fuentes", "Adrian Vega", "Sara Campos", "Oscar Reyes",
    "Teresa Aguilar", "Manuel Blanco", "Irene Caballero", "Sergio Leon",
]

PROPERTY_TITLES_BUY = [
    "Piso luminoso reformado", "Atico con terraza panoramica",
    "Casa adosada familiar", "Chalet independiente con jardin",
    "Piso de obra nueva", "Vivienda centrica con garaje",
    "Apartamento exterior soleado", "Duplex con acabados de calidad",
    "Piso con vistas al parque", "Casa con piscina comunitaria",
]

PROPERTY_TITLES_RENT = [
    "Piso amueblado en alquiler", "Estudio centrico recien reformado",
    "Apartamento temporal equipado", "Atico en alquiler con terraza",
    "Piso de alquiler cerca del metro",
]

# ---------------------------------------------------------------------------
# Generadores
# ---------------------------------------------------------------------------

def _pick_features(rng: random.Random, count_min: int = 0, count_max: int = 4) -> list[str]:
    n = rng.randint(count_min, count_max)
    return rng.sample(FEATURES_POOL, min(n, len(FEATURES_POOL)))


def _generate_properties(rng: random.Random, count: int = 100) -> list[dict]:
    properties = []
    for i in range(count):
        loc = rng.choice(LOCATIONS)
        is_rent = rng.random() < 0.25
        operation = "RENT" if is_rent else "BUY"

        if is_rent:
            ptype = rng.choice(PROPERTY_TYPES_RENT)
            pmin, pmax = rng.choice(RENT_PRICE_RANGES)
            title_base = rng.choice(PROPERTY_TITLES_RENT)
        else:
            ptype = rng.choice(PROPERTY_TYPES_BUY)
            pmin, pmax = rng.choice(BUY_PRICE_RANGES)
            title_base = rng.choice(PROPERTY_TITLES_BUY)

        price = rng.randint(pmin, pmax)
        zone = rng.choice(loc["zones"])
        bedrooms = rng.randint(0 if ptype == "STUDIO" else 1, 5)
        bathrooms = rng.randint(1, max(1, bedrooms - 1) + 1)
        surface = rng.randint(30 if ptype == "STUDIO" else 50, 300)
        features = _pick_features(rng, 0, 5)
        built_year = rng.choice([None, rng.randint(1970, 2024)])
        floor = rng.randint(0, 8) if ptype in ("FLAT", "PENTHOUSE", "STUDIO") else 0

        prop = {
            "id": f"p_{i+1:03d}",
            "title": f"{title_base} en {zone}",
            "operation": operation,
            "property_type": ptype,
            "price": price,
            "city": loc["city"],
            "zone": zone,
            "province": loc["province"],
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "surface_m2": surface,
            "built_year": built_year,
            "floor": floor,
            "features": features,
            "description": f"{title_base} en {zone}, {loc['city']}. {surface} m2, {bedrooms} hab.",
        }
        properties.append(prop)
    return properties


def _generate_clients(rng: random.Random, count: int = 35) -> list[dict]:
    clients = []
    for i in range(count):
        loc = rng.choice(LOCATIONS)
        is_rent = rng.random() < 0.25
        operation = "RENT" if is_rent else "BUY"

        if is_rent:
            ptypes = rng.sample(PROPERTY_TYPES_RENT, k=rng.randint(1, 2))
            pmin, pmax = rng.choice(RENT_PRICE_RANGES)
        else:
            ptypes = rng.sample(PROPERTY_TYPES_BUY, k=rng.randint(1, 2))
            pmin, pmax = rng.choice(BUY_PRICE_RANGES)

        zones = rng.sample(loc["zones"], k=rng.randint(1, min(3, len(loc["zones"]))))
        features_en = _pick_features(rng, 0, 3)
        features_es = [FEATURE_ALIASES_ES.get(f, f) for f in features_en]

        bedrooms_min = rng.choice([None, rng.randint(1, 4)])
        bathrooms_min = rng.choice([None, rng.randint(1, 2)])
        surface_min = rng.choice([None, rng.randint(40, 150)])

        price_max = pmax
        price_min = rng.choice([None, pmin]) if not is_rent else None

        name = CLIENT_NAMES[i % len(CLIENT_NAMES)]
        if i >= len(CLIENT_NAMES):
            name = f"{name} {i // len(CLIENT_NAMES) + 1}"

        notes = _build_natural_notes(
            rng, operation, ptypes, zones, loc["city"],
            price_max, bedrooms_min, surface_min, features_es,
        )

        client = {
            "id": f"c_{i+1:03d}",
            "name": name,
            "preference": {
                "operation": operation,
                "property_types": ptypes,
                "zones": zones,
                "cities": [loc["city"]],
                "province": loc["province"],
                "price_min": price_min,
                "price_max": price_max,
                "bedrooms_min": bedrooms_min,
                "bathrooms_min": bathrooms_min,
                "surface_min_m2": surface_min,
                "features": features_es,
                "notes": notes,
            },
        }
        clients.append(client)
    return clients


NOTE_TEMPLATES_BUY = [
    "Estoy buscando comprar un {tipo} en {zona}, {ciudad}. Presupuesto maximo de {precio} euros. {extras}",
    "Quiero compra de {tipo} por la zona de {zona} en {ciudad}. Hasta {precio}. {extras}",
    "Busco {tipo} para comprar en {ciudad}, preferiblemente {zona}. Maximo {precio} euros. {extras}",
    "Interesado en comprar {tipo} en {zona}, {ciudad}. Presupuesto hasta {precio}. {extras}",
    "Necesito un {tipo} en venta en {zona} ({ciudad}). No mas de {precio} euros. {extras}",
]

NOTE_TEMPLATES_RENT = [
    "Busco alquilar un {tipo} en {zona}, {ciudad}. Presupuesto maximo {precio} euros al mes. {extras}",
    "Necesito alquiler de {tipo} en {ciudad}, zona {zona}. Hasta {precio}/mes. {extras}",
    "Quiero alquilar {tipo} por {zona} en {ciudad}. Maximo {precio} euros mensuales. {extras}",
]

TYPE_NAME_MAP = {
    "FLAT": "piso", "HOUSE": "casa", "CHALET": "chalet",
    "PENTHOUSE": "atico", "STUDIO": "estudio",
}


def _build_natural_notes(
    rng, operation, ptypes, zones, city, price_max,
    bedrooms_min, surface_min, features_es,
):
    tipo = TYPE_NAME_MAP.get(ptypes[0], "piso")
    zona = zones[0] if zones else city

    extras_parts = []
    if bedrooms_min:
        extras_parts.append(rng.choice([
            f"Minimo {bedrooms_min} habitaciones",
            f"Al menos {bedrooms_min} dormitorios",
            f"Necesito {bedrooms_min} habitaciones como minimo",
        ]))
    if surface_min:
        extras_parts.append(rng.choice([
            f"Al menos {surface_min} m2",
            f"Minimo {surface_min} metros",
            f"Necesito {surface_min} m2 o mas",
        ]))
    if features_es:
        extras_parts.append(rng.choice([
            f"Con {', '.join(features_es)}",
            f"Importante que tenga {', '.join(features_es)}",
            f"Necesito {', '.join(features_es)}",
        ]))

    extras = ". ".join(extras_parts) + "." if extras_parts else ""

    if operation == "RENT":
        template = rng.choice(NOTE_TEMPLATES_RENT)
    else:
        template = rng.choice(NOTE_TEMPLATES_BUY)

    return template.format(
        tipo=tipo, zona=zona, ciudad=city,
        precio=price_max, extras=extras,
    ).strip()


def _generate_degraded_clients(clients: list[dict], rng: random.Random) -> list[dict]:
    """Genera versiones degradadas de clientes eliminando campos estructurados.

    Los campos eliminados siguen presentes en el texto de notes,
    lo que permite medir si el enriquecimiento los recupera.
    """
    degraded = []
    droppable_fields = [
        "property_types", "price_max", "bedrooms_min",
        "bathrooms_min", "surface_min_m2", "features",
    ]

    for client in clients:
        pref = dict(client["preference"])
        dropped = []
        for field in droppable_fields:
            if pref.get(field) and rng.random() < DEGRADED_DROP_RATE:
                if isinstance(pref[field], list):
                    pref[field] = []
                else:
                    pref[field] = None
                dropped.append(field)

        degraded.append({
            "id": client["id"],
            "name": client["name"],
            "preference": pref,
            "dropped_fields": dropped,
        })
    return degraded


def _generate_labels(
    clients_data: list[dict],
    properties_data: list[dict],
) -> list[dict]:
    """Genera etiquetas usando el baseline y anade ruido controlado.

    Estrategia:
    - Ejecuta el scoring baseline para cada par.
    - Convierte el score (0-100) a relevancia (0-3) con umbrales.
    - Anade ruido en ~10% de pares para simular discrepancias humanas.
    """
    rng = random.Random(SEED + 1)
    clients = [Client.from_dict(c) for c in clients_data]
    properties = [Property.from_dict(p) for p in properties_data]

    labels = []
    for client in clients:
        for prop in properties:
            ms = score_property_for_client(client, prop)
            score = ms.score

            if score >= 70:
                rel = 3
            elif score >= 50:
                rel = 2
            elif score >= 30:
                rel = 1
            else:
                rel = 0

            # Ruido: ~10% de las veces, alterar la etiqueta en +-1
            if rng.random() < 0.10:
                delta = rng.choice([-1, 1])
                rel = max(0, min(3, rel + delta))

            # Solo guardar pares con relevancia > 0 o una muestra de los 0
            if rel > 0 or rng.random() < 0.15:
                labels.append({
                    "client_id": client.id,
                    "property_id": prop.id,
                    "relevance": rel,
                })

    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)

    print("Generando propiedades...")
    properties = _generate_properties(rng, count=100)

    print("Generando clientes...")
    clients = _generate_clients(rng, count=35)

    print("Generando clientes degradados (para M2)...")
    degraded_rng = random.Random(SEED + 2)
    degraded_clients = _generate_degraded_clients(clients, degraded_rng)

    print("Generando etiquetas...")
    labels = _generate_labels(clients, properties)

    # Guardar
    out_props = ROOT / "data/synthetic/properties.json"
    out_clients = ROOT / "data/synthetic/clients.json"
    out_degraded = ROOT / "data/synthetic/clients_degraded.json"
    out_labels = ROOT / "data/labels/relevance_synthetic.json"

    out_props.write_text(json.dumps(properties, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_clients.write_text(json.dumps(clients, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_degraded.write_text(json.dumps(degraded_clients, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_labels.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n  Propiedades:          {len(properties)} -> {out_props}")
    print(f"  Clientes:             {len(clients)} -> {out_clients}")
    print(f"  Clientes degradados:  {len(degraded_clients)} -> {out_degraded}")
    print(f"  Etiquetas:            {len(labels)} -> {out_labels}")

    # Estadisticas de degradacion
    total_dropped = sum(len(c.get("dropped_fields", [])) for c in degraded_clients)
    clients_with_drops = sum(1 for c in degraded_clients if c.get("dropped_fields"))
    print(f"\n  Clientes con campos eliminados: {clients_with_drops}/{len(degraded_clients)}")
    print(f"  Total campos eliminados: {total_dropped}")

    # Estadisticas rapidas
    rel_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for label in labels:
        rel_counts[label["relevance"]] += 1
    print(f"\n  Distribucion de relevancia: {dict(rel_counts)}")
    print("\nDataset generado correctamente.")


if __name__ == "__main__":
    main()
