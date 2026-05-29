from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


TYPE_ALIASES = {
    "piso": "FLAT",
    "flat": "FLAT",
    "atico": "PENTHOUSE",
    "ático": "PENTHOUSE",
    "duplex": "PENTHOUSE",
    "dúplex": "PENTHOUSE",
    "casa": "HOUSE",
    "casa de pueblo": "HOUSE",
    "chalet": "CHALET",
    "villa": "CHALET",
    "local": "COMMERCIAL",
    "oficina": "COMMERCIAL",
    "terreno": "LAND",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa propiedades desde un CSV preparado para el TFM.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data/raw/idealista_magnus_template.csv"),
        help="Ruta del CSV de entrada.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data/processed/properties_from_idealista.json"),
        help="Ruta del JSON normalizado de salida.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    properties = import_csv(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(properties, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Propiedades importadas: {len(properties)}")
    print(f"Salida: {output_path}")


def import_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [normalize_row(row, index) for index, row in enumerate(reader, start=1) if has_content(row)]


def normalize_row(row: dict[str, str], index: int) -> dict:
    source_id = clean(row.get("source_id")) or f"idealista_{index:03d}"
    tags = split_list(row.get("tags"))
    features = merge_features(extract_features(row), tags)

    return {
        "id": source_id,
        "title": clean(row.get("title")) or source_id,
        "operation": normalize_operation(row.get("operation")),
        "property_type": normalize_property_type(row.get("property_type"), row.get("title")),
        "price": parse_int(row.get("price_eur")),
        "city": clean(row.get("city")),
        "zone": clean(row.get("zone")) or None,
        "province": clean(row.get("province")) or None,
        "bedrooms": parse_optional_int(row.get("bedrooms")),
        "bathrooms": parse_optional_int(row.get("bathrooms")),
        "surface_m2": parse_optional_int(row.get("surface_m2")),
        "usable_surface_m2": parse_optional_int(row.get("usable_surface_m2")),
        "built_year": parse_optional_int(row.get("built_year")),
        "floor": parse_floor(row.get("floor")),
        "features": features,
        "description": None,
        "metadata": {
            "source": "idealista",
            "source_url": clean(row.get("source_url")) or None,
            "tags": tags,
        },
    }


def extract_features(row: dict[str, str]) -> list[str]:
    features: list[str] = []
    mapping = {
        "has_elevator": "elevator",
        "has_garage": "garage",
        "has_pool": "pool",
        "has_garden": "garden",
        "has_terrace": "terrace",
    }
    for column, feature in mapping.items():
        if parse_bool(row.get(column)):
            features.append(feature)
    return features


def merge_features(features: list[str], tags: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in [*features, *tags]:
        feature = normalize_feature(value)
        if feature and feature not in seen:
            seen.add(feature)
            normalized.append(feature)
    return normalized


def normalize_feature(value: str) -> str:
    aliases = {
        "ascensor": "elevator",
        "garaje": "garage",
        "parking": "garage",
        "piscina": "pool",
        "jardin": "garden",
        "jardín": "garden",
        "terraza": "terrace",
        "balcon": "terrace",
        "balcón": "terrace",
    }
    cleaned = clean(value).lower()
    return aliases.get(cleaned, cleaned)


def normalize_operation(value: str | None) -> str:
    normalized = clean(value).lower()
    if normalized in {"rent", "alquiler", "alquilar"}:
        return "RENT"
    return "BUY"


def normalize_property_type(value: str | None, title: str | None = None) -> str:
    candidates = [clean(value).lower(), clean(title).lower()]
    for candidate in candidates:
        for alias, property_type in TYPE_ALIASES.items():
            if alias in candidate:
                return property_type
    return "OTHER"


def parse_bool(value: str | None) -> bool:
    normalized = clean(value).lower()
    return normalized in {"1", "true", "yes", "si", "sí", "y", "x"}


def parse_int(value: str | None) -> int:
    parsed = parse_optional_int(value)
    if parsed is None:
        raise ValueError("El campo price_eur es obligatorio y debe ser numerico.")
    return parsed


def parse_optional_int(value: str | None) -> int | None:
    normalized = clean(value)
    if not normalized:
        return None
    digits = "".join(character for character in normalized if character.isdigit())
    return int(digits) if digits else None


def parse_floor(value: str | None) -> int | None:
    normalized = clean(value).lower()
    if not normalized:
        return None
    if "bajo" in normalized:
        return 0
    return parse_optional_int(normalized)


def split_list(value: str | None) -> list[str]:
    normalized = clean(value)
    if not normalized:
        return []
    return [item.strip() for item in normalized.split("|") if item.strip()]


def clean(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def has_content(row: dict[str, str]) -> bool:
    return any(clean(value) for value in row.values())


if __name__ == "__main__":
    main()
