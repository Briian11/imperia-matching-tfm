from __future__ import annotations

import re

from imperia_matching_tfm.models import BuyerPreference, Operation, PropertyType


TYPE_KEYWORDS = {
    PropertyType.FLAT: ("piso", "apartamento"),
    PropertyType.HOUSE: ("casa",),
    PropertyType.CHALET: ("chalet", "villa"),
    PropertyType.PENTHOUSE: ("atico", "penthouse"),
    PropertyType.STUDIO: ("estudio",),
}


def extract_preferences_rule_based(text: str) -> BuyerPreference:
    normalized = _normalize(text)
    operation = _extract_operation(normalized)
    property_types = [
        property_type
        for property_type, keywords in TYPE_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    ]

    price_max = _extract_price_max(normalized)
    bedrooms_min = _extract_minimum(normalized, ("habitaciones", "dormitorios", "hab"))
    surface_min = _extract_surface_min(normalized)
    features = _extract_features(normalized)

    return BuyerPreference(
        operation=operation,
        property_types=property_types,
        price_max=price_max,
        bedrooms_min=bedrooms_min,
        surface_min_m2=surface_min,
        features=features,
        notes=text,
    )


def _extract_operation(text: str) -> Operation | None:
    if any(token in text for token in ("comprar", "compra", "venta")):
        return Operation.BUY
    if any(token in text for token in ("alquilar", "alquiler", "renta")):
        return Operation.RENT
    return None


def _extract_price_max(text: str) -> int | None:
    match = re.search(r"(?:hasta|maximo|presupuesto)\s*(?:de)?\s*(\d{2,4})(?:\s?\.?\s?000|k)?", text)
    if not match:
        return None
    value = int(match.group(1))
    return value * 1000 if value < 10000 else value


def _extract_minimum(text: str, labels: tuple[str, ...]) -> int | None:
    joined = "|".join(labels)
    patterns = [
        rf"(\d+)\s*(?:{joined})",
        rf"(?:minimo|al menos)\s*(\d+)\s*(?:{joined})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _extract_surface_min(text: str) -> int | None:
    match = re.search(r"(\d{2,4})\s*(?:m2|metros)", text)
    return int(match.group(1)) if match else None


def _extract_features(text: str) -> list[str]:
    candidates = ["terraza", "piscina", "garaje", "parking", "ascensor", "jardin", "amueblado"]
    return [feature for feature in candidates if feature in text]


def _normalize(text: str) -> str:
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
    }
    normalized = text.lower()
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized

