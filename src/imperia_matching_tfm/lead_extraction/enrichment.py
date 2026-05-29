"""Enriquece preferencias estructuradas con datos extraidos desde texto de leads.

Estrategia: los campos estructurados existentes tienen prioridad.
Solo se rellenan campos vacios o nulos con la informacion extraida del texto.
"""

from __future__ import annotations

from imperia_matching_tfm.lead_extraction.rule_based import extract_preferences_rule_based
from imperia_matching_tfm.models import BuyerPreference


def enrich_preference(preference: BuyerPreference) -> BuyerPreference:
    """Devuelve una copia enriquecida de la preferencia usando el texto de notes."""
    if not preference.notes:
        return preference

    extracted = extract_preferences_rule_based(preference.notes)

    return BuyerPreference(
        operation=preference.operation or extracted.operation,
        property_types=preference.property_types or extracted.property_types,
        zones=preference.zones,
        cities=preference.cities,
        province=preference.province,
        price_min=preference.price_min,
        price_max=preference.price_max or extracted.price_max,
        bedrooms_min=preference.bedrooms_min or extracted.bedrooms_min,
        bathrooms_min=preference.bathrooms_min,
        surface_min_m2=preference.surface_min_m2 or extracted.surface_min_m2,
        built_year_min=preference.built_year_min,
        floor_min=preference.floor_min,
        features=_merge_features(preference.features, extracted.features),
        notes=preference.notes,
    )


def _merge_features(existing: list[str], extracted: list[str]) -> list[str]:
    """Combina features existentes con extraidas, sin duplicados."""
    if not extracted:
        return existing
    combined = list(existing)
    existing_lower = {f.lower() for f in existing}
    for feat in extracted:
        if feat.lower() not in existing_lower:
            combined.append(feat)
    return combined
