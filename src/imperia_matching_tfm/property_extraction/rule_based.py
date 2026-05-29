from __future__ import annotations

import re
from dataclasses import asdict

from imperia_matching_tfm.models import Operation, Property, PropertyType


def extract_property_from_text(
    text: str,
    *,
    source_id: str,
    source_url: str | None = None,
    title: str | None = None,
    source_mode: str = "manual_text",
) -> dict:
    normalized = normalize_text(text)
    property_title = title or extract_title(text) or source_id
    city, zone, province = extract_location(property_title, normalized)
    tags = extract_tags(normalized)

    property_ = Property(
        id=source_id,
        title=property_title,
        operation=Operation.BUY,
        property_type=extract_property_type(property_title, normalized),
        price=extract_price(normalized),
        city=city,
        zone=zone,
        province=province,
        bedrooms=extract_number_before(normalized, ("habitaciones", "hab")),
        bathrooms=extract_number_before(normalized, ("bano", "banos", "baño", "baños")),
        surface_m2=extract_surface(normalized, "construidos"),
        usable_surface_m2=extract_surface(normalized, "utiles"),
        built_year=extract_built_year(normalized),
        floor=extract_floor(normalized),
        features=tags_to_features(tags),
        description=None,
        metadata={
            "source": source_mode,
            "source_url": source_url,
            "tags": tags,
        },
    )
    return asdict(property_)


def extract_title(text: str) -> str | None:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned and any(keyword in normalize_text(cleaned) for keyword in ("piso", "chalet", "atico", "casa", "estudio")):
            return cleaned
    return None


def extract_location(title: str, normalized: str) -> tuple[str, str | None, str | None]:
    title_and_text = f"{normalize_text(title)} {normalized}"

    city_candidates = [
        "Alicante / Alacant",
        "Alicante",
        "Alacant",
        "Xirivella",
        "València",
        "Valencia",
        "Madrid",
    ]
    city = ""
    for candidate in city_candidates:
        if normalize_text(candidate) in title_and_text:
            cleaned_candidate = normalize_text(candidate)
            if cleaned_candidate in {"valencia", "valencia"}:
                city = "València"
            elif cleaned_candidate in {"alicante alacant", "alicante", "alacant"}:
                city = "Alicante / Alacant"
            else:
                city = candidate
            break

    if any(value in title_and_text for value in ("valencia", "l'horta sud", "horta sud")):
        province = "València"
    elif any(value in title_and_text for value in ("alicante", "alacant", "l'alacanti", "alacanti")):
        province = "Alicante"
    else:
        province = city or None

    zone_candidates = [
        "Vara de Quart",
        "Fontanares",
        "Zona Centro",
        "Carolinas Altas",
        "Campoamor-Carolinas-Altozano",
        "Patraix",
        "Safranar",
        "Tres Forques",
        "La Raiosa",
        "Sant Isidre",
    ]
    for zone in zone_candidates:
        if normalize_text(zone) in title_and_text:
            return city, zone, province
    return city, None, province


def extract_property_type(title: str, normalized: str) -> PropertyType:
    value = f"{normalize_text(title)} {normalized}"
    tokens = set(value.split())
    if "chalet" in tokens or "villa" in tokens:
        return PropertyType.CHALET
    if "atico" in tokens:
        return PropertyType.PENTHOUSE
    if "casa" in tokens:
        return PropertyType.HOUSE
    if "estudio" in tokens:
        return PropertyType.STUDIO
    if "piso" in tokens or "vivienda" in tokens or "bajo" in tokens:
        return PropertyType.FLAT
    return PropertyType.OTHER


def extract_price(normalized: str) -> int:
    match = re.search(r"(\d{1,3}(?:[.\s]\d{3})+|\d{5,7})\s*€", normalized)
    if not match:
        raise ValueError("No se pudo extraer el precio de la propiedad.")
    return int(re.sub(r"\D", "", match.group(1)))


def extract_number_before(normalized: str, labels: tuple[str, ...]) -> int | None:
    joined = "|".join(labels)
    match = re.search(rf"(\d+)\s*(?:{joined})", normalized)
    return int(match.group(1)) if match else None


def extract_surface(normalized: str, qualifier: str) -> int | None:
    pattern = rf"(\d{{2,4}})\s*m[²2]\s*{qualifier}"
    match = re.search(pattern, normalized)
    if match:
        return int(match.group(1))

    if qualifier == "construidos":
        generic = re.search(r"(\d{2,4})\s*m[²2]", normalized)
        if generic:
            return int(generic.group(1))
    return None


def extract_built_year(normalized: str) -> int | None:
    match = re.search(r"construido en\s*(\d{4})", normalized)
    return int(match.group(1)) if match else None


def extract_floor(normalized: str) -> int | None:
    match = re.search(r"planta\s*(\d+)", normalized)
    if match:
        return int(match.group(1))
    if "bajo" in normalized:
        return 0
    return None


def extract_tags(normalized: str) -> list[str]:
    rules = {
        "elevator": ("ascensor",),
        "terrace": ("terraza", "balcon", "balcón"),
        "garage_optional": ("garaje opcional", "posibilidad de garaje"),
        "garage": ("garaje incluido",),
        "exterior": ("exterior",),
        "amplio": ("amplia", "amplio", "amplitud"),
        "bien_ubicado": ("bien ubicada", "bien ubicado"),
        "muy_luminoso": ("muy luminosa", "muy luminoso"),
        "luz_natural": ("luz natural",),
        "buena_distribucion": ("buena distribucion", "bien distribuida", "bien distribuido"),
        "cocina_independiente": ("cocina independiente",),
        "galeria": ("galeria", "galería"),
        "salon_comedor": ("salon-comedor", "salón-comedor", "salon comedor", "salon"),
        "buen_estado": ("buen estado",),
        "segunda_mano": ("segunda mano",),
        "sin_calefaccion": ("no dispone de calefaccion", "sin calefaccion"),
        "sin_ascensor": ("sin ascensor", "no dispone de ascensor"),
        "reformado": ("reformado", "reformada", "reforma actual"),
        "listo_entrar_vivir": ("listo para entrar a vivir",),
        "primera_vivienda": ("primera vivienda",),
        "zona_tranquila": ("zona tranquila",),
        "distribucion_funcional": ("distribucion funcional", "distribución funcional"),
        "cocina_abierta": ("cocina abierta", "tipo office"),
        "patio": ("patio", "patios"),
        "orientacion_este": ("orientacion este", "orientación este"),
        "climalit": ("climalit",),
        "transporte_publico": ("transporte publico", "transporte público", "buses", "tranvia", "tranvía"),
        "cerca_playa": ("playa",),
        "potencial_alquiler": ("potencial de alquiler",),
        "familias": ("familias",),
        "buena_ventilacion": ("buena ventilacion", "buena ventilación"),
        "zona_bien_comunicada": ("bien comunicada", "bien comunicado"),
        "servicios_cercanos": ("servicios", "supermercados", "colegios", "transporte"),
        "inversion": ("inversion", "inversión"),
        "residencia_habitual": ("residencia habitual",),
        "fontanares": ("fontanares",),
    }

    tags: list[str] = []
    for tag, keywords in rules.items():
        if any(normalize_text(keyword) in normalized for keyword in keywords):
            tags.append(tag)
    if "sin_ascensor" in tags and "elevator" in tags:
        tags.remove("elevator")
    if "garage" in tags and "garage_optional" in tags:
        tags.remove("garage_optional")
    return tags


def tags_to_features(tags: list[str]) -> list[str]:
    aliases = {
        "garage_optional": "garaje_opcional",
    }
    return [aliases.get(tag, tag) for tag in tags]


def normalize_text(text: str) -> str:
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "à": "a",
        "è": "e",
        "ì": "i",
        "ò": "o",
        "ù": "u",
        "ñ": "n",
    }
    normalized = text.lower()
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())
