from __future__ import annotations

from unicodedata import normalize

from imperia_matching_tfm.models import Client, MatchScore, Property, ScoreCriterion


DEFAULT_WEIGHTS: dict[str, float] = {
    "operation": 20,
    "property_type": 13,
    "location": 20,
    "price": 20,
    "bedrooms": 7,
    "bathrooms": 3,
    "surface": 7,
    "features": 4,
    "built_year": 3,
    "floor": 3,
}

# Alias para compatibilidad con codigo existente
WEIGHTS = DEFAULT_WEIGHTS


def score_property_for_client(
    client: Client,
    property_: Property,
    weights: dict[str, float] | None = None,
) -> MatchScore:
    w = weights or DEFAULT_WEIGHTS
    preference = client.preference
    criteria: list[ScoreCriterion] = []

    operation = _operation_score(preference.operation, property_.operation, w)
    criteria.append(operation)
    if preference.operation and not operation.matched:
        return MatchScore(
            client_id=client.id,
            property_id=property_.id,
            score=0,
            criteria=criteria,
        )

    criteria.extend(
        [
            _property_type_score(preference.property_types, property_.property_type, w),
            _location_score(preference.zones, preference.cities, preference.province, property_, w),
            _price_score(preference.price_min, preference.price_max, property_.price, w),
            _minimum_score("bedrooms", preference.bedrooms_min, property_.bedrooms, w["bedrooms"]),
            _minimum_score("bathrooms", preference.bathrooms_min, property_.bathrooms, w["bathrooms"]),
            _minimum_score("surface", preference.surface_min_m2, property_.surface_m2, w["surface"]),
            _feature_score(preference.features, property_.features, w),
            _minimum_score("built_year", preference.built_year_min, property_.built_year, w["built_year"]),
            _minimum_score("floor", preference.floor_min, property_.floor, w["floor"]),
        ]
    )

    total = round(sum(item.earned for item in criteria), 2)
    return MatchScore(client_id=client.id, property_id=property_.id, score=total, criteria=criteria)


def _operation_score(expected, actual, w: dict[str, float]) -> ScoreCriterion:
    if expected is None:
        return ScoreCriterion(name="operation", earned=w["operation"], maximum=w["operation"], matched=True)
    matched = expected == actual
    return ScoreCriterion(
        name="operation",
        earned=w["operation"] if matched else 0,
        maximum=w["operation"],
        matched=matched,
        client_value=str(expected),
        property_value=str(actual),
        note="Dealbreaker: compraventa/alquiler no coincide" if not matched else None,
    )


def _property_type_score(expected_types, actual_type, w: dict[str, float]) -> ScoreCriterion:
    if not expected_types:
        earned = w["property_type"] * 0.7
        return ScoreCriterion(name="property_type", earned=earned, maximum=w["property_type"], matched=True)
    matched = actual_type in expected_types
    return ScoreCriterion(
        name="property_type",
        earned=w["property_type"] if matched else 0,
        maximum=w["property_type"],
        matched=matched,
        client_value=", ".join(map(str, expected_types)),
        property_value=str(actual_type),
    )


def _location_score(zones: list[str], cities: list[str], province: str | None, property_: Property, w: dict[str, float]) -> ScoreCriterion:
    normalized_zones = {_clean(value) for value in zones}
    normalized_cities = {_clean(value) for value in cities}
    property_zone = _clean(property_.zone)
    property_city = _clean(property_.city)
    property_province = _clean(property_.province)

    if property_zone and property_zone in normalized_zones:
        earned = w["location"]
        matched = True
        note = "Zona exacta"
    elif property_city and property_city in normalized_cities:
        earned = w["location"] * 0.8
        matched = True
        note = "Ciudad coincidente"
    elif province and property_province == _clean(province):
        earned = w["location"] * 0.5
        matched = True
        note = "Provincia coincidente"
    elif not zones and not cities and not province:
        earned = w["location"] * 0.5
        matched = True
        note = "Cliente sin preferencia geografica"
    else:
        earned = 0
        matched = False
        note = "Ubicacion fuera de preferencia"

    return ScoreCriterion(
        name="location",
        earned=earned,
        maximum=w["location"],
        matched=matched,
        client_value=", ".join(zones + cities + ([province] if province else [])) or None,
        property_value=", ".join(value for value in [property_.zone, property_.city, property_.province] if value),
        note=note,
    )


def _price_score(price_min: int | None, price_max: int | None, actual_price: int, w: dict[str, float]) -> ScoreCriterion:
    if price_min is None and price_max is None:
        return ScoreCriterion(name="price", earned=w["price"] * 0.5, maximum=w["price"], matched=True)

    above_min = price_min is None or actual_price >= price_min
    below_max = price_max is None or actual_price <= price_max
    if above_min and below_max:
        earned = w["price"]
        matched = True
        note = "Dentro del presupuesto"
    elif price_max and actual_price <= price_max * 1.1:
        earned = w["price"] * 0.6
        matched = True
        note = "Ligeramente por encima del presupuesto"
    elif price_max and actual_price <= price_max * 1.2:
        earned = w["price"] * 0.3
        matched = False
        note = "Por encima del presupuesto con margen amplio"
    else:
        earned = 0
        matched = False
        note = "Fuera del presupuesto"

    return ScoreCriterion(
        name="price",
        earned=earned,
        maximum=w["price"],
        matched=matched,
        client_value=_range_text(price_min, price_max),
        property_value=str(actual_price),
        note=note,
    )


def _minimum_score(name: str, expected_min: int | None, actual: int | None, maximum: float) -> ScoreCriterion:
    if expected_min is None:
        return ScoreCriterion(name=name, earned=maximum * 0.5, maximum=maximum, matched=True)
    if actual is None:
        return ScoreCriterion(
            name=name,
            earned=0,
            maximum=maximum,
            matched=False,
            client_value=str(expected_min),
            property_value=None,
            note="Dato no disponible en la propiedad",
        )

    matched = actual >= expected_min
    earned = maximum if matched else 0
    return ScoreCriterion(
        name=name,
        earned=earned,
        maximum=maximum,
        matched=matched,
        client_value=str(expected_min),
        property_value=str(actual),
    )


def _feature_score(expected_features: list[str], actual_features: list[str], w: dict[str, float]) -> ScoreCriterion:
    if not expected_features:
        return ScoreCriterion(name="features", earned=w["features"] * 0.5, maximum=w["features"], matched=True)

    expected = {_clean_feature(value) for value in expected_features}
    actual = {_clean_feature(value) for value in actual_features}
    matches = expected.intersection(actual)
    ratio = len(matches) / len(expected)
    earned = round(w["features"] * ratio, 2)

    return ScoreCriterion(
        name="features",
        earned=earned,
        maximum=w["features"],
        matched=bool(matches),
        client_value=", ".join(sorted(expected)),
        property_value=", ".join(sorted(actual)),
        note=f"{len(matches)}/{len(expected)} preferencias coinciden",
    )


def _clean(value: str | None) -> str:
    if not value:
        return ""
    ascii_value = normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_value.lower().strip().split())


def _clean_feature(value: str) -> str:
    aliases = {
        "terraza": "terrace",
        "balcon": "terrace",
        "piscina": "pool",
        "garaje": "garage",
        "parking": "garage",
        "ascensor": "elevator",
        "jardin": "garden",
        "amueblado": "furnished",
    }
    cleaned = _clean(value)
    return aliases.get(cleaned, cleaned)


def _range_text(minimum: int | None, maximum: int | None) -> str:
    if minimum is not None and maximum is not None:
        return f"{minimum}-{maximum}"
    if minimum is not None:
        return f">={minimum}"
    if maximum is not None:
        return f"<={maximum}"
    return ""

