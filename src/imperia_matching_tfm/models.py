from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Operation(StrEnum):
    BUY = "BUY"
    RENT = "RENT"


class PropertyType(StrEnum):
    FLAT = "FLAT"
    HOUSE = "HOUSE"
    CHALET = "CHALET"
    PENTHOUSE = "PENTHOUSE"
    STUDIO = "STUDIO"
    COMMERCIAL = "COMMERCIAL"
    LAND = "LAND"
    OTHER = "OTHER"


@dataclass
class Property:
    id: str
    title: str
    operation: Operation
    property_type: PropertyType
    price: int
    city: str
    zone: str | None = None
    province: str | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    surface_m2: int | None = None
    usable_surface_m2: int | None = None
    built_year: int | None = None
    floor: int | None = None
    features: list[str] = field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Property:
        return cls(
            **{
                **payload,
                "operation": Operation(payload["operation"]),
                "property_type": PropertyType(payload["property_type"]),
            }
        )


@dataclass
class BuyerPreference:
    operation: Operation | None = None
    property_types: list[PropertyType] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)
    cities: list[str] = field(default_factory=list)
    province: str | None = None
    price_min: int | None = None
    price_max: int | None = None
    bedrooms_min: int | None = None
    bathrooms_min: int | None = None
    surface_min_m2: int | None = None
    built_year_min: int | None = None
    floor_min: int | None = None
    features: list[str] = field(default_factory=list)
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BuyerPreference:
        operation = payload.get("operation")
        return cls(
            **{
                **payload,
                "operation": Operation(operation) if operation else None,
                "property_types": [PropertyType(value) for value in payload.get("property_types", [])],
            }
        )


@dataclass
class Client:
    id: str
    name: str
    preference: BuyerPreference

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Client:
        return cls(
            id=payload["id"],
            name=payload["name"],
            preference=BuyerPreference.from_dict(payload["preference"]),
        )


@dataclass
class ScoreCriterion:
    name: str
    earned: float
    maximum: float
    matched: bool
    client_value: str | None = None
    property_value: str | None = None
    note: str | None = None


@dataclass
class MatchScore:
    client_id: str
    property_id: str
    score: float
    criteria: list[ScoreCriterion]
