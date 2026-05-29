"""Análisis cualitativo de casos representativos del sistema de matching.

Casos analizados:
  1. Match perfecto   - cliente con perfil completo, propiedad que cumple todos los criterios
  2. Datos incompletos - el mismo cliente con campos eliminados, cómo cambia la puntuación

Produce:
  reports/results/qualitative_cases.json

Uso:
    PYTHONPATH=src python3 scripts/run_qualitative_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imperia_matching_tfm.matching.baseline import DEFAULT_WEIGHTS, score_property_for_client
from imperia_matching_tfm.models import Client, Property


def load_data():
    clients_full = json.loads((ROOT / "data/synthetic/clients.json").read_text(encoding="utf-8"))
    clients_deg = json.loads((ROOT / "data/synthetic/clients_degraded.json").read_text(encoding="utf-8"))
    properties = [
        Property.from_dict(p)
        for p in json.loads((ROOT / "data/synthetic/properties.json").read_text(encoding="utf-8"))
    ]
    labels = json.loads((ROOT / "data/labels/relevance_synthetic.json").read_text(encoding="utf-8"))
    relevance = {(r["client_id"], r["property_id"]): int(r["relevance"]) for r in labels}
    return clients_full, clients_deg, properties, relevance


def _criteria_to_dict(score) -> list[dict]:
    return [
        {
            "criterion": c.name,
            "earned": c.earned,
            "maximum": c.maximum,
            "matched": c.matched,
            "client_value": c.client_value,
            "property_value": c.property_value,
            "note": c.note,
        }
        for c in score.criteria
    ]


def _top_k_ranking(client: Client, properties: list[Property], weights: dict, k: int = 5) -> list[dict]:
    scored = sorted(
        [score_property_for_client(client, p, weights) for p in properties],
        key=lambda s: s.score,
        reverse=True,
    )
    return [{"rank": i + 1, "property_id": s.property_id, "score": s.score} for i, s in enumerate(scored[:k])]


def case_match_perfecto(clients_full, properties, relevance) -> dict:
    # c_005 Laura Sanchez: 6 campos rellenos, top property p_003 con score 91.5 y relevance=3
    c_data = next(c for c in clients_full if c["id"] == "c_005")
    client = Client.from_dict(c_data)
    prop = next(p for p in properties if p.id == "p_003")

    score = score_property_for_client(client, prop, DEFAULT_WEIGHTS)
    rel = relevance.get((client.id, prop.id), 0)
    ranking = _top_k_ranking(client, properties, DEFAULT_WEIGHTS)

    return {
        "caso": "match_perfecto",
        "descripcion": (
            "Cliente con perfil completo. La propiedad top del ranking cumple todos los criterios "
            "y tiene la máxima relevancia en las etiquetas."
        ),
        "cliente": {
            "id": client.id,
            "nombre": client.name,
            "preferencia": {
                "operation": str(client.preference.operation),
                "property_types": [str(t) for t in client.preference.property_types],
                "zones": client.preference.zones,
                "cities": client.preference.cities,
                "price_max": client.preference.price_max,
                "bedrooms_min": client.preference.bedrooms_min,
                "bathrooms_min": client.preference.bathrooms_min,
                "surface_min_m2": client.preference.surface_min_m2,
                "features": client.preference.features,
            },
        },
        "propiedad": {
            "id": prop.id,
            "title": prop.title,
            "operation": str(prop.operation),
            "property_type": str(prop.property_type),
            "price": prop.price,
            "city": prop.city,
            "zone": prop.zone,
            "bedrooms": prop.bedrooms,
            "bathrooms": prop.bathrooms,
            "surface_m2": prop.surface_m2,
            "features": prop.features,
        },
        "score_total": score.score,
        "relevance_etiqueta": rel,
        "posicion_en_ranking": next((r["rank"] for r in ranking if r["property_id"] == prop.id), None),
        "ranking_top5": ranking,
        "desglose_criterios": _criteria_to_dict(score),
        "observacion": (
            "Todos los criterios están cubiertos. La propiedad aparece en el puesto 1 del ranking "
            "con score 91.5/100 y relevance=3. Los puntos no perfectos en bedrooms, built_year y floor "
            "se deben a que el cliente no especificó mínimo para esos campos (puntuación parcial por defecto)."
        ),
    }


def case_datos_incompletos(clients_full, clients_deg, properties, relevance) -> dict:
    # Mismo cliente c_005 en versión degradada: elimina property_types, price_max, bathrooms_min
    c_full_data = next(c for c in clients_full if c["id"] == "c_005")
    c_deg_data = next(c for c in clients_deg if c["id"] == "c_005")

    client_full = Client.from_dict(c_full_data)
    clean_deg = {k: v for k, v in c_deg_data.items() if k != "dropped_fields"}
    client_deg = Client.from_dict(clean_deg)
    dropped_fields = c_deg_data.get("dropped_fields", [])

    prop = next(p for p in properties if p.id == "p_003")

    score_full = score_property_for_client(client_full, prop, DEFAULT_WEIGHTS)
    score_deg = score_property_for_client(client_deg, prop, DEFAULT_WEIGHTS)

    ranking_full = _top_k_ranking(client_full, properties, DEFAULT_WEIGHTS)
    ranking_deg = _top_k_ranking(client_deg, properties, DEFAULT_WEIGHTS)

    diff_per_criterion = []
    for cf, cd in zip(score_full.criteria, score_deg.criteria):
        if cf.earned != cd.earned:
            diff_per_criterion.append({
                "criterion": cf.name,
                "score_completo": cf.earned,
                "score_degradado": cd.earned,
                "diferencia": round(cd.earned - cf.earned, 2),
            })

    return {
        "caso": "datos_incompletos",
        "descripcion": (
            "El mismo cliente con campos eliminados (simulando un lead parcial). "
            "Se analiza cómo cambia la puntuación y el ranking cuando faltan campos clave."
        ),
        "cliente_id": "c_005",
        "nombre": client_full.name,
        "campos_eliminados": dropped_fields,
        "score_con_perfil_completo": score_full.score,
        "score_con_datos_incompletos": score_deg.score,
        "diferencia_score": round(score_deg.score - score_full.score, 2),
        "posicion_ranking_completo": next((r["rank"] for r in ranking_full if r["property_id"] == prop.id), None),
        "posicion_ranking_degradado": next((r["rank"] for r in ranking_deg if r["property_id"] == prop.id), None),
        "ranking_top5_completo": ranking_full,
        "ranking_top5_degradado": ranking_deg,
        "criterios_afectados": diff_per_criterion,
        "desglose_completo": _criteria_to_dict(score_full),
        "desglose_degradado": _criteria_to_dict(score_deg),
        "observacion": (
            "Al eliminar property_types, price_max y bathrooms_min, el sistema aplica puntuación parcial "
            "en lugar de penalizar con cero: sin tipo de propiedad se otorga el 70% del peso; sin precio "
            "máximo se otorga el 50%. Esto refleja incertidumbre sin descartar la propiedad. "
            "La propiedad sigue apareciendo en el top del ranking pero con menor score."
        ),
    }


def main() -> None:
    clients_full, clients_deg, properties, relevance = load_data()

    cases = {
        "caso_1_match_perfecto": case_match_perfecto(clients_full, properties, relevance),
        "caso_2_datos_incompletos": case_datos_incompletos(clients_full, clients_deg, properties, relevance),
    }

    output = ROOT / "reports/results/qualitative_cases.json"
    output.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Análisis guardado en {output.relative_to(ROOT)}")

    # Resumen en consola
    print("\n=== CASO 1: Match perfecto ===")
    c1 = cases["caso_1_match_perfecto"]
    print(f"  Cliente: {c1['cliente']['nombre']} ({c1['cliente']['id']})")
    print(f"  Propiedad: {c1['propiedad']['id']} — {c1['propiedad']['title']}")
    print(f"  Score: {c1['score_total']} | Relevance: {c1['relevance_etiqueta']} | Posición: #{c1['posicion_en_ranking']}")
    print("  Criterios:")
    for cr in c1["desglose_criterios"]:
        print(f"    {cr['criterion']:<14} {cr['earned']:>5.1f}/{cr['maximum']:<4.0f}  {'✓' if cr['matched'] else '✗'}  {cr['note'] or ''}")

    print("\n=== CASO 2: Datos incompletos ===")
    c2 = cases["caso_2_datos_incompletos"]
    print(f"  Cliente: {c2['nombre']} ({c2['cliente_id']})")
    print(f"  Campos eliminados: {c2['campos_eliminados']}")
    print(f"  Score completo:   {c2['score_con_perfil_completo']} (pos. #{c2['posicion_ranking_completo']})")
    print(f"  Score degradado:  {c2['score_con_datos_incompletos']} (pos. #{c2['posicion_ranking_degradado']})")
    print(f"  Diferencia:       {c2['diferencia_score']:+.1f} puntos")
    if c2["criterios_afectados"]:
        print("  Criterios afectados:")
        for cr in c2["criterios_afectados"]:
            print(f"    {cr['criterion']:<14} {cr['score_completo']:>5.1f} → {cr['score_degradado']:.1f}  ({cr['diferencia']:+.1f})")


if __name__ == "__main__":
    main()
