"""M4: matching hibrido estructurado + semantico.

Combina el score estructurado (DEFAULT_WEIGHTS de baseline) con una
componente semantica derivada de embeddings. El score semantico se
incorpora como un criterio adicional explicable, no como sustituto.

score_total = alpha * score_estructurado + (1 - alpha) * score_semantico * 100

Por defecto alpha=0.7 prioriza el desglose explicable pero permite
que la afinidad textual recupere matches que el matching exacto pierde
(sinonimos de barrio, preferencias implicitas en notas).
"""
from __future__ import annotations

from imperia_matching_tfm.matching.baseline import score_property_for_client
from imperia_matching_tfm.matching.semantic import semantic_score
from imperia_matching_tfm.models import Client, MatchScore, Property, ScoreCriterion


DEFAULT_ALPHA = 0.7


def score_property_for_client_m4(
    client: Client,
    property_: Property,
    embeddings_cache: dict,
    alpha: float = DEFAULT_ALPHA,
    weights: dict[str, float] | None = None,
) -> MatchScore:
    structured = score_property_for_client(client, property_, weights=weights)

    # Mantener el dealbreaker de operacion: si el score estructurado es 0
    # por incompatibilidad de compraventa/alquiler, no rescatar via semantica.
    if structured.score == 0:
        return structured

    sim = semantic_score(client.id, property_.id, embeddings_cache)
    semantic_points = sim * 100.0
    combined = alpha * structured.score + (1.0 - alpha) * semantic_points

    semantic_criterion = ScoreCriterion(
        name="semantic",
        earned=round((1.0 - alpha) * semantic_points, 2),
        maximum=round((1.0 - alpha) * 100.0, 2),
        matched=sim >= 0.5,
        client_value=None,
        property_value=None,
        note=f"Afinidad semantica: {sim:.2f}",
    )

    return MatchScore(
        client_id=client.id,
        property_id=property_.id,
        score=round(combined, 2),
        criteria=[*structured.criteria, semantic_criterion],
    )
