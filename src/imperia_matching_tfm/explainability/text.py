from __future__ import annotations

from imperia_matching_tfm.models import MatchScore


def summarize_match(score: MatchScore, top_n: int = 3) -> list[str]:
    ordered = sorted(score.criteria, key=lambda item: item.earned, reverse=True)
    explanations: list[str] = []
    for criterion in ordered[:top_n]:
        if criterion.earned <= 0:
            continue
        if criterion.note:
            explanations.append(f"{criterion.name}: {criterion.note}")
        else:
            explanations.append(f"{criterion.name}: {criterion.earned:g}/{criterion.maximum:g} puntos")
    return explanations

