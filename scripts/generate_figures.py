"""Genera las gráficas comparativas de modelos para la memoria del TFM.

Produce tres figuras en reports/figures/:
  - ndcg_comparison.png   : barras agrupadas NDCG@K por modelo
  - recall_comparison.png : barras agrupadas Recall@K por modelo
  - weights_radar.png     : radar de pesos M0 vs M1 vs M3

Uso:
    python3 scripts/generate_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports/figures"
RESULTS_DIR = ROOT / "reports/results"

MODELS = ["M0 Baseline", "M1 Ajustado", "M2 Enriquecido", "M3 Hibrido", "M4 Semantico"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
K_VALUES = [3, 5, 10]
DEGRADED_MODELS = ["M0d Degradado", "M2 Enriquecido", "M4d Semantico+Degradado"]
DEGRADED_COLORS = ["#888888", "#C44E52", "#CCB974"]


def load_results() -> dict:
    return json.loads((RESULTS_DIR / "comparison_all.json").read_text(encoding="utf-8"))


def load_weights() -> dict[str, dict]:
    m0 = {
        "operation": 20, "property_type": 13, "location": 20, "price": 20,
        "bedrooms": 7, "bathrooms": 3, "surface": 7, "features": 4,
        "built_year": 3, "floor": 3,
    }
    m1 = json.loads((RESULTS_DIR / "m1_weights.json").read_text(encoding="utf-8"))["weights"]
    m3 = json.loads((RESULTS_DIR / "m3_weights.json").read_text(encoding="utf-8"))["weights"]
    return {"M0 Baseline": m0, "M1 Ajustado": m1, "M3 Híbrido": m3}


def _bar_comparison(
    results: dict,
    metric_prefix: str,
    ylabel: str,
    title: str,
    filename: str,
    models: list[str] | None = None,
    colors: list[str] | None = None,
) -> None:
    models = models or MODELS
    colors = colors or COLORS
    available = [m for m in models if m in results]
    if len(available) < 2:
        return
    x = np.arange(len(K_VALUES))
    width = 0.85 / max(len(available), 1)
    offsets = np.linspace(-(len(available) - 1) / 2, (len(available) - 1) / 2, len(available)) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model, color) in enumerate(zip(available, colors)):
        values = [results[model][f"{metric_prefix}@{k}"] for k in K_VALUES]
        bars = ax.bar(x + offsets[i], values, width, label=model, color=color, alpha=0.88)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_xlabel("K", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES])
    ax.legend(fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardada: {path.relative_to(ROOT)}")


def _radar(weights: dict[str, dict], filename: str) -> None:
    criteria = list(next(iter(weights.values())).keys())
    n = len(criteria)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    radar_colors = ["#4C72B0", "#55A868", "#8172B2"]
    for (model, w), color in zip(weights.items(), radar_colors):
        total = sum(w.values())
        values = [w[c] / total * 100 for c in criteria]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    labels = [c.replace("_", "\n").capitalize() for c in criteria]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 30)
    ax.set_yticks([5, 10, 15, 20, 25])
    ax.set_yticklabels(["5%", "10%", "15%", "20%", "25%"], fontsize=7, color="gray")
    ax.set_title("Distribución de pesos por criterio (%)", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {path.relative_to(ROOT)}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    weights = load_weights()

    print("Generando gráficas...")
    _bar_comparison(results, "ndcg", "NDCG@K", "NDCG@K por modelo", "ndcg_comparison.png")
    _bar_comparison(results, "recall", "Recall@K", "Recall@K por modelo", "recall_comparison.png")
    _bar_comparison(
        results, "ndcg", "NDCG@K",
        "NDCG@K sobre clientes degradados",
        "ndcg_degraded.png", models=DEGRADED_MODELS, colors=DEGRADED_COLORS,
    )
    _bar_comparison(
        results, "recall", "Recall@K",
        "Recall@K sobre clientes degradados",
        "recall_degraded.png", models=DEGRADED_MODELS, colors=DEGRADED_COLORS,
    )
    _radar(weights, "weights_radar.png")
    print("Listo.")


if __name__ == "__main__":
    main()
