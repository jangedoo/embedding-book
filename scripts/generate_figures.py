"""Generate explanatory SVG figures for the embedding book.

Run from the repository root:

    poetry run python scripts/generate_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/embedding-book-matplotlib")

import matplotlib.pyplot as plt
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "figures"

COLORS = {
    "query": "#1f77b4",
    "good": "#2ca02c",
    "warn": "#ff7f0e",
    "bad": "#d62728",
    "purple": "#9467bd",
    "gray": "#6b7280",
    "light_gray": "#f3f4f6",
    "line": "#111827",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "svg.fonttype": "none",
            "figure.dpi": 160,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=2.2, color=color, shrinkA=0, shrinkB=0),
    )


def metric_comparison() -> None:
    q = np.array([1.0, 0.0])
    candidates = {
        "A aligned\nshort": np.array([0.75, 0.02]),
        "B long\nless aligned": np.array([1.65, 0.55]),
        "C nearby\nbut angled": np.array([0.82, 0.42]),
        "D opposite": np.array([-0.55, 0.05]),
    }
    candidate_colors = [COLORS["good"], COLORS["warn"], COLORS["purple"], COLORS["bad"]]

    values = np.stack(list(candidates.values()))
    names = list(candidates.keys())
    dot = values @ q
    cosine = dot / (np.linalg.norm(values, axis=1) * np.linalg.norm(q))
    euclidean = np.linalg.norm(values - q, axis=1)

    fig = plt.figure(figsize=(10, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.1], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(0, color="#d1d5db", lw=1)
    ax.axvline(0, color="#d1d5db", lw=1)
    ax.add_patch(patches.Circle((0, 0), 1, fill=False, ec="#cbd5e1", lw=1.2, ls="--"))
    arrow(ax, (0, 0), tuple(q), COLORS["query"])
    ax.text(q[0] + 0.04, q[1] - 0.06, "query", color=COLORS["query"], weight="bold")

    for (name, point), color in zip(candidates.items(), candidate_colors):
        arrow(ax, (0, 0), tuple(point), color)
        ax.scatter(point[0], point[1], s=38, color=color, zorder=3)
        ax.text(point[0] + 0.05, point[1] + 0.04, name, color=color, fontsize=8)

    ax.set_title("One query, several candidate vectors")
    ax.set_xlim(-0.8, 1.9)
    ax.set_ylim(-0.45, 0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax = fig.add_subplot(gs[0, 1])
    metrics = [
        ("cosine\nhigher is better", cosine, np.argmax(cosine), False),
        ("dot product\nhigher is better", dot, np.argmax(dot), False),
        ("Euclidean\nlower is better", euclidean, np.argmin(euclidean), True),
    ]

    y_base = np.arange(len(names))
    bar_height = 0.2
    offsets = [-0.24, 0.0, 0.24]
    for offset, (label, score, winner, invert_axis) in zip(offsets, metrics):
        display_score = -score if invert_axis else score
        bars = ax.barh(y_base + offset, display_score, height=bar_height, label=label)
        for idx, bar in enumerate(bars):
            bar.set_color(candidate_colors[idx])
            bar.set_alpha(1.0 if idx == winner else 0.45)
            if idx == winner:
                ax.text(
                    bar.get_width() + 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    "wins",
                    va="center",
                    fontsize=8,
                    weight="bold",
                )
    ax.axvline(0, color="#d1d5db", lw=1)
    ax.set_yticks(y_base)
    ax.set_yticklabels(names)
    ax.set_xlabel("ranking score; Euclidean is plotted as negative distance")
    ax.set_title("Metric choice changes the winner")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "distance-similarity-metric-comparison.svg")


def draw_matrix(
    ax: plt.Axes,
    x: float,
    y: float,
    rows: int,
    cols: int,
    cell: float,
    label: str,
    color: str,
    highlight_row: int | None = None,
) -> None:
    for r in range(rows):
        for c in range(cols):
            face = "#ffffff"
            if highlight_row is not None and r == highlight_row:
                face = "#e0f2fe"
            rect = patches.Rectangle((x + c * cell, y - r * cell), cell, cell, fc=face, ec="#9ca3af", lw=0.8)
            ax.add_patch(rect)
    ax.add_patch(patches.Rectangle((x, y - (rows - 1) * cell), cols * cell, rows * cell, fill=False, ec=color, lw=2))
    ax.text(x + cols * cell / 2, y + 0.28, label, ha="center", va="bottom", weight="bold", color=color)


def factorized_embeddings() -> None:
    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.6)

    draw_matrix(ax, 0.45, 4.25, rows=5, cols=5, cell=0.45, label="Full table E\nV x d", color=COLORS["query"], highlight_row=2)
    ax.text(1.55, 1.65, "one independent\nrow per token", ha="center", fontsize=8, color=COLORS["gray"])

    ax.text(3.35, 3.15, "=", fontsize=24, weight="bold", ha="center", va="center")
    draw_matrix(ax, 4.0, 4.25, rows=5, cols=2, cell=0.45, label="Latent codes A\nV x r", color=COLORS["good"], highlight_row=2)
    ax.text(4.45, 1.55, "token owns\nr numbers", ha="center", fontsize=8, color=COLORS["gray"])

    ax.text(5.4, 3.15, "x", fontsize=20, weight="bold", ha="center", va="center")
    draw_matrix(ax, 6.0, 3.8, rows=2, cols=5, cell=0.45, label="Shared projection B\nr x d", color=COLORS["warn"])
    ax.text(7.12, 2.35, "shared basis\ndirections", ha="center", fontsize=8, color=COLORS["gray"])

    ax.text(9.3, 3.15, "->", fontsize=20, weight="bold", ha="center", va="center")
    draw_matrix(ax, 10.0, 4.25, rows=5, cols=5, cell=0.45, label="Effective rows AB\nrank <= r", color=COLORS["purple"], highlight_row=2)

    code_y = 4.25 - 2 * 0.45 + 0.22
    for c, txt in enumerate(["a_i1", "a_i2"]):
        ax.text(4.0 + c * 0.45 + 0.22, code_y, txt, ha="center", va="center", fontsize=8, color=COLORS["line"])
        start = (4.0 + c * 0.45 + 0.22, code_y - 0.18)
        end = (6.0, 3.8 - c * 0.45 + 0.22)
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.4))

    ax.annotate(
        "low-rank bottleneck: all token vectors are mixtures of the same r directions",
        xy=(7.2, 1.55),
        xytext=(7.2, 0.65),
        ha="center",
        arrowprops=dict(arrowstyle="-[,widthB=7.5,lengthB=0.8", lw=1.4, color=COLORS["purple"]),
        color=COLORS["purple"],
        fontsize=9,
    )
    save(fig, "factorized-embedding-table.svg")


def canonical_token_collapse() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    tokens = ["Token 1042\ncolor", "Token 7819\ncolour", "Token 9121\ncolr", "Token 220\ncat", "Token 557\ncatalog"]
    token_y = [6.1, 5.1, 4.1, 2.7, 1.7]
    canonical = [("Canonical 17\nCOLOR", 5.05), ("Canonical 52\nCAT", 2.2)]

    for label, y in zip(tokens, token_y):
        ax.add_patch(patches.FancyBboxPatch((0.4, y - 0.34), 1.8, 0.68, boxstyle="round,pad=0.03", fc="#ffffff", ec="#94a3b8"))
        ax.text(1.3, y, label, ha="center", va="center", fontsize=8)

    for label, y in canonical:
        ax.add_patch(patches.FancyBboxPatch((4.0, y - 0.42), 1.9, 0.84, boxstyle="round,pad=0.03", fc="#ecfdf5", ec=COLORS["good"], lw=1.6))
        ax.text(4.95, y, label, ha="center", va="center", fontsize=9, weight="bold", color=COLORS["good"])

    for y in token_y[:3]:
        ax.annotate("", xy=(4.0, 5.05), xytext=(2.2, y), arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.2))
    for y in token_y[3:]:
        ax.annotate("", xy=(4.0, 2.2), xytext=(2.2, y), arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.2))

    ax.text(3.1, 6.45, "many original IDs", ha="center", fontsize=9, color=COLORS["gray"])
    ax.text(4.95, 6.45, "fewer lookup IDs", ha="center", fontsize=9, color=COLORS["gray"])

    table_x, table_y = 7.1, 5.85
    for r in range(5):
        face = "#fee2e2" if r == 1 else "#ffffff"
        ax.add_patch(patches.Rectangle((table_x, table_y - r * 0.6), 2.0, 0.55, fc=face, ec="#9ca3af"))
    ax.add_patch(patches.Rectangle((table_x, table_y - 4 * 0.6), 2.0, 5 * 0.6, fill=False, ec=COLORS["warn"], lw=1.8))
    ax.text(table_x + 1.0, 6.35, "smaller embedding table", ha="center", weight="bold", color=COLORS["warn"])
    ax.text(table_x + 1.0, table_y - 0.6 + 0.27, "row 17 shared", ha="center", va="center", fontsize=9, color=COLORS["bad"], weight="bold")
    ax.annotate("", xy=(table_x, table_y - 0.6 + 0.27), xytext=(5.9, 5.05), arrowprops=dict(arrowstyle="->", color=COLORS["bad"], lw=1.5))

    ax.annotate(
        "decoder ambiguity",
        xy=(5.9, 5.05),
        xytext=(8.15, 1.35),
        ha="center",
        color=COLORS["bad"],
        arrowprops=dict(arrowstyle="->", color=COLORS["bad"], lw=1.4, connectionstyle="arc3,rad=-0.2"),
    )
    for i, word in enumerate(["color", "colour", "colr"]):
        ax.add_patch(patches.FancyBboxPatch((9.0, 0.7 + i * 0.55), 1.25, 0.38, boxstyle="round,pad=0.03", fc="#fff7ed", ec=COLORS["warn"]))
        ax.text(9.62, 0.89 + i * 0.55, word, ha="center", va="center", fontsize=8)

    ax.text(0.45, 0.55, "retrieval risk: merged rare distinctions can change top-k results", fontsize=9, color=COLORS["purple"])
    ax.plot([0.45, 11.0], [0.95, 0.95], color="#e5e7eb", lw=1)
    save(fig, "canonical-token-id-collapse.svg")


def linear_relu_geometry() -> None:
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    points = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    colors = np.where(labels == 1, COLORS["good"], COLORS["bad"])

    ax = axs[0]
    ax.scatter(points[:, 0], points[:, 1], s=70, c=colors, edgecolor="white", linewidth=1)
    ax.axhline(0, color="#d1d5db")
    ax.axvline(0, color="#d1d5db")
    ax.set_title("XOR in 2D")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")

    ax = axs[1]
    projection = points @ np.array([1.0, 0.0])
    ax.scatter(projection, np.zeros_like(projection), s=70, c=colors, edgecolor="white", linewidth=1)
    for x in [-1, 1]:
        ax.text(x, 0.18, "two points\ncollapse", ha="center", fontsize=8, color=COLORS["gray"])
    ax.axhline(0, color="#d1d5db")
    ax.set_yticks([])
    ax.set_title("1D projection")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.45, 0.65)

    ax = axs[2]
    grid_x = np.linspace(-1.6, 1.6, 100)
    grid_y = np.linspace(-1.6, 1.6, 100)
    xx, yy = np.meshgrid(grid_x, grid_y)
    gate = xx + yy > 0
    ax.contourf(xx, yy, gate, levels=[-0.1, 0.5, 1.1], colors=["#f3f4f6", "#dbeafe"], alpha=1)
    ax.contour(xx, yy, xx + yy, levels=[0], colors=[COLORS["query"]], linewidths=2)
    ax.scatter(points[:, 0], points[:, 1], s=55, c=colors, edgecolor="white", linewidth=1)
    ax.set_title("ReLU gate")
    ax.text(-1.45, -1.35, "off", color=COLORS["gray"], weight="bold")
    ax.text(0.75, 1.15, "on", color=COLORS["query"], weight="bold")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")

    ax = axs[3]
    h = np.maximum(
        0,
        np.column_stack(
            [
                points[:, 0] + points[:, 1],
                -points[:, 0] - points[:, 1],
                points[:, 0] - points[:, 1],
                -points[:, 0] + points[:, 1],
            ]
        ),
    )
    ax.scatter(h[:, 0] + h[:, 1], h[:, 2] + h[:, 3], s=70, c=colors, edgecolor="white", linewidth=1)
    ax.axline((0.4, 0), slope=1, color=COLORS["purple"], lw=2)
    ax.text(0.2, 1.55, "hidden ReLU\nfeatures separate\nclasses", fontsize=8, color=COLORS["purple"])
    ax.set_title("MLP feature space")
    ax.set_xlim(-0.1, 2.3)
    ax.set_ylim(-0.1, 2.3)
    ax.set_xlabel("diagonal gates")
    ax.set_ylabel("anti-diagonal gates")

    for ax in axs:
        ax.tick_params(labelsize=8)
    save(fig, "linear-relu-geometry.svg")


def ann_recall_latency() -> None:
    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.35)
    rng = np.random.default_rng(7)
    points = rng.normal(size=(18, 2))
    query = np.array([0.15, 0.25])

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(points[:, 0], points[:, 1], s=24, color="#94a3b8")
    ax.scatter(*query, s=90, marker="*", color=COLORS["query"], zorder=3)
    for p in points:
        ax.plot([query[0], p[0]], [query[1], p[1]], color="#e5e7eb", lw=0.7)
    ax.set_title("Exact search\nchecks every vector")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(points[:, 0], points[:, 1], s=24, color="#94a3b8")
    ax.scatter(*query, s=90, marker="*", color=COLORS["query"], zorder=3)
    order = [3, 9, 14, 7, 4]
    path = np.vstack([query, points[order]])
    ax.plot(path[:, 0], path[:, 1], color=COLORS["good"], lw=2)
    ax.scatter(points[order[-2:]][:, 0], points[order[-2:]][:, 1], s=42, color=COLORS["good"])
    ax.set_title("Graph search\nwalks toward neighbors")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(points[:, 0], points[:, 1], s=24, color="#94a3b8")
    centers = np.array([[-1.1, -0.9], [-0.9, 1.0], [0.9, -0.7], [0.95, 0.8]])
    for center in centers:
        ax.add_patch(patches.Circle(center, 0.9, fill=False, ec="#cbd5e1", lw=1.2))
    ax.scatter(centers[:, 0], centers[:, 1], s=50, marker="x", color=COLORS["warn"])
    ax.scatter(*query, s=90, marker="*", color=COLORS["query"], zorder=3)
    ax.add_patch(patches.Circle((0.95, 0.8), 0.9, fill=True, fc="#dbeafe", ec=COLORS["query"], alpha=0.35))
    ax.set_title("IVF-style search\nprobes nearby cells")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    candidates = np.array([100, 250, 500, 1000, 2000, 5000])
    recall = np.array([0.42, 0.61, 0.74, 0.86, 0.93, 1.0])
    latency = np.array([2.0, 2.7, 3.8, 5.6, 8.4, 16.0])
    ax.plot(latency, recall, marker="o", color=COLORS["purple"], lw=2)
    for x, y, c in zip(latency, recall, candidates):
        ax.text(x, y + 0.025, str(c), ha="center", fontsize=8, color=COLORS["gray"])
    ax.set_xlabel("latency budget")
    ax.set_ylabel("ANN recall@10 vs exact")
    ax.set_ylim(0.35, 1.05)
    ax.set_title("More candidates\nusually improve recall")
    ax.grid(True, color="#e5e7eb", lw=0.8)
    save(fig, "ann-index-recall-latency.svg")


def main() -> None:
    configure_matplotlib()
    metric_comparison()
    factorized_embeddings()
    canonical_token_collapse()
    linear_relu_geometry()
    ann_recall_latency()
    print(f"Wrote SVG figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
