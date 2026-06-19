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
            "svg.hashsalt": "embedding-book",
            "figure.dpi": 160,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
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


def label_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = "#ffffff",
    ec: str = "#94a3b8",
    color: str = COLORS["line"],
    fontsize: int = 9,
    weight: str = "normal",
) -> None:
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=fontsize, weight=weight)


def embedding_lookup_to_space() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    rows = ["token 17", "item 42", "doc 108", "user 9", "token 511"]
    for i, row in enumerate(rows):
        y = 4.9 - i * 0.75
        face = "#dbeafe" if i in [1, 3] else "#ffffff"
        label_box(ax, 0.6, y, 2.0, 0.48, row, fc=face, ec="#94a3b8")
        for c in range(5):
            ax.add_patch(patches.Rectangle((2.85 + c * 0.38, y), 0.34, 0.48, fc=face, ec="#cbd5e1", lw=0.8))
    ax.text(1.95, 5.65, "embedding table E", ha="center", weight="bold", color=COLORS["query"])
    ax.text(3.8, 5.65, "d coordinates per row", ha="center", fontsize=9, color=COLORS["gray"])

    points = np.array([[7.0, 4.5], [8.0, 4.05], [8.6, 2.2], [6.7, 2.0], [9.3, 4.8], [7.7, 2.7]])
    ax.scatter(points[:, 0], points[:, 1], s=60, color="#94a3b8")
    ax.scatter(points[[1, 3], 0], points[[1, 3], 1], s=120, color=[COLORS["good"], COLORS["purple"]], edgecolor="white", linewidth=1)
    for src, dst, color in [((4.75, 4.15), tuple(points[1]), COLORS["good"]), ((4.75, 2.65), tuple(points[3]), COLORS["purple"])]:
        ax.annotate("", xy=dst, xytext=src, arrowprops=dict(arrowstyle="->", lw=1.7, color=color))
    ax.text(8.1, 5.45, "same rows become locations", ha="center", weight="bold", color=COLORS["good"])
    ax.add_patch(patches.Circle(points[1], 0.8, fill=False, ec=COLORS["good"], lw=1.5, ls="--"))
    ax.text(8.95, 3.45, "a metric later decides\nwhich locations are close", fontsize=9, color=COLORS["gray"])
    save(fig, "embedding-lookup-to-space.svg")


def vectors_norms_angles_projection() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.axhline(0, color="#d1d5db", lw=1)
    ax.axvline(0, color="#d1d5db", lw=1)
    x = np.array([3.0, 2.0])
    y = np.array([3.5, 0.6])
    proj = (x @ y) / (y @ y) * y
    arrow(ax, (0, 0), tuple(x), COLORS["query"])
    arrow(ax, (0, 0), tuple(y), COLORS["good"])
    ax.plot([x[0], proj[0]], [x[1], proj[1]], color=COLORS["warn"], ls="--", lw=1.5)
    ax.scatter([proj[0]], [proj[1]], color=COLORS["warn"], zorder=3)
    ax.add_patch(patches.Arc((0, 0), 1.45, 1.45, theta1=np.degrees(np.arctan2(y[1], y[0])), theta2=np.degrees(np.arctan2(x[1], x[0])), color=COLORS["purple"], lw=2))
    ax.text(2.0, 1.7, "x", color=COLORS["query"], weight="bold")
    ax.text(2.7, 0.28, "y", color=COLORS["good"], weight="bold")
    ax.text(1.55, 0.95, "angle", color=COLORS["purple"], fontsize=9)
    ax.text(proj[0] + 0.1, proj[1] - 0.35, "projection:\ncomponent of x along y", color=COLORS["warn"], fontsize=9)
    ax.text(3.2, 1.35, "length = norm", color=COLORS["query"], fontsize=9)
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(-0.45, 2.8)
    ax.set_aspect("equal")
    ax.set_title("Norms, angles, and projections")
    ax.set_xlabel("coordinate 1")
    ax.set_ylabel("coordinate 2")
    save(fig, "vectors-norms-angles-projection.svg")


def vector_length_ranking() -> None:
    fig = plt.figure(figsize=(10, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    q = np.array([1.0, 0.0])
    aligned_short = np.array([0.8, 0.0])
    long_tilted = 2.3 * np.array([0.86, 0.5])
    arrow(ax, (0, 0), tuple(q), COLORS["query"])
    arrow(ax, (0, 0), tuple(aligned_short), COLORS["good"])
    arrow(ax, (0, 0), tuple(long_tilted), COLORS["warn"])
    ax.add_patch(patches.Circle((0, 0), 1, fill=False, ec="#cbd5e1", ls="--"))
    ax.text(1.04, -0.1, "query", color=COLORS["query"], weight="bold")
    ax.text(0.35, 0.18, "same direction\nshorter norm", color=COLORS["good"], fontsize=8)
    ax.text(1.42, 1.23, "longer but\nless aligned", color=COLORS["warn"], fontsize=8)
    ax.set_xlim(-0.2, 2.35)
    ax.set_ylim(-0.45, 1.65)
    ax.set_aspect("equal")
    ax.set_title("Length can change dot-product ranking")

    ax = fig.add_subplot(gs[0, 1])
    candidates = ["aligned\nshort", "tilted\nlong"]
    cosine = [1.0, long_tilted[0] / np.linalg.norm(long_tilted)]
    dot = [aligned_short @ q, long_tilted @ q]
    y = np.arange(2)
    ax.barh(y - 0.14, cosine, height=0.25, color=COLORS["good"], label="cosine")
    ax.barh(y + 0.14, dot, height=0.25, color=COLORS["warn"], label="dot product")
    ax.set_yticks(y)
    ax.set_yticklabels(candidates)
    ax.set_xlabel("score")
    ax.set_title("Cosine ignores length; dot product does not")
    ax.legend(frameon=False)
    save(fig, "vector-length-ranking.svg")


def high_dimensional_concentration_hubs() -> None:
    rng = np.random.default_rng(3)
    dims = [32, 128, 768]
    samples = {d: rng.normal(size=(3000, d)) for d in dims}
    cosines = {}
    for d, x in samples.items():
        a = x[:1500] / np.linalg.norm(x[:1500], axis=1, keepdims=True)
        b = x[1500:] / np.linalg.norm(x[1500:], axis=1, keepdims=True)
        cosines[d] = np.sum(a * b, axis=1)

    X = rng.normal(size=(180, 64))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    bias = np.zeros(64)
    bias[0] = 1.8
    Y = X + bias
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    sims = Y @ Y.T
    np.fill_diagonal(sims, -np.inf)
    top = np.argpartition(-sims, 8, axis=1)[:, :8].ravel()
    counts = np.bincount(top, minlength=len(Y))
    top_counts = np.sort(counts)[-12:][::-1]

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for d, color in zip(dims, [COLORS["warn"], COLORS["good"], COLORS["query"]]):
        ax.hist(cosines[d], bins=40, density=True, alpha=0.35, label=f"d={d}", color=color)
    ax.axvline(0, color=COLORS["line"], lw=1)
    ax.set_title("Random cosine similarities concentrate near zero")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(np.arange(len(top_counts)), top_counts, color=[COLORS["bad"] if i < 3 else COLORS["gray"] for i in range(len(top_counts))])
    ax.set_title("Shared bias creates hub neighbors")
    ax.set_xlabel("most frequent retrieved points")
    ax.set_ylabel("times appearing in top-8")
    ax.text(0.4, max(top_counts) * 0.88, "a few points become\nnearest neighbors for many queries", color=COLORS["bad"], fontsize=9)
    save(fig, "high-dimensional-concentration-hubs.svg")


def batch_embedding_gradient_rows() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    draw_matrix(ax, 0.8, 5.0, rows=7, cols=5, cell=0.45, label="Embedding table", color=COLORS["query"])
    for r in [0, 2, 5]:
        ax.add_patch(patches.Rectangle((0.8, 5.0 - r * 0.45), 5 * 0.45, 0.45, fc="#dbeafe", ec=COLORS["query"], lw=1.2))
    ax.text(1.95, 1.55, "only IDs in the batch\nare looked up", ha="center", color=COLORS["query"], fontsize=9)
    ids = ["id 0", "id 2", "id 2", "id 5"]
    for i, text in enumerate(ids):
        label_box(ax, 4.2, 4.7 - i * 0.7, 1.0, 0.42, text, fc="#f8fafc")
        target_r = [0, 2, 2, 5][i]
        ax.annotate("", xy=(3.05, 5.2 - target_r * 0.45), xytext=(4.2, 4.9 - i * 0.7), arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1))
    label_box(ax, 6.1, 3.2, 1.8, 0.8, "downstream\nlayers", fc="#ecfdf5", ec=COLORS["good"], color=COLORS["good"], weight="bold")
    ax.annotate("", xy=(6.1, 3.6), xytext=(5.2, 3.6), arrowprops=dict(arrowstyle="->", color=COLORS["good"], lw=1.7))
    draw_matrix(ax, 9.0, 5.0, rows=7, cols=5, cell=0.45, label="Gradient", color=COLORS["warn"])
    for r in [0, 2, 5]:
        ax.add_patch(patches.Rectangle((9.0, 5.0 - r * 0.45), 5 * 0.45, 0.45, fc="#ffedd5", ec=COLORS["warn"], lw=1.2))
    ax.text(10.12, 1.55, "backprop updates\nselected rows", ha="center", color=COLORS["warn"], fontsize=9)
    ax.annotate("", xy=(9.0, 3.6), xytext=(7.9, 3.6), arrowprops=dict(arrowstyle="->", color=COLORS["warn"], lw=1.7))
    save(fig, "batch-embedding-gradient-rows.svg")


def prediction_objectives_neighborhoods() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(size=(15, 2))
    cat = np.array([[0, 0], [2.1, 0], [1.0, 1.7]])[np.arange(15) % 3] + rng.normal(scale=0.18, size=(15, 2))
    price = np.column_stack([np.linspace(-1.8, 1.8, 15), rng.normal(scale=0.18, size=15)])
    rel = base.copy()
    rel[:5] += np.array([1.5, 0.8])
    rel[5:10] += np.array([-1.2, 0.3])
    data = [cat, price, rel]
    titles = ["category objective\nclusters by class", "regression objective\norders by score", "retrieval objective\npulls positives together"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    colors = [COLORS["query"], COLORS["good"], COLORS["warn"]]
    for ax, pts, title in zip(axs, data, titles):
        ax.scatter(pts[:, 0], pts[:, 1], c=[colors[i % 3] for i in range(15)], s=54, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    axs[2].annotate("", xy=rel[1], xytext=rel[7], arrowprops=dict(arrowstyle="->", color=COLORS["purple"], lw=1.8))
    axs[2].text(rel[1, 0], rel[1, 1] + 0.28, "positive\npair", color=COLORS["purple"], fontsize=8)
    save(fig, "prediction-objectives-neighborhoods.svg")


def contrastive_ranking_forces() -> None:
    fig = plt.figure(figsize=(10.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    q = np.array([0.0, 0.0])
    pos = np.array([1.5, 0.4])
    negs = np.array([[0.8, 1.6], [1.8, -0.9], [-1.1, 0.9], [-1.4, -0.5]])
    ax.scatter(*q, marker="*", s=180, color=COLORS["query"], label="query")
    ax.scatter(*pos, s=90, color=COLORS["good"], label="positive")
    ax.scatter(negs[:, 0], negs[:, 1], s=70, color=COLORS["bad"], alpha=0.75, label="negatives")
    ax.annotate("", xy=pos, xytext=q, arrowprops=dict(arrowstyle="->", color=COLORS["good"], lw=2.2))
    for n in negs:
        ax.annotate("", xy=n + 0.35 * (n - q) / np.linalg.norm(n - q), xytext=n, arrowprops=dict(arrowstyle="->", color=COLORS["bad"], lw=1.4))
    ax.text(0.65, 0.52, "pull", color=COLORS["good"], weight="bold")
    ax.text(-1.65, 1.35, "push", color=COLORS["bad"], weight="bold")
    ax.set_title("Contrastive forces in embedding space")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[0, 1])
    docs = ["positive doc", "hard negative", "easy negative", "other negative"]
    scores = [0.83, 0.72, 0.18, 0.07]
    colors = [COLORS["good"], COLORS["bad"], "#fca5a5", "#fecaca"]
    ax.barh(np.arange(4), scores, color=colors)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(docs)
    ax.set_xlabel("similarity score")
    ax.set_title("Training tries to move the positive above negatives")
    ax.text(0.76, 1, "margin pressure", color=COLORS["bad"], fontsize=9, va="center")
    save(fig, "contrastive-ranking-forces.svg")


def embedding_memory_accounting() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    labels = ["weights", "gradients", "Adam m", "Adam v"]
    colors = [COLORS["query"], COLORS["warn"], COLORS["good"], COLORS["purple"]]
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = 0.8 + i * 2.2
        ax.add_patch(patches.Rectangle((x, 2.0), 1.7, 2.5, fc="#ffffff", ec=color, lw=2))
        ax.text(x + 0.85, 4.8, label, ha="center", weight="bold", color=color)
        ax.text(x + 0.85, 3.2, "V x d", ha="center", va="center", fontsize=15, color=color)
    ax.text(4.1, 1.25, "optimizer choice can multiply stored embedding memory", ha="center", color=COLORS["gray"], fontsize=10)
    for r in [0.2, 0.7, 1.5]:
        ax.add_patch(patches.Rectangle((9.95, 2.0 + r), 1.35, 0.22, fc="#dbeafe", ec=COLORS["query"]))
    label_box(ax, 9.55, 4.55, 2.2, 0.55, "batch touches\nfew rows", fc="#eff6ff", ec=COLORS["query"], color=COLORS["query"], weight="bold")
    ax.annotate("", xy=(10.6, 3.6), xytext=(10.6, 4.55), arrowprops=dict(arrowstyle="->", color=COLORS["query"], lw=1.5))
    ax.text(10.62, 1.25, "stored size scales with V x d;\nupdate work scales with unique IDs", ha="center", color=COLORS["gray"], fontsize=9)
    save(fig, "embedding-memory-accounting.svg")


def quantization_ranking_pq() -> None:
    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.05, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    vals = np.array([0.82, 0.31, -0.12, -0.64, 0.48, 0.12, -0.28, 0.7])
    qvals = np.round(vals * 3) / 3
    x = np.arange(len(vals))
    ax.vlines(x - 0.1, 0, vals, color=COLORS["query"], lw=6, alpha=0.55, label="float")
    ax.vlines(x + 0.1, 0, qvals, color=COLORS["warn"], lw=6, alpha=0.8, label="quantized")
    ax.axhline(0, color="#d1d5db")
    ax.set_title("Continuous values snap to levels")
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    before = [0.91, 0.88, 0.65]
    after = [0.89, 0.90, 0.63]
    y = np.arange(3)
    ax.barh(y - 0.13, before, height=0.22, color=COLORS["good"], label="before")
    ax.barh(y + 0.13, after, height=0.22, color=COLORS["warn"], label="after")
    ax.set_yticks(y)
    ax.set_yticklabels(["doc A", "doc B", "doc C"])
    ax.invert_yaxis()
    ax.set_xlabel("similarity")
    ax.set_title("Small score changes can swap neighbors")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    for i, color in enumerate([COLORS["query"], COLORS["good"], COLORS["warn"], COLORS["purple"]]):
        label_box(ax, 0.3 + i * 0.85, 2.6, 0.62, 0.55, f"x{i+1}", fc="#ffffff", ec=color, color=color)
        ax.annotate("", xy=(0.6 + i * 0.85, 2.35), xytext=(0.6 + i * 0.85, 2.6), arrowprops=dict(arrowstyle="->", color=color))
        label_box(ax, 0.15 + i * 0.85, 1.55, 0.9, 0.55, f"code\n{i+7}", fc="#f8fafc", ec=color, color=color, fontsize=8)
    ax.text(1.8, 3.45, "product quantization", ha="center", weight="bold")
    ax.text(1.8, 0.85, "split vector into chunks;\nstore codebook IDs per chunk", ha="center", color=COLORS["gray"], fontsize=9)
    save(fig, "quantization-ranking-pq.svg")


def retrieval_rag_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.add_patch(patches.Circle((2.15, 2.85), 1.25, fill=False, ec="#cbd5e1", lw=1.2))
    q = np.array([2.15, 2.85])
    docs = np.array([[3.0, 3.15], [2.65, 3.85], [0.95, 3.55], [1.25, 1.85], [3.3, 1.75]])
    ax.scatter(docs[:, 0], docs[:, 1], s=50, color="#94a3b8")
    ax.scatter(*q, marker="*", s=160, color=COLORS["query"])
    for i, p in enumerate(docs[:3], start=1):
        ax.plot([q[0], p[0]], [q[1], p[1]], color=COLORS["good"] if i == 1 else "#d1d5db", lw=1.2)
        ax.text(p[0] + 0.08, p[1] + 0.08, f"d{i}", fontsize=8)
    ax.text(2.15, 4.55, "geometry ranks chunks", ha="center", weight="bold", color=COLORS["query"])
    steps = [("nearest\nchunks", 4.7), ("reranker", 6.25), ("context\nwindow", 7.8), ("generator", 9.55)]
    for text, x in steps:
        label_box(ax, x, 2.5, 1.15, 0.75, text, fc="#ffffff", ec=COLORS["purple"], color=COLORS["purple"], weight="bold")
    for x1, x2 in [(4.0, 4.7), (5.85, 6.25), (7.4, 7.8), (9.15, 9.55)]:
        ax.annotate("", xy=(x2, 2.88), xytext=(x1, 2.88), arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.4))
    ax.text(7.8, 1.35, "missing evidence here is\nusually unrecoverable later", ha="center", color=COLORS["bad"], fontsize=9)
    ax.annotate("", xy=(7.9, 2.5), xytext=(7.9, 1.65), arrowprops=dict(arrowstyle="->", color=COLORS["bad"], lw=1.2))
    save(fig, "retrieval-rag-pipeline.svg")


def evaluation_metrics_ranked_list() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    rels = [0, 2, 0, 1, 2, 0, 0, 1, 0, 0]
    rel_colors = {0: "#fee2e2", 1: "#fef3c7", 2: "#dcfce7"}
    rel_text = {0: "irrelevant", 1: "partial", 2: "answer"}
    for i, rel in enumerate(rels):
        y = 5.35 - i * 0.46
        ax.add_patch(patches.Rectangle((0.8, y), 4.3, 0.36, fc=rel_colors[rel], ec="#cbd5e1"))
        ax.text(1.05, y + 0.18, f"rank {i+1}", va="center", fontsize=8)
        ax.text(3.7, y + 0.18, rel_text[rel], va="center", ha="center", fontsize=8)
    label_box(ax, 6.1, 4.7, 2.5, 0.55, "Recall@5 = found answer", fc="#ecfdf5", ec=COLORS["good"], color=COLORS["good"])
    label_box(ax, 6.1, 3.85, 2.5, 0.55, "MRR = 1 / first answer rank", fc="#eff6ff", ec=COLORS["query"], color=COLORS["query"])
    label_box(ax, 6.1, 3.0, 2.5, 0.55, "nDCG rewards early strong relevance", fc="#f5f3ff", ec=COLORS["purple"], color=COLORS["purple"])
    label_box(ax, 6.1, 1.7, 2.5, 0.7, "split by document,\nnot leaked chunks", fc="#fff7ed", ec=COLORS["warn"], color=COLORS["warn"], weight="bold")
    ax.text(2.95, 5.85, "one ranked list, different metric stories", ha="center", weight="bold")
    save(fig, "evaluation-metrics-ranked-list.svg")


def hybrid_reranking_timeline() -> None:
    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5)
    lists = [("dense", 0.4, [COLORS["good"], COLORS["good"], "#e5e7eb", COLORS["bad"]]), ("BM25", 2.2, [COLORS["bad"], "#e5e7eb", COLORS["good"], COLORS["good"]]), ("reranked", 4.0, [COLORS["good"], COLORS["bad"], COLORS["good"], "#e5e7eb"])]
    for title, x, cols in lists:
        ax.text(x + 0.65, 4.55, title, ha="center", weight="bold")
        for i, c in enumerate(cols):
            ax.add_patch(patches.Rectangle((x, 3.8 - i * 0.65), 1.3, 0.46, fc=c, ec="#cbd5e1"))
            ax.text(x + 0.65, 4.03 - i * 0.65, f"doc {i+1}", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(2.1, 2.8), xytext=(1.7, 2.8), arrowprops=dict(arrowstyle="->", color=COLORS["gray"]))
    ax.annotate("", xy=(3.9, 2.8), xytext=(3.5, 2.8), arrowprops=dict(arrowstyle="->", color=COLORS["gray"]))
    ax.text(3.0, 0.8, "fusion keeps semantic matches and exact-token matches alive", ha="center", color=COLORS["gray"], fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.set_yticks([])
    ax.set_xlabel("request time")
    segments = [("dense", 0.5, 2.5, COLORS["good"], 2.1), ("BM25", 0.5, 1.5, COLORS["query"], 1.3), ("merge", 3.1, 1.0, COLORS["warn"], 1.7), ("rerank", 4.4, 2.0, COLORS["purple"], 1.7), ("pack", 6.8, 1.1, COLORS["bad"], 1.7)]
    for label, x, w, color, y in segments:
        ax.add_patch(patches.Rectangle((x, y), w, 0.45, fc=color, ec=color, alpha=0.75))
        ax.text(x + w / 2, y + 0.22, label, ha="center", va="center", fontsize=8, color="white", weight="bold")
    ax.set_title("Parallel search, then serial reranking")
    save(fig, "hybrid-reranking-timeline.svg")


def projection_dashboard_neighbor_edges() -> None:
    rng = np.random.default_rng(5)
    pts = np.vstack([
        rng.normal(loc=(-1.1, 0.8), scale=0.25, size=(15, 2)),
        rng.normal(loc=(1.0, 0.5), scale=0.28, size=(15, 2)),
        rng.normal(loc=(0.0, -1.0), scale=0.22, size=(15, 2)),
    ])
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    titles = ["original\nneighbors", "PCA", "UMAP\nseed 1", "UMAP\nseed 2"]
    transforms = [pts, pts @ np.array([[1.0, 0.2], [-0.1, 0.9]]), pts + rng.normal(scale=0.08, size=pts.shape), pts @ np.array([[0.4, -0.9], [0.9, 0.4]])]
    colors = [COLORS["query"]] * 15 + [COLORS["good"]] * 15 + [COLORS["warn"]] * 15
    edges = [(1, 5), (17, 22), (33, 39), (4, 20)]
    for ax, p, title in zip(axs, transforms, titles):
        ax.scatter(p[:, 0], p[:, 1], s=26, c=colors, alpha=0.85)
        for i, j in edges:
            ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color=COLORS["line"], lw=0.8, alpha=0.55)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    save(fig, "projection-dashboard-neighbor-edges.svg")


def concept_direction_analogy() -> None:
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    neg = np.array([-1.2, -0.3])
    pos = np.array([1.1, 0.7])
    pts = np.array([[-0.9, 0.4], [-0.5, -1.0], [0.2, 0.1], [0.9, 1.1], [1.3, -0.4]])
    ax.scatter(pts[:, 0], pts[:, 1], s=55, color="#94a3b8")
    ax.scatter(*neg, s=90, color=COLORS["bad"])
    ax.scatter(*pos, s=90, color=COLORS["good"])
    arrow(ax, tuple(neg), tuple(pos), COLORS["purple"])
    direction = pos - neg
    direction = direction / np.linalg.norm(direction)
    for p in pts:
        t = neg + direction * ((p - neg) @ direction)
        ax.plot([p[0], t[0]], [p[1], t[1]], color="#d1d5db", ls="--", lw=1)
    ax.set_title("Concept score is projection onto a direction")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[0, 1])
    a, b, c = np.array([0.4, 0.7]), np.array([1.3, 1.7]), np.array([2.2, 0.4])
    target = b - a + c
    ax.scatter([a[0], b[0], c[0], target[0]], [a[1], b[1], c[1], target[1]], s=80, c=[COLORS["query"], COLORS["good"], COLORS["warn"], COLORS["purple"]])
    for p, text in [(a, "a"), (b, "b"), (c, "c"), (target, "b - a + c")]:
        ax.text(p[0] + 0.06, p[1] + 0.06, text, fontsize=9)
    ax.plot([a[0], b[0], target[0], c[0], a[0]], [a[1], b[1], target[1], c[1], a[1]], color="#cbd5e1", lw=1.3)
    ax.scatter([target[0] + 0.25], [target[1] - 0.45], s=70, color=COLORS["bad"])
    ax.text(target[0] + 0.32, target[1] - 0.45, "wrong neighbor", fontsize=8, color=COLORS["bad"])
    ax.set_title("Analogy arithmetic proposes a target")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    save(fig, "concept-direction-analogy.svg")


def anisotropy_whitening_spectrum() -> None:
    rng = np.random.default_rng(6)
    raw = rng.normal(size=(220, 2)) @ np.array([[1.9, 0.1], [0.0, 0.35]]) + np.array([2.0, 1.0])
    centered = raw - raw.mean(axis=0)
    whitened = centered @ np.linalg.inv(np.linalg.cholesky(np.cov(centered.T))).T
    fig = plt.figure(figsize=(12, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.32)
    for i, (pts, title) in enumerate([(raw, "raw mean offset"), (centered, "centered"), (whitened, "whitened")]):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color=COLORS["query"], alpha=0.5)
        ax.axhline(0, color="#e5e7eb", lw=0.8)
        ax.axvline(0, color="#e5e7eb", lw=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    ax = fig.add_subplot(gs[0, 3])
    spectrum = np.array([7.0, 3.4, 1.5, 0.8, 0.38, 0.2, 0.12, 0.08])
    ax.plot(np.arange(1, len(spectrum) + 1), spectrum, marker="o", color=COLORS["purple"])
    ax.set_title("singular-value spectrum")
    ax.set_xlabel("component")
    ax.set_ylabel("strength")
    ax.text(4.2, 4.5, "weak tail can\namplify noise", fontsize=8, color=COLORS["bad"])
    save(fig, "anisotropy-whitening-spectrum.svg")


def lm_token_embedding_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    steps = [("text", 0.4), ("token IDs", 1.9), ("input E rows", 3.55), ("+ positions", 5.35), ("transformer\nblocks", 7.0), ("output logits", 9.0), ("decode", 10.7)]
    for text, x in steps:
        label_box(ax, x, 2.25, 1.15, 0.75, text, fc="#ffffff", ec=COLORS["query"], color=COLORS["query"], weight="bold")
    for (_, x1), (_, x2) in zip(steps, steps[1:]):
        ax.annotate("", xy=(x2, 2.62), xytext=(x1 + 1.15, 2.62), arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.3))
    ax.plot([4.1, 9.6], [1.8, 1.8], color=COLORS["purple"], lw=1.6)
    ax.annotate("", xy=(4.1, 2.25), xytext=(4.1, 1.8), arrowprops=dict(arrowstyle="->", color=COLORS["purple"]))
    ax.annotate("", xy=(9.6, 2.25), xytext=(9.6, 1.8), arrowprops=dict(arrowstyle="->", color=COLORS["purple"]))
    ax.text(6.85, 1.35, "with tied weights, the same vocabulary table participates in input and output scoring", ha="center", color=COLORS["purple"], fontsize=9)
    ax.text(6.0, 4.0, "vocabulary edits are model, tokenizer, and decoding edits", ha="center", color=COLORS["bad"], fontsize=10, weight="bold")
    save(fig, "lm-token-embedding-pipeline.svg")


def rag_context_budget() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    rels = [("answer", COLORS["good"], 0.9), ("near miss", COLORS["warn"], 0.7), ("lexical distractor", COLORS["bad"], 0.8), ("background", "#cbd5e1", 0.55), ("duplicate", "#e5e7eb", 0.5)]
    for i, (label, color, width) in enumerate(rels):
        label_box(ax, 0.7, 4.45 - i * 0.75, 2.5, 0.5, f"rank {i+1}: {label}", fc=color, ec=color, color="white" if color != "#e5e7eb" else COLORS["line"], weight="bold")
        ax.annotate("", xy=(5.0, 4.7 - i * 0.75), xytext=(3.25, 4.7 - i * 0.75), arrowprops=dict(arrowstyle="->", color=COLORS["gray"]))
        ax.add_patch(patches.Rectangle((5.0, 4.48 - i * 0.75), width * 2.0, 0.38, fc=color, ec=color, alpha=0.75))
    ax.add_patch(patches.Rectangle((5.0, 0.75), 5.2, 4.0, fill=False, ec=COLORS["query"], lw=2))
    ax.text(7.6, 4.95, "fixed context window budget", ha="center", weight="bold", color=COLORS["query"])
    ax.axhline(2.0, xmin=0.42, xmax=0.85, color=COLORS["bad"], lw=2, ls="--")
    ax.text(10.45, 1.85, "cutoff", color=COLORS["bad"], weight="bold")
    ax.text(7.6, 0.35, "top-k is a budget decision, not only a retrieval decision", ha="center", color=COLORS["gray"], fontsize=9)
    save(fig, "rag-context-budget.svg")


def recommender_norm_bias_funnel() -> None:
    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    users = np.array([[0.0, 0.0], [0.2, 0.15]])
    item_pop = np.array([2.3, 0.45])
    item_niche = np.array([1.0, 0.95])
    for u in users:
        ax.scatter(*u, marker="*", s=150, color=COLORS["query"])
    arrow(ax, (0, 0), tuple(item_pop), COLORS["warn"])
    arrow(ax, (0, 0), tuple(item_niche), COLORS["good"])
    ax.text(item_pop[0] - 0.1, item_pop[1] + 0.2, "high-norm\npopular item", color=COLORS["warn"], fontsize=8)
    ax.text(item_niche[0] + 0.1, item_niche[1] + 0.05, "aligned\nniche item", color=COLORS["good"], fontsize=8)
    ax.set_title("Dot product mixes alignment and item norm")
    ax.set_xlim(-0.2, 2.8)
    ax.set_ylim(-0.25, 1.55)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4.5)
    widths = [3.2, 2.25, 1.35, 0.72]
    labels = ["all items", "ANN candidates", "reranked", "shown"]
    for i, (w, label) in enumerate(zip(widths, labels)):
        x = (4 - w) / 2
        y = 3.8 - i * 0.85
        ax.add_patch(patches.Rectangle((x, y), w, 0.5, fc=COLORS["purple"], alpha=0.25 + i * 0.15, ec=COLORS["purple"]))
        ax.text(2, y + 0.25, label, ha="center", va="center", fontsize=9)
    ax.text(2, 0.65, "recall lost in the first stage\ncannot be recovered later", ha="center", color=COLORS["bad"], fontsize=9)
    save(fig, "recommender-norm-bias-funnel.svg")


def graph_entity_embeddings() -> None:
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    nodes = {"A": (0.8, 2.7), "B": (1.8, 3.4), "C": (2.2, 2.4), "D": (4.2, 3.2), "E": (5.0, 2.3), "F": (4.0, 1.8)}
    edges = [("A", "B"), ("A", "C"), ("B", "C"), ("D", "E"), ("D", "F"), ("E", "F"), ("C", "D")]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color="#cbd5e1", lw=1.2)
    for n, p in nodes.items():
        ax.scatter(*p, s=150, color=COLORS["query"] if n in "ABC" else COLORS["good"], edgecolor="white", linewidth=1)
        ax.text(p[0], p[1], n, ha="center", va="center", color="white", weight="bold")
    ax.set_title("Graph neighborhoods and bridges")
    ax.set_xlim(0, 5.8)
    ax.set_ylim(1.1, 3.9)

    ax = fig.add_subplot(gs[0, 1])
    pts = np.array([[-1.0, 0.6], [-0.7, 1.0], [-0.45, 0.35], [0.8, 0.8], [1.1, 0.35], [0.55, 0.15]])
    ax.scatter(pts[:3, 0], pts[:3, 1], s=90, color=COLORS["query"], label="community 1")
    ax.scatter(pts[3:, 0], pts[3:, 1], s=90, color=COLORS["good"], label="community 2")
    h = np.array([-0.8, -0.9])
    r = np.array([1.2, 0.55])
    t = h + r
    arrow(ax, tuple(h), tuple(t), COLORS["purple"])
    ax.scatter([h[0], t[0]], [h[1], t[1]], s=80, color=[COLORS["warn"], COLORS["bad"]])
    ax.text(-0.95, -1.1, "head", fontsize=8)
    ax.text(t[0] + 0.06, t[1], "tail", fontsize=8)
    ax.text(0.05, -0.45, "relation vector", color=COLORS["purple"], fontsize=8)
    ax.set_title("Entity relation as translation")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "graph-entity-embeddings.svg")


def matrix_factorization_rank_tradeoff() -> None:
    fig = plt.figure(figsize=(11.5, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 4.8)
    draw_matrix(ax, 0.4, 3.9, rows=5, cols=5, cell=0.42, label="Sparse interactions", color=COLORS["query"], highlight_row=1)
    ax.text(2.8, 2.9, "~", fontsize=20, weight="bold")
    draw_matrix(ax, 3.4, 3.9, rows=5, cols=2, cell=0.42, label="Users", color=COLORS["good"], highlight_row=1)
    ax.text(4.55, 2.9, "x", fontsize=16, weight="bold")
    draw_matrix(ax, 5.0, 3.2, rows=2, cols=5, cell=0.42, label="Items", color=COLORS["warn"])
    ax.annotate("one cell is reconstructed\nby a user-item dot product", xy=(2.0, 3.48), xytext=(3.6, 0.85), ha="center", color=COLORS["purple"], arrowprops=dict(arrowstyle="->", color=COLORS["purple"]))

    ax = fig.add_subplot(gs[0, 1])
    ranks = np.array([4, 8, 16, 32, 64, 128])
    error = np.array([0.52, 0.41, 0.30, 0.22, 0.18, 0.16])
    memory = np.array([0.08, 0.12, 0.2, 0.34, 0.62, 1.0])
    ax.plot(memory, 1 - error, marker="o", color=COLORS["purple"])
    for x, y, r in zip(memory, 1 - error, ranks):
        ax.text(x, y + 0.015, f"r={r}", ha="center", fontsize=8)
    ax.set_xlabel("relative memory")
    ax.set_ylabel("reconstruction quality")
    ax.set_title("Rank is a capacity-cost knob")
    ax.grid(True, color="#e5e7eb")
    save(fig, "matrix-factorization-rank-tradeoff.svg")


def probabilistic_softmax_candidates() -> None:
    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    label_box(ax, 0.4, 2.6, 1.1, 0.55, "query", fc="#eff6ff", ec=COLORS["query"], color=COLORS["query"])
    for i, (text, y, color) in enumerate([("positive", 3.1, COLORS["good"]), ("negative", 2.2, COLORS["bad"]), ("negative", 1.3, COLORS["bad"])]):
        label_box(ax, 2.0, y, 1.35, 0.45, text, fc="#ffffff", ec=color, color=color)
        ax.annotate("", xy=(2.0, y + 0.22), xytext=(1.5, 2.88), arrowprops=dict(arrowstyle="->", color=COLORS["gray"]))
        label_box(ax, 4.0, y, 1.1, 0.45, f"logit {i}", fc="#f8fafc", ec=COLORS["purple"], color=COLORS["purple"])
    ax.text(4.55, 0.8, "softmax turns\nrelative logits\ninto probabilities", ha="center", fontsize=9, color=COLORS["purple"])

    ax = fig.add_subplot(gs[0, 1])
    labels = ["same positive\nwith easy negatives", "same positive\nwith hard negatives"]
    probs = [[0.82, 0.09, 0.05, 0.04], [0.43, 0.29, 0.18, 0.10]]
    bottom = np.zeros(2)
    colors = [COLORS["good"], COLORS["bad"], "#fca5a5", "#fecaca"]
    for j in range(4):
        vals = [probs[0][j], probs[1][j]]
        ax.bar(labels, vals, bottom=bottom, color=colors[j], width=0.5)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("softmax probability mass")
    ax.set_title("Probability depends on the candidate set")
    save(fig, "probabilistic-softmax-candidates.svg")


def experiment_matrix_pareto() -> None:
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    variants = ["baseline", "normalize", "ANN tuned", "rerank"]
    metrics = ["recall", "MRR", "latency", "memory", "subgroup"]
    values = np.array([[0.62, 0.55, 0.35, 0.45, 0.50], [0.68, 0.59, 0.35, 0.45, 0.61], [0.66, 0.57, 0.75, 0.55, 0.58], [0.74, 0.70, 0.48, 0.55, 0.69]])
    for i, variant in enumerate(variants):
        ax.text(0.9, 3.7 - i * 0.65, variant, ha="right", va="center", fontsize=9)
    for j, metric in enumerate(metrics):
        ax.text(1.35 + j * 0.8, 4.1, metric, ha="center", va="center", fontsize=8, weight="bold")
    for i in range(len(variants)):
        for j in range(len(metrics)):
            color = plt.cm.Greens(values[i, j])
            ax.add_patch(patches.Rectangle((1.05 + j * 0.8, 3.45 - i * 0.65), 0.62, 0.45, fc=color, ec="#cbd5e1"))
    ax.add_patch(patches.Rectangle((1.05, 3.45 - 3 * 0.65), 0.62 * 5 + 0.18 * 4, 0.45, fill=False, ec=COLORS["purple"], lw=2))
    ax.text(3.0, 0.6, "choose variants by trade-off, not one metric alone", ha="center", color=COLORS["gray"], fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    latency = np.array([20, 25, 12, 38, 16, 45])
    recall = np.array([0.62, 0.68, 0.58, 0.74, 0.66, 0.77])
    names = ["base", "norm", "fast ANN", "rerank", "ANN tuned", "large"]
    ax.scatter(latency, recall, s=80, color=COLORS["query"])
    for x, y, n in zip(latency, recall, names):
        ax.text(x + 0.5, y + 0.005, n, fontsize=8)
    ax.plot([12, 25, 38], [0.58, 0.68, 0.74], color=COLORS["purple"], lw=1.4, ls="--")
    ax.set_xlabel("latency")
    ax.set_ylabel("recall@5")
    ax.set_title("Pareto frontier")
    ax.grid(True, color="#e5e7eb")
    save(fig, "experiment-matrix-pareto.svg")


def main() -> None:
    configure_matplotlib()
    embedding_lookup_to_space()
    vectors_norms_angles_projection()
    metric_comparison()
    vector_length_ranking()
    high_dimensional_concentration_hubs()
    batch_embedding_gradient_rows()
    prediction_objectives_neighborhoods()
    contrastive_ranking_forces()
    factorized_embeddings()
    embedding_memory_accounting()
    canonical_token_collapse()
    quantization_ranking_pq()
    linear_relu_geometry()
    retrieval_rag_pipeline()
    ann_recall_latency()
    evaluation_metrics_ranked_list()
    hybrid_reranking_timeline()
    projection_dashboard_neighbor_edges()
    concept_direction_analogy()
    anisotropy_whitening_spectrum()
    lm_token_embedding_pipeline()
    rag_context_budget()
    recommender_norm_bias_funnel()
    graph_entity_embeddings()
    matrix_factorization_rank_tradeoff()
    probabilistic_softmax_candidates()
    experiment_matrix_pareto()
    print(f"Wrote SVG figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
