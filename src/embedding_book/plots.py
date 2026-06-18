from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_vectors(vectors: dict[str, np.ndarray], title: str = "Vectors"):
    """
    Plot 2D vectors from the origin.

    This helper intentionally stays simple so chapter visuals are easy to read.
    """
    fig, ax = plt.subplots()
    for label, vector in vectors.items():
        ax.arrow(0, 0, vector[0], vector[1], head_width=0.04, length_includes_head=True)
        ax.text(vector[0], vector[1], label)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    return fig, ax
