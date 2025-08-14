# plotting/similarity_breakpoints_plotter.py
from __future__ import annotations
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from ..data_plotter.data_plotter import DataPlotter


class SimilarityBreakpointsPlotter(DataPlotter):
    """
    Plotter for visualizing cosine distances between adjacent sentence embeddings
    and the chosen breakpoints used for chunking.
    """

    def plot_distances_with_breakpoints(
        self,
        distances: np.ndarray,
        breakpoints: List[int],   # kept for API compatibility, no longer used for vertical lines
        threshold: float,
        *,
        title: str = "Cosine distances (with threshold highlights)",
        filename: str = "distances_threshold_points.png",
        show: bool = False,
        with_timestamp: bool = True,
    ) -> str:
        """
        Plot cosine distances as a line; highlight points above threshold in red and draw a colored threshold line.
        No vertical lines are drawn.
        """
        nm1 = int(len(distances))
        xs = np.arange(nm1)

        # adaptive marker size for large series
        if   nm1 <= 500:  msize = 3
        elif nm1 <= 5000: msize = 2
        else:             msize = 1

        fig, ax = plt.subplots(figsize=(12, 4))

        # base line
        ax.plot(xs, distances, marker="o", linewidth=1.2, markersize=msize, label="cosine distance")

        # colored horizontal threshold
        ax.axhline(threshold, color="tab:orange", linestyle="--", linewidth=1.2, label=f"threshold = {threshold:.3g}")

        # highlight points above threshold in red
        mask = np.asarray(distances) > float(threshold)
        if np.any(mask):
            ax.scatter(xs[mask], np.asarray(distances)[mask], s=(msize+1)**2, color="red", label="above threshold", zorder=3)

        ax.set_title(title)
        ax.set_xlabel("Sentence pair index (i = between sentence i and i+1)")
        ax.set_ylabel("Cosine distance")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(loc="best")

        if show:
            plt.show()

        return self.save_figure(fig, filename, with_timestamp=with_timestamp)


    def plot_chunk_boundaries(
        self,
        num_sentences: int,
        chunk_boundaries: List[int],
        *,
        title: str = "Final chunk boundaries (sentence indices)",
        filename: str = "chunk_boundaries.png",
        show: bool = False,
        with_timestamp: bool = True,
    ) -> str:
        """
        Visualize final chunk boundaries on a horizontal line.
        A boundary at index i means a chunk ends at sentence i and the next starts at i+1.

        Args:
            num_sentences: total number of sentences after combining.
            chunk_boundaries: sorted indices of boundaries.
            title: plot title.
            filename: output file name.
            show: if True, display window.
            with_timestamp: append timestamp to file name.

        Returns:
            Absolute file path of the saved figure.
        """
        fig, ax = plt.subplots(figsize=(12, 1.8))
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, num_sentences - 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Sentence index")
        ax.set_title(title)
        ax.hlines(0.5, -0.5, num_sentences - 0.5, linewidth=2)

        for bp in chunk_boundaries:
            ax.vlines(bp + 0.5, 0.2, 0.8, colors="red", linestyles=":", linewidth=1.5)

        ax.grid(True, axis="x", linestyle=":", alpha=0.5)

        if show:
            plt.show()

        return self.save_figure(fig, filename, with_timestamp=with_timestamp)
