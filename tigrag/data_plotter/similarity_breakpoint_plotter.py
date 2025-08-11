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
        breakpoints: List[int],
        threshold: float,
        *,
        title: str = "Cosine distances & breakpoints",
        filename: str = "distances_breakpoints.png",
        show: bool = False,
        with_timestamp: bool = True,
    ) -> str:
        """
        Plot cosine distances as a line with the percentile threshold and vertical
        lines for each breakpoint. Save under `<data_path>/plots`.

        Args:
            distances: shape (N-1,) where N is #sentences.
            breakpoints: indices i where a break occurs (between sentence i and i+1).
            threshold: percentile threshold used to mark raw breakpoints.
            title: plot title.
            filename: output file name (extension optional; defaults to .png).
            show: if True, display window (useful in notebooks).
            with_timestamp: append timestamp to file name for uniqueness.

        Returns:
            Absolute file path of the saved figure.
        """
        nm1 = int(len(distances))
        xs = np.arange(nm1)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(xs, distances, marker="o", linewidth=1.5, markersize=3, label="cosine distance")
        ax.axhline(threshold, linestyle="--", linewidth=1.0, label=f"threshold={threshold:.3f}")

        # Draw vertical lines at breakpoints
        for bp in breakpoints:
            if 0 <= bp < nm1:
                ax.axvline(bp, color="red", linestyle=":", linewidth=1.0)

        ax.set_title(title)
        ax.set_xlabel("Sentence pair index (i means between sentence i and i+1)")
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
