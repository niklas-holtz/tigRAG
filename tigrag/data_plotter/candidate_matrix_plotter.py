import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ..data_plotter.data_plotter import DataPlotter


class CandidateMatrixPlotter(DataPlotter):
    """
    Plots a similarity candidate matrix as a heatmap.
    Inherits from DataPlotter to save plots into `<data_path>/plots`.
    """

    def plot_matrix(
        self,
        matrix: np.ndarray,
        title: str = "Candidate Similarity Matrix",
        filename: str = "candidate_matrix_heatmap.png",
        with_timestamp: bool = True,
        show: bool = False
    ) -> str:
        """
        Plots the similarity matrix as a heatmap and saves it.

        Args:
            matrix: 2D numpy array (NxN) with similarity scores (0..1).
            title: Title for the plot.
            filename: Output file name (png).
            with_timestamp: If True, append UTC timestamp to filename.
            show: If True, display the plot interactively.

        Returns:
            Absolute path to the saved file.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square (NxN).")

        logging.info("Plotting candidate matrix heatmap...")

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(matrix, cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)

        ax.set_title(title)
        ax.set_xlabel("Target Event Index")
        ax.set_ylabel("Source Event Index")

        # Add colorbar
        cbar = fig.colorbar(cax)
        cbar.set_label("Similarity Score (cosine)")

        # Optional: tick labels for events
        n = matrix.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(range(n))
        ax.set_yticklabels(range(n))

        # Smaller tick labels for large matrices
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)

        # Save using DataPlotter's method
        path = self.save_figure(fig, filename, with_timestamp=with_timestamp)

        

        if show:
            plt.show()

        return path
