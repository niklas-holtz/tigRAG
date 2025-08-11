# plotting/silhouette_curve_plotter.py
from __future__ import annotations
from typing import Sequence, Optional
import matplotlib.pyplot as plt
import numpy as np
from ..data_plotter.data_plotter import DataPlotter  # your existing base class

class SilhouetteCurvePlotter(DataPlotter):
    """
    Plots the silhouette score as a function of the number of clusters k.
    Saves figures into <data_path>/plots.
    """

    def plot_curve(
        self,
        ks: Sequence[int],
        scores: Sequence[float],
        *,
        best_k: Optional[int] = None,
        title: str = "Silhouette score vs. number of clusters (k)",
        filename: str = "silhouette_curve.png",
        with_timestamp: bool = True,
        show: bool = False,
    ) -> str:
        """Render and save the silhouette curve plot."""
        # Ensure numpy arrays for convenience
        ks = np.asarray(list(ks), dtype=int)
        scores = np.asarray(list(scores), dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ks, scores, marker="o", linewidth=1.5)
        ax.set_xlabel("k (number of clusters)")
        ax.set_ylabel("silhouette score")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Mark the best_k if provided and present in ks
        if best_k is not None and best_k in set(ks.tolist()):
            try:
                idx = int(np.where(ks == best_k)[0][0])
                ax.scatter([best_k], [scores[idx]], s=60, zorder=3)
                ax.annotate(
                    f"best k={best_k}\nscore={scores[idx]:.4f}",
                    xy=(best_k, scores[idx]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", lw=1),
                )
            except Exception:
                pass

        if show:
            plt.show()

        return self.save_figure(fig, filename, with_timestamp=with_timestamp)
