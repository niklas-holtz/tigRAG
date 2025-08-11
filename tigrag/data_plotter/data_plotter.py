# plotting/data_plotter.py
from __future__ import annotations
import os
from datetime import datetime
from typing import Optional
import matplotlib.pyplot as plt


class DataPlotter:
    """
    Base class for plotters that know a data path and write figures to `<data_path>/plots`.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.plots_dir = os.path.join(self.data_path, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def _ts(self) -> str:
        """Return a compact timestamp for file names."""
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        *,
        with_timestamp: bool = True,
        dpi: int = 150,
    ) -> str:
        """
        Save a Matplotlib figure into `<data_path>/plots` and return the absolute file path.
        If `with_timestamp` is True, append a timestamp to the file name stem.
        """
        stem, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"
        if with_timestamp:
            stem = f"{stem}_{self._ts()}"
        path = os.path.join(self.plots_dir, stem + ext)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path
