# datasets_provider/base.py
import os
from typing import Dict, Any
import pandas as pd
from abc import ABC, abstractmethod


class DatasetProvider(ABC):
    """
    Abstract base class for dataset providers.
    Handles common attributes such as save_dir and loaded datasets.
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.datasets: Dict[str, pd.DataFrame] = {}

    @abstractmethod
    def load(self) -> None:
        """Load (and possibly download) the datasets into self.datasets."""
        pass

    def get(self, name: str, column: str = None) -> Any:
        """
        Retrieve a dataset or a specific column from a dataset.
        Args:
            name: dataset name key
            column: optional column name to extract
        Returns:
            DataFrame or list
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' is not loaded. Available: {list(self.datasets.keys())}")

        df = self.datasets[name]
        if column is None:
            return df
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset '{name}'.")
        return df[column].dropna().tolist()
