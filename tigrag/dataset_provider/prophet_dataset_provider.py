# datasets_provider/prophet.py
import os
import json
import logging
from typing import Optional, Union, List, Dict

import requests
import pandas as pd

from ..dataset_provider.dataset_provider import DatasetProvider


class ProphetDatasetProvider(DatasetProvider):
    """
    Loads the PROPHET dataset (L1) from GitHub, caches it locally as JSON,
    and provides it as a pandas DataFrame in self.datasets.
    """

    RAW_URL = "https://raw.githubusercontent.com/TZWwww/PROPHET/main/data_2024-8/dataset_L1.json"
    LOCAL_FILENAME = "dataset_L1.json"
    DATASET_KEY = "prophet_L1"

    def _to_dataframe(self, data: Union[List, Dict]) -> pd.DataFrame:
        """
        Convert the JSON structure into a pandas DataFrame.
        Handles common cases (list of dicts, dict with 'data' key, etc.).
        """
        if isinstance(data, list):
            if len(data) == 0:
                return pd.DataFrame()
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
            return pd.DataFrame({"value": data})

        if isinstance(data, dict):
            for k in ("data", "items", "examples", "records"):
                if k in data and isinstance(data[k], list):
                    inner = data[k]
                    if len(inner) == 0:
                        return pd.DataFrame()
                    if isinstance(inner[0], dict):
                        return pd.DataFrame(inner)
                    return pd.DataFrame({k: inner})

            if all(isinstance(v, list) for v in data.values()):
                lengths = {len(v) for v in data.values()}
                if len(lengths) == 1:
                    return pd.DataFrame(data)

            return pd.DataFrame([{"key": k, "value": v} for k, v in data.items()])

        return pd.DataFrame([{"value": data}])

    def load(self, max_rows: Optional[int] = None) -> None:
        """
        Load the PROPHET L1 dataset.
        - If cached locally, load from disk.
        - Otherwise, download from GitHub and save locally.
        - Optionally limit the number of rows kept in memory.
        """
        json_path = os.path.join(self.save_dir, self.LOCAL_FILENAME)
        df = None

        # Load from local cache if available
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                df = self._to_dataframe(data)
                logging.info(f"✓ {self.DATASET_KEY} loaded locally: {df.shape}")
            except Exception as e:
                logging.info(f"✗ Failed to load {self.DATASET_KEY} locally: {e}")
                df = None

        # Otherwise, download from GitHub
        if df is None:
            try:
                response = requests.get(self.RAW_URL, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    df = self._to_dataframe(data)

                    os.makedirs(self.save_dir, exist_ok=True)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    logging.info(
                        f"✓ {self.DATASET_KEY} downloaded and saved: {df.shape}"
                    )
                else:
                    logging.info(
                        f"✗ Failed to download {self.DATASET_KEY}: HTTP {response.status_code}"
                    )
                    return
            except Exception as e:
                logging.info(f"✗ Error downloading {self.DATASET_KEY}: {e}")
                return

        # Optionally limit rows in memory
        if df is not None:
            if max_rows is not None:
                if max_rows <= 0:
                    logging.info(
                        f"↷ Skipping row limit for {self.DATASET_KEY}: max_rows={max_rows} (must be > 0)"
                    )
                else:
                    df = df.iloc[:max_rows].copy()
                    logging.info(
                        f"→ {self.DATASET_KEY} limited to first {len(df)} rows"
                    )

            self.datasets[self.DATASET_KEY] = df
