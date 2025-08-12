# datasets_provider/ultra_domain.py
import os
import json
import requests
import pandas as pd
import logging
from typing import Optional, Dict
from ..dataset_provider.dataset_provider import DatasetProvider


class UltraDomainDatasetProvider(DatasetProvider):
    BASE_URL = "https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/"
    FILES = ["agriculture.jsonl", "cooking.jsonl", "history.jsonl", "bioprotocol.jsonl"]

    def load(self) -> None:
        """
        Download or load UltraDomain datasets and store them in self.datasets.
        """
        for file in self.FILES:
            dataset_name = file.replace('.jsonl', '')
            json_path = os.path.join(self.save_dir, f"{dataset_name}.json")

            df = None

            # If already downloaded → load from disk
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    logging.info(f"✓ {dataset_name} loaded locally: {df.shape}")
                except Exception as e:
                    logging.info(f"✗ Failed to load {dataset_name}: {e}")
                    continue
            else:
                # Download from HF and save as JSON
                try:
                    url = self.BASE_URL + file
                    response = requests.get(url, timeout=30)

                    if response.status_code == 200:
                        lines = response.text.strip().split('\n')
                        data = [json.loads(line) for line in lines if line.strip()]
                        if not data:
                            logging.info(f"✗ {dataset_name} is empty.")
                            continue
                        df = pd.DataFrame(data)

                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)

                        logging.info(f"✓ {dataset_name} downloaded and saved: {df.shape}")
                    else:
                        logging.info(f"✗ Failed to download {dataset_name}: HTTP {response.status_code}")
                        continue
                except Exception as e:
                    logging.info(f"✗ Error downloading {file}: {e}")
                    continue

            if df is not None:
                self.datasets[dataset_name] = df
