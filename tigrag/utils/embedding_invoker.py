import logging
import os
import pickle
import hashlib
import re
import string
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import json, time, random
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError
from math import exp
import torch

# Bedrock
import boto3

# Ensure NLTK stopwords are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Define stopword sets
EN_STOPWORDS = set(stopwords.words("english"))
DE_STOPWORDS = set(stopwords.words("german"))
ALL_STOPWORDS = EN_STOPWORDS | DE_STOPWORDS

# Precompiled regex patterns
PUNCTUATION_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")
NUM_WS_REGEX = re.compile(r"\d+|\s+")


class EmbeddingPreprocessor:
    def __init__(self):
        self.stop_words = ALL_STOPWORDS

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text).lower()
        text = PUNCTUATION_REGEX.sub("", text)
        text = NUM_WS_REGEX.sub(" ", text).strip()
        return " ".join(w for w in text.split() if w not in self.stop_words)


class EmbeddingInvoker:
    """
    Embedding invoker with a registry mapping: model_name -> (init_fn, call_fn)
    - 'local' uses SentenceTransformer 'paraphrase-MiniLM-L6-v2'
    - 'titan' uses Amazon Bedrock Titan Embeddings (amazon.titan-embed-text-v2:0)
    """

    _loaded_models: Dict[str, Any] = {}

    def __init__(self, model_name: str = "local", cache_dir: str = "."):
        # Registry
        self.models: Dict[str, Tuple[Callable[..., None], Callable[..., Any]]] = {
            "local": (self.init_paraphrase_model, self.call_paraphrase),
            "titan": (self.init_titan_model, self.call_titan_model),
        }

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")

        self.model_name = model_name
        self.init_fn, self.call_fn = self.models[model_name]

        self.preprocessor = EmbeddingPreprocessor()
        self.cache_path = os.path.join(cache_dir, f"embedding_cache_{model_name.replace('/', '_')}.pkl")
        self.cache = self._load_cache()

        # Initialize selected model (no-op for Titan/Bedrock)
        self.init_fn()

    # ---------------------------
    # Public API
    # ---------------------------
    def __call__(self, *args, **kwargs):
        # get and remove 'sentences' from kwargs to avoid duplication
        sentences = kwargs.pop("sentences", None)
        if not sentences:
            raise ValueError('Missing argument "sentences".')

        # preprocess
        if isinstance(sentences, list):
            cleaned = [self.preprocessor.clean(s) for s in sentences]
        else:
            cleaned = self.preprocessor.clean(sentences)

        # cache
        cache_key = self._generate_cache_key(cleaned)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # forward WITHOUT the original 'sentences' in kwargs
        result = self.call_fn(sentences=cleaned, **kwargs)

        self.cache[cache_key] = result
        self._save_cache()
        return result


    # ---------------------------
    # Cache helpers
    # ---------------------------
    def _generate_cache_key(self, sentences):
        if isinstance(sentences, list):
            text = "|||".join(sentences)
        else:
            text = sentences
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    def _save_cache(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logging.warning(f"Error saving embedding cache: {e}")

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Error loading embedding cache: {e}")
        return {}

    # ---------------------------
    # Local paraphrase (SentenceTransformers)
    # ---------------------------
    def init_paraphrase_model(self):
        if "paraphrase" not in EmbeddingInvoker._loaded_models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Loading embedding model on {device.upper()}..."
                         f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
            model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)
            EmbeddingInvoker._loaded_models["paraphrase"] = model
        self.model = EmbeddingInvoker._loaded_models["paraphrase"]

    def call_paraphrase(self, *_, sentences, **__):
        """
        Returns:
          - single string input -> vector: List[float]
          - list input          -> List[List[float]]
        """
        # SentenceTransformer.encode accepts str or List[str]
        emb = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
        return emb.tolist()

    # ---------------------------
    # Bedrock Titan
    # ---------------------------
    def init_titan_model(self):
        # No local initialization required for Bedrock
        return

    def call_titan_model(
        self,
        *,
        sentences,
        region: str = "eu-west-1",
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,      # valid: 256, 512, 1024
        normalize: bool = True,
        max_chars: int = 50000,
        max_workers: int = 20,
        max_retries: int = 5, 
        **__
    ):
        """
        Create embeddings with Amazon Titan Embeddings v2 on Bedrock.

        Args:
        sentences: str or List[str] (already preprocessed upstream)
        region: AWS region for Bedrock Runtime
        model_id: Titan model id
        dimensions: output vector size (256, 512, 1024)
        normalize: whether to L2-normalize on the server
        max_chars: truncate overly long inputs to avoid validation errors

        Returns:
        - str input  -> List[float]
        - list input -> List[List[float]]
        """

        if dimensions not in (256, 512, 1024):
            raise ValueError("Titan v2 unterstützt 256, 512 oder 1024 Dimensionen.")

        client = boto3.client("bedrock-runtime", region_name=region)

        def _zero_vec() -> list:
            return [0.0] * dimensions

        def _embed_one(text: str) -> list:
            if not text:
                return _zero_vec()

            payload = {
                "inputText": text[:max_chars],
                "dimensions": dimensions,
                "normalize": normalize,
            }

            # Retries mit expon. Backoff + Jitter
            delay = 0.5
            for attempt in range(1, max_retries + 1):
                try:
                    resp = client.invoke_model(
                        modelId=model_id,
                        body=json.dumps(payload),
                        accept="application/json",
                        contentType="application/json",
                    )
                    data = json.loads(resp["body"].read())
                    return data["embedding"]
                except ClientError as e:
                    code = e.response.get("Error", {}).get("Code", "")
                    # typische, temporäre Fehler
                    transient = {
                        "ThrottlingException",
                        "TooManyRequestsException",
                        "ServiceQuotaExceededException",
                        "LimitExceededException",
                        "RequestTimeout",
                        "InternalServerException",
                    }
                    if code in transient and attempt < max_retries:
                        time.sleep(delay + random.random() * 0.25)
                        delay = min(8.0, delay * 2.0)  # Deckelung
                        continue
                    raise
            return _zero_vec()

        # Einzeltext
        if not isinstance(sentences, list):
            return _embed_one(sentences)

        # Liste -> parallel, Reihenfolge bewahren
        results = [None] * len(sentences)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_embed_one, s): i for i, s in enumerate(sentences)}
            for fut in tqdm(as_completed(futures), total=len(sentences), desc="Creating Titan embeddings", unit="sent"):
                i = futures[fut]
                results[i] = fut.result()
        return results
