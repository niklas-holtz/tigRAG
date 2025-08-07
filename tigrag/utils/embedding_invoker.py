import logging
import os
import pickle
import hashlib
import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

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
        text = PUNCTUATION_REGEX.sub('', text)
        text = NUM_WS_REGEX.sub(' ', text).strip()
        return ' '.join(w for w in text.split() if w not in self.stop_words)


class EmbeddingInvoker:
    _loaded_models = {}

    def __init__(self, model_name="local", cache_dir: str = "."):
        self.models = {
            "local": (self._embed_with_paraphrase, self._init_paraphase_model)
        }

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        self.model_name = model_name
        self.model_method, self.init_method = self.models[model_name]
        self.preprocessor = EmbeddingPreprocessor()
        self.cache_path = os.path.join(cache_dir, f"embedding_cache_{model_name.replace('/', '_')}.pkl")
        self.cache = self._load_cache()

        if self.init_method:
            self.init_method()

    def __call__(self, *args, **kwargs):
        sentences = kwargs.get("sentences")
        if not sentences:
            raise ValueError('Missing argument "sentences".')

        if isinstance(sentences, list):
            sentences = [self.preprocessor.clean(s) for s in sentences]
        else:
            sentences = self.preprocessor.clean(sentences)

        cache_key = self._generate_cache_key(sentences)
        if cache_key in self.cache:
            return self.cache[cache_key]

        kwargs["sentences"] = sentences
        embedding = self.model_method(*args, **kwargs)
        self.cache[cache_key] = embedding
        self._save_cache()
        return embedding

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

    def _init_paraphase_model(self):
        if 'paraphrase' not in EmbeddingInvoker._loaded_models:
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            EmbeddingInvoker._loaded_models['paraphrase'] = model
        self.model = EmbeddingInvoker._loaded_models['paraphrase']

    def _embed_with_paraphrase(self, *args, **kwargs):
        # Überprüfe, ob 'sentences' im kwargs enthalten ist
        if 'sentences' not in kwargs:
            raise Exception('Missing argument "sentences".')

        # Extrahiere die Sätze
        sentences = kwargs['sentences']

        # Baue die Embeddings mit dem 'paraphrase-MiniLM-L6-v2' Modell
        # logging.info('Building embedding with "paraphrase-MiniLM-L6-v2" model ..')

        # Berechne die Embeddings
        embedding = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)

        # Statt NumPy-Array geben wir den Tensor zurück oder eine Python-Liste
        return embedding.tolist()  # Konvertiere den Tensor in eine Python-Liste