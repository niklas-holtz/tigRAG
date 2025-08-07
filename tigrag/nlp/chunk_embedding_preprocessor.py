import os
import sys
import subprocess

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import spacy
import pytextrank

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect
import nltk

# Ensure NLTK data is available
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# Ensure spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load spaCy pipeline and add TextRank
_spacy_nlp = spacy.load("en_core_web_sm")
_spacy_nlp.add_pipe("textrank", last=True)


class ChunkEmbeddingPreprocessor:
    def __init__(self, method="tfidf", **method_kwargs):
        """
        :param method: 'tfidf', 'textrank', or 'combined'
        :param method_kwargs: method-specific parameters like:
            - n_keywords: int
            - n_phrases: int
        """
        self.method = method
        self.method_kwargs = method_kwargs
        self._identifier = method_kwargs.pop('identifier', 'text')

        self.method_dispatch = {
            "tfidf": self._tfidf_keywords
        }

        if method not in self.method_dispatch:
            raise ValueError(f"Unknown method: {method}")

        self._vectorizer = None
        self._tfidf_matrix = None
        self._feature_names = None

    def run(self, chunk):
        """
        Processes a single chunk and returns a string suitable for embedding.
        """
        text = chunk.get(self._identifier, "")
        return self.method_dispatch[self.method](text, **self.method_kwargs)

    def _tfidf_keywords(self, text, corpus=None, n_keywords=15, reverse=True):
        if corpus is None:
            raise ValueError("TF-IDF preprocessing requires a full corpus.")

        if self._tfidf_matrix is None or self._vectorizer is None:
            processed_corpus = [self._preprocess(doc) for doc in corpus]

            self._vectorizer = TfidfVectorizer()
            self._tfidf_matrix = self._vectorizer.fit_transform(processed_corpus)
            self._feature_names = self._vectorizer.get_feature_names_out()

            self._text_index_map = {
                doc: i for i, doc in enumerate(processed_corpus)
            }

        processed_text = self._preprocess(text)
        index = self._text_index_map.get(processed_text)

        if index is not None:
            row = self._tfidf_matrix[index].toarray()[0]
        else:
            tfidf_vector = self._vectorizer.transform([processed_text])
            row = tfidf_vector.toarray()[0]

        sorted_indices = row.argsort()[::-1]
        top_indices = sorted_indices[:n_keywords]
        keywords = [self._feature_names[i] for i in top_indices if row[i] > 0]

        if reverse:
            keywords = list(reversed(keywords))

        doc = _spacy_nlp(text)
        entities = [ent.text for ent in doc.ents]

        combined = list(dict.fromkeys(entities + keywords))
        return ", ".join(combined)

    def _detect_language(self, text):
        try:
            lang_code = detect(text)
            if lang_code.startswith('de'):
                return 'german'
            elif lang_code.startswith('en'):
                return 'english'
        except:
            pass

        return "german" if any(ch in text.lower() for ch in "äöüß") else "english"

    def _preprocess(self, text):
        if not text:
            return ""

        lang = self._detect_language(text)
        stemmer = SnowballStemmer(lang)
        stop_words = set(stopwords.words(lang))

        tokens = word_tokenize(text.lower())
        tokens = [w for w in tokens if w.isalpha() and w not in stop_words]

        if not tokens:
            return ""

        return " ".join([stemmer.stem(w) for w in tokens])
