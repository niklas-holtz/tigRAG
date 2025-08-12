import os
import sys
import subprocess

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import spacy
import pytextrank
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect
import nltk
from rake_nltk import Rake

import yake

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
        self.method = 'rake'
        self.method_kwargs = method_kwargs
        self._identifier = method_kwargs.pop('identifier', 'text')

        self.method_dispatch = {
            "tfidf": self._tfidf_keywords,
            "yake": self._yake_keywords,
            "rake": self._rake_keywords 
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

    def _topk_from_csr_row(self, csr_row, feature_names, k):
        # csr_row ist 1xV CSR-Matrix
        start, end = csr_row.indptr[0], csr_row.indptr[1]
        idx = csr_row.indices[start:end]
        val = csr_row.data[start:end]
        if val.size == 0:
            return []
        k = min(k, val.size)
        topk_local = np.argpartition(val, -k)[-k:]  # unsortiertes Top-k
        order = np.argsort(-val[topk_local])  # absteigend sortieren
        top = [feature_names[idx[i]] for i in topk_local[order] if val[i] > 0]
        return top

    def _tfidf_keywords(self, text, corpus=None, n_keywords=15, reverse=True):
        if corpus is None:
            raise ValueError("TF-IDF preprocessing requires a full corpus.")

        if self._tfidf_matrix is None or self._vectorizer is None:
            processed_corpus = [self._preprocess(doc) for doc in corpus]
            self._vectorizer = TfidfVectorizer(
                dtype=np.float32,  # halber Speicher, schneller
                max_features=200_000,  # begrenzen, je nach Bedarf
                min_df=2,  # Rauschen kappen
                stop_words='english'        # ggf. passend zur Sprache setzen
                # ngram_range=(1,2),           # nur wenn sinnvoll
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(processed_corpus)
            self._feature_names = self._vectorizer.get_feature_names_out()

            self._text_index_map = {
                doc: i for i, doc in enumerate(processed_corpus)
            }

        processed_text = self._preprocess(text)
        index = self._text_index_map.get(processed_text)

        if index is not None:
            row = self._tfidf_matrix[index]  # 1xV CSR
        else:
            row = self._vectorizer.transform([processed_text])  # 1xV CSR

        keywords = self._topk_from_csr_row(row, self._feature_names, n_keywords)
        if reverse:
            keywords = list(reversed(keywords))

        doc = _spacy_nlp(text)
        entities = [ent.text for ent in doc.ents]
        combined = list(dict.fromkeys(entities + keywords))
        return ", ".join(combined)

    # ---------------- RAKE ----------------
    def _rake_keywords(
        self,
        text: str,
        n_keywords: int = 15,
        include_entities: bool = False,
        language: str = 'english',
        corpus=None
    ) -> str:
        """
        Extrahiert Keywords/Keyphrases aus EINEM Dokument mit RAKE.
        - Schnell, keine Korpora nötig.
        - Liefert bevorzugt Mehrwort-Phrasen.
        Params:
            n_keywords: Anzahl Top-Phrasen/Wörter
            include_entities: spaCy-Entities zusätzlich aufnehmen
            language: 'english' | 'german' (auto, wenn None)
        """
        if not text:
            return ""

        # Sprache bestimmen (leichtgewichtig)
        if language is None:
            language = "german" if any(ch in text for ch in "äöüÄÖÜß") else "english"

        # RAKE-Extractor (nutzt NLTK-Stopwörter; in deinem Setup bereits vorhanden)
        r = Rake(language=language)
        r.extract_keywords_from_text(text)
        phrases = r.get_ranked_phrases()[:n_keywords]

        if include_entities:
            doc = _spacy_nlp(text)
            entities = [ent.text for ent in doc.ents]
            phrases = list(dict.fromkeys(entities + phrases))

        return ", ".join(phrases)

    # ---------------- YAKE ----------------

    def _yake_keywords(
        self,
        text,
        n_keywords=15,
        ngram=1,
        deduplication_threshold=0.9,
        include_entities=True,
        lan='en',
        corpus=None
    ):
        """
        Extrahiert Keywords aus EINEM Dokument mit YAKE (ohne Korpus).
        """
        if not text:
            return ""

        # Sprache bestimmen und für YAKE mappen
        if lan is None:
            lang = self._detect_language(text)  # 'english'/'german'
            if lang == 'german':
                lan = 'de'
            else:
                lan = 'en'  # Fallback

        # YAKE-Extractor
        kw_extractor = yake.KeywordExtractor(
            lan=lan,
            n=ngram,
            top=n_keywords,
            dedupLim=deduplication_threshold
        )
        kw_with_scores = kw_extractor.extract_keywords(text)
        # YAKE gibt (phrase, score). Kleinere Scores = besser.
        # Sicherheitshalber sortieren und nur Strings übernehmen:
        kw_with_scores.sort(key=lambda x: x[1])
        yake_keywords = [kw for kw, _ in kw_with_scores[:n_keywords]]

        if include_entities:
            doc = _spacy_nlp(text)
            entities = [ent.text for ent in doc.ents]
            combined = list(dict.fromkeys(entities + yake_keywords))
            return ", ".join(combined)
        else:
            return ", ".join(dict.fromkeys(yake_keywords))

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
