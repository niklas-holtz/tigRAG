import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from ..utils.embedding_invoker import EmbeddingInvoker
from scipy.signal import argrelextrema, find_peaks
from tqdm import tqdm
import logging
from ..nlp.chunk_embedding_preprocessor import ChunkEmbeddingPreprocessor
from ..data_plotter.similarity_breakpoint_plotter import SimilarityBreakpointsPlotter
from bisect import bisect_right, bisect_left
from typing import List, Tuple, Dict, Any, Iterable, Optional


class TextChunker():

    def __init__(self, embedding_func: EmbeddingInvoker, working_dir: str) -> None:
        self.embedding_func = embedding_func
        self.working_dir = working_dir
        # --- Precompiled regex patterns ---
        sentence_end_chars = r"\.\!\?…"  # extend if needed
        closing_quotes = r"\"'»«‚‘’“”)]"  # optional closing punctuation

        # Matches sentences ending in terminator OR last sentence without terminator
        self.SENTENCE_REGEX = re.compile(
            rf"""
            (                                   # group 1: sentence with terminator
              [^{sentence_end_chars}\n]+?       # non-terminator chars
              [{sentence_end_chars}]            # sentence end char
              [{closing_quotes}]*               # optional closing quotes/brackets
            )
            |
            (                                   # group 2: trailing text without terminator
              [^{sentence_end_chars}\n][^\n]*?$ # last sentence without punctuation
            )
            """,
            re.VERBOSE | re.UNICODE | re.MULTILINE
        )

        # Simple abbreviation list to avoid false sentence breaks
        self.COMMON_ABBR = {
            # German
            "z. b.", "z.b.", "bzw.", "bspw.", "vgl.", "ca.", "usw.",
            "u. a.", "u.a.", "dr.", "prof.", "dipl.", "etc.",

            # English titles & honorifics
            "mr.", "mrs.", "ms.", "miss.", "dr.", "prof.", "sr.", "jr.", "rev.", "hon.", "fr.", "st.",

            # Latin/English common
            "e.g.", "eg.", "i.e.", "ie.", "vs.", "etc.", "cf.",

            # Months
            "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",

            # Units of measurement (imperial/metric)
            "in.", "ft.", "yd.", "mi.", "oz.", "lb.", "pt.", "qt.", "gal.",
            "mm.", "cm.", "m.", "km.", "mg.", "g.", "kg.", "ml.", "l.",

            # Time
            "a.m.", "am.", "p.m.", "pm.",

            # Misc common abbreviations
            "no.", "nos.", "vol.", "vols.", "pp.", "chap.", "ch.", "ed.", "eds.",
            "al.", "fig.", "figs.", "dept.", "est.", "gov.", "inc.", "ltd.", "co.",
            "corp.", "univ.", "col.", "mt.", "mts.", "ave.", "blvd.", "rd.", "hwy.", "apt.", "bldg.",
            "misc.", "min.", "sec."
        }

        self.PARA_SPLIT_REGEX = re.compile(
            r"""
        (?:\r\n|\r|\n){2,}                         # blank line(s)
        | ^\s*#{1,6}\s+.*$                         # Markdown heading
        | ^\s*(?:[-*+]\s+|\d+\.\s+)                # Markdown list bullets / numbered
        | ^\s*[-*_]{3,}\s*$                        # Markdown horizontal rule
        | <\s*(?:p|div|section|article|h[1-6]|ul|ol|li|br)\b[^>]*>  # HTML-ish block start
        """,
            re.IGNORECASE | re.MULTILINE | re.VERBOSE
        )

    def _extract_sentences(self, text: str):
        """
        Extract sentences from text with improved regex-based segmentation.
        """
        # Normalize newlines
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        sentences = []
        for m in self.SENTENCE_REGEX.finditer(t):
            sent_text = m.group(1) if m.group(1) else m.group(2)
            if not sent_text:
                continue

            start, end = m.span()
            while start < end and t[start].isspace():
                start += 1
            while end > start and t[end - 1].isspace():
                end -= 1
            if start >= end:
                continue

            sentence_clean = t[start:end]
            tail = sentence_clean.lower().strip()
            if any(tail.endswith(abbr) for abbr in self.COMMON_ABBR) and len(sentence_clean) < 8:
                continue

            sentences.append({
                "sentence": sentence_clean,
                "start": start
            })

        return sentences


    # --- Paragraph detection ------------------------------------------------------

    def detect_paragraph_bounds(self, text: str) -> List[Tuple[int, int]]:
        """
        Return (start, end) character spans for paragraphs.
        More robust than just r'\n\s*\n': recognizes Markdown/HTML-ish boundaries.
        Complexity: O(N) in text length.
        """
        # Normalize newlines to '\n' to keep offsets consistent
        t = text.replace('\r\n', '\n').replace('\r', '\n')

        # Collect boundary 'cuts' (character positions where a new paragraph may start)
        cuts: List[int] = [0]
        for m in self.PARA_SPLIT_REGEX.finditer(t):
            # A boundary ends *before* the matched delimiter and the next paragraph starts *after* it.
            end = m.start()
            next_start = m.end()
            if end > cuts[-1]:
                cuts.append(end)         # end of previous paragraph
            if next_start > cuts[-1]:
                cuts.append(next_start)  # start of next paragraph

        if cuts[-1] < len(t):
            cuts.append(len(t))

        # Build (start, end) spans, de-duplicated and sorted
        cuts = sorted(set(cuts))
        bounds = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]
        # Drop zero-length spans
        return [(s, e) for (s, e) in bounds if e > s]

    # --- Fast alignment -----------------------------------------------------------

    def align_breakpoints_fast(
        self,
        text: str,
        sentences: List[Dict[str, Any]],
        raw_breakpoints: Iterable[int],
        *,
        # Tolerance controls how far we are willing to "snap" forward to a paragraph end.
        # Avoid using mean paragraph length; fixed or capped tolerance is more stable.
        tolerance_chars: int = 400,
        # never snap more than 60% of the current paragraph length
        tolerance_cap_ratio: float = 0.6,
        # Avoid extreme jumps: allow snapping at most within the same paragraph (0) or into the next one (1).
        max_jump_paragraphs: int = 1,
        # Short paragraphs (e.g., list items) can be too tiny; treat them specially.
        short_para_threshold: int = 120,       # chars; below this counts as "short"
    ) -> List[int]:
        """
        Snap raw breakpoints to paragraph ends efficiently.

        Performance:
        Preprocessing: O(P + S)
        Per breakpoint: O(log P + log S) via binary search
        Total: O(P + S + B log(P+S))  [vs. O(B*(P+S)) in the original nested loops]

        Robustness improvements:
        - Paragraph detection understands Markdown/HTML-style blocks.
        - Fixed/capped tolerance; no average paragraph length.
        - Limited forward jump (0 = same paragraph only, 1 = allow next).
        - Short-paragraph handling to avoid huge shifts caused by tiny list items.
        """
        raw_breakpoints = list(raw_breakpoints)

        # 1) Detect paragraph spans once
        paragraph_bounds = self.detect_paragraph_bounds(text)
        P = len(paragraph_bounds)
        if P == 0:
            return sorted(raw_breakpoints)

        paragraph_end_pos = [end for _, end in paragraph_bounds]  # monotonic

        # 2) Precompute sentence start/end positions once
        S = len(sentences)
        sentence_starts = [s['start'] for s in sentences]
        sentence_ends = [s['start'] + len(s['sentence']) for s in sentences]

        # 3) Map each sentence to its paragraph (two-pointer merge in O(P+S))
        para_of_sentence = [0] * S
        last_sentence_in_para = [-1] * P
        p = 0
        for si, s_end in enumerate(sentence_ends):
            while p < P and s_end > paragraph_bounds[p][1]:
                p += 1
            if p >= P:
                p = P - 1
            para_of_sentence[si] = p
            # overwritten -> last sentence index in paragraph
            last_sentence_in_para[p] = si

        # 4) Helper to clamp tolerance per paragraph
        def effective_tolerance(p_idx: int) -> int:
            """Cap tolerance at a fraction of the current paragraph length."""
            start, end = paragraph_bounds[p_idx]
            plen = max(1, end - start)
            cap = int(plen * tolerance_cap_ratio)
            return min(tolerance_chars, cap)

        shifted = []

        for i in raw_breakpoints:
            # Position right AFTER the breakpoint (between sentence i and i+1)
            if i + 1 >= S:  # safeguard
                shifted.append(i)
                continue

            bp_pos = sentences[i + 1]['start']

            # Locate current paragraph by end position
            p_idx = bisect_right(paragraph_end_pos, bp_pos)
            if p_idx >= P:
                p_idx = P - 1

            # Candidate paragraphs to snap to:
            # - the same paragraph end if within tolerance
            # - optionally the next paragraph end (if allowed) and within tolerance
            candidates = []

            # current paragraph end
            tol_curr = effective_tolerance(p_idx)
            end_curr = paragraph_end_pos[p_idx]
            if bp_pos < end_curr <= bp_pos + tol_curr:
                candidates.append((p_idx, end_curr))

            # next paragraph end (if allowed)
            if max_jump_paragraphs >= 1 and p_idx + 1 < P:
                tol_next = effective_tolerance(p_idx + 1)
                end_next = paragraph_end_pos[p_idx + 1]
                # Special handling: if current paragraph is *very* short (e.g., a list item),
                # allow snapping to the next paragraph end preferentially (within its tolerance).
                if (paragraph_bounds[p_idx][1] - paragraph_bounds[p_idx][0]) < short_para_threshold:
                    if bp_pos < end_next <= bp_pos + tol_next:
                        candidates.append((p_idx + 1, end_next))
                else:
                    # Only consider next if current didn't qualify; avoids extreme shifts
                    if not candidates and (bp_pos < end_next <= bp_pos + tol_next):
                        candidates.append((p_idx + 1, end_next))

            # If no candidate, keep original breakpoint index
            if not candidates:
                shifted.append(i)
                continue

            # Choose the nearest valid paragraph end (smallest forward delta)
            cand_p_idx, target_end = min(
                candidates, key=lambda pe: pe[1] - bp_pos)

            # Find the sentence just before or at the paragraph end (binary search on sentence ends)
            j = bisect_left(sentence_ends, target_end)
            # 'j' is first sentence with end >= target_end; the breakpoint index should be j-1
            new_break_index = max(0, min(S - 2, j - 1))

            # Never move the breakpoint *backwards* beyond the original i unless it's in the same paragraph;
            # this keeps behavior intuitive when tolerance is generous.
            if para_of_sentence[new_break_index] < para_of_sentence[i]:
                new_break_index = i

            shifted.append(new_break_index)

        # Return unique, sorted indices (preserving increasing order)
        return sorted(set(shifted))

    def _interleave_keywords(self, kw_lists, limit=50):
        """
        Build a single keyword list from multiple sentence keyword lists using round-robin ordering.

        Process:
        1. Take the first keyword from each sentence (in order), then the second from each, etc.
        2. Skip duplicates (case-insensitive) to ensure each keyword appears only once.
        3. Stop once the list reaches the given limit.

        This ensures that the most prominent keyword from every sentence appears early in the chunk keyword list,
        rather than letting one sentence's keywords dominate the start.
        """
        maxlen = max((len(k) for k in kw_lists), default=0)
        out, seen = [], set()
        for k in range(maxlen):
            for lst in kw_lists:
                if k < len(lst):
                    kw = lst[k]
                    key = kw.lower()
                    if key not in seen:
                        seen.add(key)
                        out.append(kw)
                        if len(out) >= limit:
                            return out
        return out

    def _pool_embeddings(self, vecs):
        """
        Create a single embedding vector by mean-pooling a list of sentence embeddings, followed by L2-normalization.

        Steps:
        1. Stack all input vectors into a matrix.
        2. L2-normalize each vector to ensure cosine similarity consistency.
        3. Compute the mean vector across all normalized vectors.
        4. L2-normalize the resulting mean vector before returning.

        This produces a chunk-level embedding that represents the average semantic content
        of its constituent sentence embeddings.
        """
        E = np.vstack(vecs).astype(np.float32, copy=False)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / np.maximum(norms, 1e-12)
        v = E.mean(axis=0)
        n = np.linalg.norm(v)
        return v / np.maximum(n, 1e-12)

    def __combine_sentences(self, sentences, buffer_size=0):
        # Go through each sentence dict
        for i in range(len(sentences)):

            # Create a string that will hold the sentences which are joined
            combined_sentence = ''

            # Add sentences before the current one, based on the buffer size.
            for j in range(i - buffer_size, i):
                # Check if the index j is not negative (to avoid index out of range like on the first one)
                if j >= 0:
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += sentences[j]['sentence'] + ' '

            # Add the current sentence
            combined_sentence += sentences[i]['sentence']

            # Add sentences after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + buffer_size):
                # Check if the index j is within the range of the sentences list
                if j < len(sentences):
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += ' ' + sentences[j]['sentence']

            # Then add the whole thing to your dict
            # Store the combined sentence in the current sentence dict
            sentences[i]['combined_sentence'] = combined_sentence

        return sentences

    def __calculate_cosine_distances(self, sentences, assume_unit_norm: bool = False):
        """
        Compute cosine distances between consecutive sentence embeddings in a vectorized way.

        Why it's faster:
        - Avoids Python loops and sklearn overhead.
        - Single vstack + (optional) L2-normalization + einsum dot-products.

        Parameters
        ----------
        sentences : list[dict]
            Each dict must contain 'combined_sentence_embedding' as a 1D array-like.
        assume_unit_norm : bool
            If True, skip normalization (use when your embeddings are already L2-normalized).

        Returns
        -------
        distances_list : list[float]
            Cosine distances between consecutive embeddings (len = len(sentences) - 1).
        sentences : list[dict]
            Same list with 'distance_to_next' filled for each sentence except the last.
        """
        n = len(sentences)
        if n < 2:
            return [], sentences

        # Stack embeddings into a single (n, d) array (float32 to reduce memory traffic)
        E = np.vstack(
            [np.asarray(s['combined_sentence_embedding'],
                        dtype=np.float32, order='C') for s in sentences]
        )

        # L2-normalize rows unless caller guarantees unit norm
        if not assume_unit_norm:
            norms = np.linalg.norm(E, axis=1, keepdims=True)
            E = E / np.maximum(norms, 1e-12)

        # Cosine similarity of consecutive rows: dot(E[i], E[i+1])
        # einsum is efficient and avoids creating temporaries
        sims = np.einsum('ij,ij->i', E[:-1], E[1:])

        # Convert to cosine distance
        distances = 1.0 - sims

        # Write back distances to the sentence dicts (except for the last)
        # Keep as Python floats for JSON-friendliness
        for i, d in enumerate(distances):
            sentences[i]['distance_to_next'] = float(d)
        # Optionally: sentences[-1]['distance_to_next'] = None

        return distances.tolist(), sentences

    def chunk(self, text: str, breakpoint_percentile_threshold=95, min_length=400, align_to_paragraphs=True, keyword_extraction_method = 'tfidf'):
        logging.info("Step 1: Extracting sentences...")
        sentences_raw = self._extract_sentences(text)
        if len(sentences_raw) < 2:
            logging.info("Less than 2 sentences found — returning empty list.")
            return []

        logging.info(f'Extracted {len(sentences_raw)} sentences from text.')

        logging.info("Step 2: Combining sentences with buffer...")
        sentences = [{'sentence': s['sentence'], 'start': s['start'],
                      'index': i} for i, s in enumerate(sentences_raw)]
        sentences = self.__combine_sentences(sentences, buffer_size=2)

        logging.info("Step 3: Running preprocessing before embedding...")
        combined_sentences = [x['combined_sentence'] for x in sentences]
        pre = ChunkEmbeddingPreprocessor(
            method=keyword_extraction_method, n_keywords=30, corpus=combined_sentences, identifier='text')

        reduced_sentences = []
        for i, s in enumerate(tqdm(combined_sentences, desc="Sentence preprocessing", unit="sent", leave=False)):
            kw_str = pre.run({'text': s}) 
            reduced_sentences.append(kw_str)
            sentences[i]['keywords'] = [kw.strip()
                                        for kw in kw_str.split(',') if kw.strip()]
        logging.info("Preprocessing complete.")

        logging.info("Creating embeddings for reduced sentences...")
        embeddings = self.embedding_func(sentences=reduced_sentences)

        for i, sentence in enumerate(tqdm(sentences, desc="Creating embeddings", unit="sent"), start=1):
            sentence['combined_sentence_embedding'] = embeddings[i - 1]

        logging.info(
            "Step 4: Calculating cosine distances between embeddings...")
        distances, sentences = self.__calculate_cosine_distances(sentences)

        logging.info("Step 5: Determining breakpoints...")
        breakpoint_distance_threshold = np.percentile(
            distances, breakpoint_percentile_threshold)
        raw_breakpoints = [i for i, x in enumerate(
            distances) if x > breakpoint_distance_threshold]
        logging.info(f"  Found {len(raw_breakpoints)} raw breakpoints.")

        plotter = SimilarityBreakpointsPlotter(
            data_path=self.working_dir)  # oder ein anderer Pfad
        raw_plot = plotter.plot_distances_with_breakpoints(
            distances=distances,
            breakpoints=raw_breakpoints,
            threshold=breakpoint_distance_threshold,
            title="Raw cosine distances & raw breakpoints",
            filename="raw_distances_breakpoints.png",
            show=False,
        )

        if align_to_paragraphs:
            logging.info(
                "Step 6: Aligning breakpoints to paragraph boundaries...")
            indices_above_thresh = self.align_breakpoints_fast(
                text=text,
                sentences=sentences,
                raw_breakpoints=raw_breakpoints,
                tolerance_chars=400,            # fixed tolerance in characters
                tolerance_cap_ratio=0.6,        # cap tolerance to 60% of paragraph length
                max_jump_paragraphs=1,          # allow snapping into the next paragraph at most
                # treat very short paragraphs (bullets) specially
                short_para_threshold=120
            )
        else:
            indices_above_thresh = sorted(raw_breakpoints)

        logging.info(
            f"Step 7: Building chunks from {len(indices_above_thresh)} breakpoints...")
        chunks = []
        start_index = 0
        global_chunk_idx = 0

        for chunk_idx, index in enumerate(
            tqdm(indices_above_thresh, desc="Building chunks",
                 unit="chunk", leave=False)
        ):
            end_index = index  # inclusive
            group = sentences[start_index:end_index + 1]

            combined_text = ' '.join([d['sentence'] for d in group])

            # Interleave keywords from all sentences in the chunk
            kw_lists = [d.get('keywords', []) for d in group]
            inter_kws = self._interleave_keywords(kw_lists, limit=50)

            # Pool sentence embeddings into a single chunk embedding
            vecs = [d['combined_sentence_embedding'] for d in group]
            chunk_vec = self._pool_embeddings(vecs)

            chunks.append({
                'chunk_idx': chunk_idx,
                'from_idx': start_index,
                'to_idx': end_index,  # inclusive
                'text': combined_text,
                'keywords': inter_kws,
                'chunk_embedding': chunk_vec
            })

            global_chunk_idx = chunk_idx
            start_index = index + 1

        # Final chunk if there is remaining text
        if start_index < len(sentences):
            end_index = len(sentences) - 1  # inclusive
            group = sentences[start_index:end_index + 1]

            combined_text = ' '.join([d['sentence'] for d in group])

            kw_lists = [d.get('keywords', []) for d in group]
            inter_kws = self._interleave_keywords(kw_lists, limit=50)

            vecs = [d['combined_sentence_embedding'] for d in group]
            chunk_vec = self._pool_embeddings(vecs)

            chunks.append({
                'chunk_idx': global_chunk_idx + 1,
                'from_idx': start_index,
                'to_idx': end_index,  # inclusive
                'text': combined_text,
                'keywords': inter_kws,
                'chunk_embedding': chunk_vec
            })
            logging.info(f"  Created final chunk {global_chunk_idx + 2}")

        logging.info("Step 8: Merging short chunks...")
        i = 0
        while i < len(chunks):
            if len(chunks[i]['text']) < min_length or (i < len(chunks) - 1 and len(chunks[i + 1]['text']) < min_length):
                added_once = False
                while i + 1 < len(chunks) and (
                        len(chunks[i]['text']) + len(chunks[i + 1]['text']) <= min_length or not added_once or len(
                            chunks[i + 1]['text']) < min_length):
                    added_once = True
                    chunks[i]['text'] += ' ' + chunks[i + 1]['text']
                    chunks[i]['to_idx'] = chunks[i + 1]['to_idx']
                    del chunks[i + 1]
            i += 1

        logging.info("Step 9: Finalizing chunks...")
        for idx, chunk in enumerate(chunks, start=1):
            chunk['chunk_idx'] = int(chunk['chunk_idx'])
            chunk['from_idx'] = int(chunk['from_idx'])
            chunk['to_idx'] = int(chunk['to_idx'])
            chunk['text'] = str(chunk['text'])
        logging.info(f"Chunking complete: {len(chunks)} chunks created.")
        return chunks
