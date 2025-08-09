import re 
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from ..utils.embedding_invoker import EmbeddingInvoker
from scipy.signal import argrelextrema
import logging
from ..nlp.chunk_embedding_preprocessor import ChunkEmbeddingPreprocessor

class TextChunker():

    def __init__(self, embedding_func: EmbeddingInvoker) -> None:
        self.embedding_func = embedding_func

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

    def __calculate_cosine_distances(self, sentences):
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            
            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]['distance_to_next'] = distance

        # Optionally handle the last sentence
        # sentences[-1]['distance_to_next'] = None  # or a default value

        return distances, sentences

    def chunk(self, text: str, breakpoint_percentile_threshold=95, min_length=400, align_to_paragraphs=True):
        # 1. Find paragraph boundaries
        paragraph_starts = [m.end() for m in re.finditer(r'\n\s*\n', text)]

        # 2. Extract sentences with start positions
        sentence_matches = list(re.finditer(r'([^.?!\n]+[.?!])', text))
        if len(sentence_matches) < 2:
            return []

        sentences_raw = []
        for match in sentence_matches:
            sentence = match.group().strip()
            if sentence:
                sentences_raw.append({
                    'sentence': sentence,
                    'start': match.start()
                })

        # 3. Add index and combine with buffer
        sentences = [{'sentence': s['sentence'], 'start': s['start'], 'index': i} for i, s in enumerate(sentences_raw)]
        sentences = self.__combine_sentences(sentences, buffer_size=2)

        # 4. TF-IDF preprocessing before embedding
        combined_sentences = [x['combined_sentence'] for x in sentences]
        pre = ChunkEmbeddingPreprocessor(method="tfidf", n_keywords=30, corpus=combined_sentences, identifier='text')
        reduced_sentences = [pre.run({'text': s}) for s in combined_sentences]
        embeddings = self.embedding_func(sentences=reduced_sentences)

        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

        # 5. Compute cosine distances between sentence embeddings
        distances, sentences = self.__calculate_cosine_distances(sentences)

        # 6. Determine breakpoints based on distance threshold
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        raw_breakpoints = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

        # 7. Optionally align breakpoints to paragraphs
        if align_to_paragraphs:
            paragraph_matches = list(re.finditer(r'\n\s*\n', text))
            paragraph_bounds = []
            last_pos = 0
            for match in paragraph_matches:
                end = match.start()
                paragraph_bounds.append((last_pos, end))
                last_pos = match.end()
            paragraph_bounds.append((last_pos, len(text)))

            paragraph_lengths = [end - start for start, end in paragraph_bounds]
            paragraph_tolerance = int(np.mean(paragraph_lengths)) if paragraph_lengths else 400

            shifted_breakpoints = set()
            for i in raw_breakpoints:
                bp_pos = sentences[i + 1]['start']

                new_break_index = None
                for start, end in paragraph_bounds:
                    if bp_pos < end <= bp_pos + paragraph_tolerance:
                        for j in range(i + 1, len(sentences)):
                            sent_end = sentences[j]['start'] + len(sentences[j]['sentence'])
                            if sent_end >= end:
                                new_break_index = j - 1
                                break
                        break

                if new_break_index is not None:
                    shifted_breakpoints.add(new_break_index)
                else:
                    shifted_breakpoints.add(i)

            indices_above_thresh = sorted(shifted_breakpoints)
        else:
            indices_above_thresh = raw_breakpoints

        # 8. Build chunks
        chunks = []
        start_index = 0
        global_chunk_idx = 0

        for chunk_idx, index in enumerate(indices_above_thresh):
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append({
                'chunk_idx': chunk_idx,
                'from_idx': start_index,
                'to_idx': end_index,
                'text': combined_text
            })
            global_chunk_idx = chunk_idx
            start_index = index + 1

        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append({
                'chunk_idx': global_chunk_idx + 1,
                'from_idx': start_index,
                'to_idx': len(sentences),
                'text': combined_text
            })

        # 9. Merge short chunks
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

        # 10. Make JSON-compatible
        # 10. Make JSON-compatible and add keyword embeddings
        for chunk in chunks:
            chunk['chunk_idx'] = int(chunk['chunk_idx'])
            chunk['from_idx'] = int(chunk['from_idx'])
            chunk['to_idx'] = int(chunk['to_idx'])

            # Get TF-IDF keywords for the chunk text
            keyword_string = pre.run({'text': chunk['text']})
            keywords = [kw.strip() for kw in keyword_string.split(',') if kw.strip()]
            chunk['keywords'] = keywords
            chunk['text'] = str(chunk['text'])

            # Create embedding for the keyword string
            if keywords:
                keyword_text = " ".join(keywords)
                keyword_embedding = self.embedding_func(sentences=[keyword_text])[0]
                chunk['keyword_embedding'] = keyword_embedding
            else:
                chunk['keyword_embedding'] = None

        return chunks


