import os
import json
import sqlite3
import numpy as np
from .utils.embedding_invoker import EmbeddingInvoker
from .utils.tig_param import TigParam
from .nlp.text_chunker import TextChunker
from .storage.sqlite_chunk_storage import SQLiteChunkStorage


class TemporalInfluenceGraph:
    def __init__(self, working_dir: str, query_param: TigParam):
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

        # Create embedding function
        self.embedding_func = EmbeddingInvoker(
            model_name=query_param.embedding_model_name,
            cache_dir=self.working_dir
        )

        # Init repository (can be swapped out later)
        db_path = os.path.join(self.working_dir, "chunks.db")
        self.chunk_stor = SQLiteChunkStorage(db_path)

    def insert(self, text: str):
        text_chunker = TextChunker(self.embedding_func)
        chunks = text_chunker.chunk(text, min_length=10)

        for chunk in chunks:
            self.chunk_stor.insert_chunk(
                from_idx=int(chunk['from_idx']),
                to_idx=int(chunk['to_idx']),
                text=chunk['text'],
                keywords=chunk.get("keywords", []),
                embedding=chunk.get("keyword_embedding", None)
            )

    def retrieve(self, query: str):
        # Optional: implement similarity search or keyword lookup here
        pass

    def __del__(self):
        # Cleanly close DB connection when object is destroyed
        if hasattr(self, 'conn'):
            self.conn.close()
