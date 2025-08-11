import os
import json
import sqlite3
import numpy as np
from .utils.embedding_invoker import EmbeddingInvoker
from .utils.llm_invoker import LLMInvoker
from .utils.tig_param import TigParam
from .nlp.text_chunker import TextChunker
from .storage.sqlite_chunk_storage import SQLiteChunkStorage
from .steps.step import RetrieveContext
from .steps.chunk_preparation_step import ChunkPreparationStep
from .steps.chunk_selection_step import ChunkSelectionStep
from .steps.event_extractor_step import EventExtractorStep

class TemporalInfluenceGraph:
    def __init__(self, query_param: TigParam):
        self.query_param = query_param
        os.makedirs(self.query_param.working_dir, exist_ok=True)

        # Create embedding function
        self.embedding_func = EmbeddingInvoker(
            model_name=query_param.embedding_model_name,
            cache_dir=self.query_param.working_dir
        )

        # Init repository (can be swapped out later)
        db_path = os.path.join(self.query_param.working_dir, "chunks.db")
        self.chunk_stor = SQLiteChunkStorage(db_path)

    def insert(self, text: str):
        # 1) upsert full document and get doc_id
        doc_id = self.chunk_stor.upsert_document(text)

        # 2) chunk the text
        text_chunker = TextChunker(self.embedding_func, self.query_param.working_dir)
        chunks = text_chunker.chunk(text, min_length=10)

        for chunk in chunks:
            self.chunk_stor.insert_chunk(
                from_idx=int(chunk['from_idx']),
                to_idx=int(chunk['to_idx']),
                text=chunk['text'],
                keywords=chunk.get("keywords", []),
                embedding=chunk.get("keyword_embedding", None),
                doc_id=doc_id,
            )

    def retrieve(self, query: str):
        ctx = RetrieveContext(
            query=query,
            chunk_storage=self.chunk_stor,
            working_dir=self.query_param.working_dir,
            llm_invoker=LLMInvoker(self.query_param.working_dir)
        )

        pipeline = [
            ChunkPreparationStep(),
            ChunkSelectionStep(),
            EventExtractorStep()
        ]

        for step in pipeline:
            ctx = step.run(ctx)

        return ctx

    def __del__(self):
        # Cleanly close DB connection when object is destroyed
        if hasattr(self, 'conn'):
            self.conn.close()
