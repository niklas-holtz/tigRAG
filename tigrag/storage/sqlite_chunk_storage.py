import sqlite3
import json
import numpy as np
from .chunk_storage import ChunkStorage

class SQLiteChunkStorage(ChunkStorage):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_idx INTEGER,
                to_idx INTEGER,
                text TEXT UNIQUE,
                keywords TEXT,
                keyword_embedding BLOB
            )
        """)
        self.conn.commit()

    def insert_chunk(self, from_idx: int, to_idx: int, text: str, keywords: list, embedding: list | None):
        if embedding is not None:
            embedding_blob = sqlite3.Binary(np.array(embedding, dtype=np.float32).tobytes())
        else:
            embedding_blob = None

        try:
            self.cursor.execute("""
                INSERT INTO chunks (from_idx, to_idx, text, keywords, keyword_embedding)
                VALUES (?, ?, ?, ?, ?)
            """, (
                from_idx,
                to_idx,
                text,
                json.dumps(keywords),
                embedding_blob
            ))
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Skip duplicates
            pass

    def close(self):
        self.conn.close()
