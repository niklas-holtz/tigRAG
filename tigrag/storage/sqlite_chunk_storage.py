import sqlite3
import json
import numpy as np
import hashlib
from typing import Optional, Dict, Any, List
from .chunk_storage import ChunkStorage
from datetime import datetime

class SQLiteChunkStorage(ChunkStorage):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    # --- schema ---
    def _create_tables(self):
        # documents table for full texts (deduplicated by sha256)
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS documents
                            (
                                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                                sha256 TEXT UNIQUE NOT NULL,
                                text   TEXT        NOT NULL
                            )
                            """)
        # chunks table (may already exist; create with new columns for fresh DBs)
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS chunks
                            (
                                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                                doc_id            INTEGER,
                                from_idx          INTEGER,
                                to_idx            INTEGER,
                                text              TEXT UNIQUE,
                                keywords          TEXT,
                                keyword_embedding BLOB,
                                FOREIGN KEY (doc_id) REFERENCES documents (id)
                            )
                            """)
        # events table for extracted events
        # events table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS events
            (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                query          TEXT NOT NULL,
                retrieved_at   TEXT NOT NULL,
                events_json    TEXT NOT NULL,
                relations_json TEXT DEFAULT '[]',
                ratings_json   TEXT DEFAULT '[]'
            )
        """)
        self.conn.commit()

    # --- documents API ---
    def upsert_document(self, text: str) -> int:
        digest = self._sha256(text)
        self.cursor.execute("SELECT id FROM documents WHERE sha256 = ?", (digest,))
        row = self.cursor.fetchone()
        if row:
            return int(row[0])
        self.cursor.execute(
            "INSERT INTO documents (sha256, text) VALUES (?, ?)",
            (digest, text),
        )
        self.conn.commit()
        return int(self.cursor.lastrowid)

    def get_document_text(self, doc_id: int) -> Optional[str]:
        self.cursor.execute("SELECT text FROM documents WHERE id = ?", (doc_id,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    # --- chunks API ---
    def insert_chunk(
        self,
        from_idx: int,
        to_idx: int,
        text: str,
        keywords: list,
        embedding: list | None,
        *,
        doc_id: Optional[int] = None
    ):
        if embedding is not None:
            embedding_blob = sqlite3.Binary(np.array(embedding, dtype=np.float32).tobytes())
        else:
            embedding_blob = None
        try:
            self.cursor.execute("""
                INSERT INTO chunks (doc_id, from_idx, to_idx, text, keywords, keyword_embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                from_idx,
                to_idx,
                text,
                json.dumps(keywords),
                embedding_blob
            ))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass

    def get_all_chunks(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        sql = """
            SELECT id, doc_id, from_idx, to_idx, text, keywords, keyword_embedding
            FROM chunks
            ORDER BY id ASC
        """
        params: list = []
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])

        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()

        out = []
        for _id, doc_id, from_idx, to_idx, text, keywords_json, emb_blob in rows:
            keywords = json.loads(keywords_json) if keywords_json else []
            embedding = np.frombuffer(emb_blob, dtype=np.float32) if emb_blob else None
            out.append({
                "id": _id,
                "doc_id": doc_id,
                "from_idx": from_idx,
                "to_idx": to_idx,
                "text": text,
                "keywords": keywords,
                "embedding": embedding,
            })
        return out

     # --- events API ---
    def insert_events(self, query: str, events: List[Dict[str, Any]], retrieved_at: Optional[str] = None) -> int:
        if retrieved_at is None:
            retrieved_at = datetime.utcnow().isoformat()
        events_json_str = json.dumps(events, ensure_ascii=False)
        self.cursor.execute("""
            INSERT INTO events (query, retrieved_at, events_json, relations_json, ratings_json)
            VALUES (?, ?, ?, '[]', '[]')
        """, (query, retrieved_at, events_json_str))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_relations(self, event_id: int, relations: List[Dict[str, Any]]):
        relations_json_str = json.dumps(relations, ensure_ascii=False)
        self.cursor.execute("""
            UPDATE events
            SET relations_json = ?
            WHERE id = ?
        """, (relations_json_str, event_id))
        self.conn.commit()

    def insert_ratings(self, event_id: int, ratings: List[Dict[str, Any]]):
        ratings_json_str = json.dumps(ratings, ensure_ascii=False)
        self.cursor.execute("""
            UPDATE events
            SET ratings_json = ?
            WHERE id = ?
        """, (ratings_json_str, event_id))
        self.conn.commit()

    def get_events(self, query: Optional[str] = None) -> tuple[Optional[int], List[Dict[str, Any]]]:
        if query:
            self.cursor.execute("""
                SELECT id, events_json
                FROM events
                WHERE query = ?
                ORDER BY retrieved_at DESC
                LIMIT 1
            """, (query,))
        else:
            self.cursor.execute("""
                SELECT id, events_json
                FROM events
                ORDER BY retrieved_at DESC
                LIMIT 1
            """)
        row = self.cursor.fetchone()
        if not row:
            return None, []
        eid, events_json = row
        try:
            events_list = json.loads(events_json) if events_json else []
        except json.JSONDecodeError:
            events_list = []
        return eid, events_list

    def get_events_full(self, query: Optional[str] = None) -> tuple[Optional[int], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Lädt die neueste Events-Zeile (optional gefiltert per query) und gibt
        (id, events_list, relations_list, ratings_list) zurück.
        """
        if query:
            self.cursor.execute("""
                SELECT id, events_json, relations_json, ratings_json
                FROM events
                WHERE query = ?
                ORDER BY retrieved_at DESC
                LIMIT 1
            """, (query,))
        else:
            self.cursor.execute("""
                SELECT id, events_json, relations_json, ratings_json
                FROM events
                ORDER BY retrieved_at DESC
                LIMIT 1
            """)
        row = self.cursor.fetchone()
        if not row:
            return None, [], [], []

        eid, events_json, relations_json, ratings_json = row
        try:
            events_list = json.loads(events_json) if events_json else []
        except json.JSONDecodeError:
            events_list = []
        try:
            relations_list = json.loads(relations_json) if relations_json else []
        except json.JSONDecodeError:
            relations_list = []
        try:
            ratings_list = json.loads(ratings_json) if ratings_json else []
        except json.JSONDecodeError:
            ratings_list = []

        return eid, events_list, relations_list, ratings_list

    # --- helper ---
    @staticmethod
    def _sha256(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def close(self):
        self.conn.close()
