from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class ChunkStorage(ABC):
    # --- chunks API ---
    @abstractmethod
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
        """Insert a chunk into storage (optionally linked to a document with char offsets)."""
        pass

    @abstractmethod
    def get_all_chunks(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve chunks with their metadata and embeddings."""
        pass

    # --- documents API ---
    @abstractmethod
    def upsert_document(self, text: str) -> int:
        """Insert full text as a document if not already present; return doc_id."""
        pass

    @abstractmethod
    def get_document_text(self, doc_id: int) -> Optional[str]:
        """Retrieve full text of a document by id."""
        pass

    # --- events API ---
    @abstractmethod
    def insert_events(self, query: str, events: List[Dict[str, Any]], retrieved_at: Optional[str] = None) -> None:
        """
        Store a list of event objects (as JSON) associated with a query and a retrieval timestamp.
        Implementations should serialize `events` to JSON internally.
        """
        pass

    @abstractmethod
    def get_events(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve stored events. If `query` is provided, filter by that query.
        Return format suggestion (per row/item):
            {
              "query": str,
              "retrieved_at": str,   # ISO-8601
              "events": List[Dict[str, Any]]
            }
        """
        pass

    @abstractmethod
    def close(self):
        """Close any open connections/resources."""
        pass
