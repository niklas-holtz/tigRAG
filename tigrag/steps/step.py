from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from ..storage.chunk_storage import ChunkStorage
from ..utils.llm_invoker import LLMInvoker
from ..utils.embedding_invoker import EmbeddingInvoker
import os


@dataclass
class RetrieveContext:
    query: str
    chunk_storage: ChunkStorage
    working_dir: str
    llm_invoker: LLMInvoker
    embedding_invoker: EmbeddingInvoker
    llm_worker_nodes: int = 1
    event_row_id: int = None
    selected: List = field(default_factory=list)
    # Zwischenstände:
    clusters: List[List[Dict[str, Any]]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    scored: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    result: List[Dict[str, Any]] = field(default_factory=list)
    # Beliebige Metadaten:
    meta: Dict[str, Any] = field(default_factory=dict)


class Step(ABC):
    name: str = "step"

    @abstractmethod
    def run(self, ctx: RetrieveContext) -> RetrieveContext:
        pass

    def _load_prompt(self, path: str) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(base_dir, path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Prompt file not found: {abs_path}")
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()