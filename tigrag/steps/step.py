from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from ..storage.chunk_storage import ChunkStorage
from ..utils.llm_invoker import LLMInvoker


@dataclass
class RetrieveContext:
    query: str
    chunk_storage: ChunkStorage
    working_dir: str
    llm_invoker: LLMInvoker
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