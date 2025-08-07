from dataclasses import dataclass
from typing import Callable


@dataclass
class TigParam:
    """Minimal configuration for TemporalInfluenceGraph."""

    embedding_model_name: str
    """Name of the embedding model to use. E.g., 'local', 'local_fast', 'titan'."""

    llm_name: str
    """LLM function used to generate responses."""
