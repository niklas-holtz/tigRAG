from dataclasses import dataclass

@dataclass
class TigParam:
    """Minimal configuration for TemporalInfluenceGraph."""

    embedding_model_name: str
    """Name of the embedding model to use (e.g., 'local', 'local_fast', 'titan')."""

    llm_name: str
    """Name or identifier of the LLM to use for generating responses."""

    working_dir: str
    """Directory for storing intermediate files, plots, and outputs."""

    llm_worker_nodes: int = 1
    """Number of LLM worker nodes to use in parallel processing."""

    text_chunker_breakpoint_percentile_threshold = 95
    """Percentile threshold for chunk breakpoints."""

    text_chunker_min_chunk_size = 800
    """Minimum chunk size in characters."""

    keyword_extraction_method = 'rake'
    """Keyword extraction method ('tfidf', 'yake', or 'rake')."""
