import logging
import numpy as np
from typing import List, Dict, Any, Optional
from ..steps.step import Step, RetrieveContext
from string import Template
from tqdm import tqdm
import heapq
from ..data_plotter.candidate_matrix_plotter import CandidateMatrixPlotter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class EventRelationStep(Step):
    """
    Builds a candidate similarity matrix between events:
    - Compares each event's summary to each other event's influence causal chains
    - Uses ctx.embedding_invoker to embed text
    - Stores the candidate similarity matrix in ctx.event_relation_candidates
    """

    def __init__(self, relation_prompt_path: str = "../prompts/event_relation_prompt.txt",
                 max_workers: int = 10, top_n: int = 20):
        self.max_workers = max_workers
        self.top_n = top_n
        self.prompt = self._load_prompt(relation_prompt_path)
        self._prompt_tmpl = Template(self.prompt) if self.prompt else None

    def _load_events(self, ctx: RetrieveContext) -> tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Loads events and their database row ID from the context or from chunk_storage.

        Returns:
            events (List[Dict[str, Any]]): The list of events.
            row_id (Optional[int]): The database ID for these events if available.
        """
        # Try events from context first
        events = getattr(ctx, "events", []) or []
        row_id = getattr(ctx, "event_row_id", None)

        # If not available in context, try from storage
        if not events and getattr(ctx, "chunk_storage", None):
            if hasattr(ctx.chunk_storage, "get_events"):
                row_id, events = ctx.chunk_storage.get_events(query=ctx.query)
                logging.info(f"Loaded {len(events)} events from storage (row_id={row_id}).")
            else:
                logging.error("chunk_storage has no 'get_events' method.")
        else:
            logging.debug("Using events from context.")

        return events, row_id

    def _get_top_n_pairs(self, matrix: np.ndarray, events: list[dict], n: int) -> list[tuple]:
        """Find top N most similar event pairs (i,j) from the similarity matrix."""
        pairs = []
        size = matrix.shape[0]
        for i in range(size):
            for j in range(size):
                if i != j:
                    pairs.append((matrix[i, j], i, j))
        # get top N by similarity score
        top_pairs = heapq.nlargest(n, pairs, key=lambda x: x[0])
        return top_pairs

    def _build_relation_prompt(self, ev_a: dict, ev_b: dict) -> str:
        """Fill relation prompt with data from both events."""
        return self._prompt_tmpl.safe_substitute(
            event_a_summary=ev_a.get("summary", ""),
            event_a_influence="\n".join(ev_a.get("influence", [])),
            event_b_summary=ev_b.get("summary", ""),
            event_b_influence="\n".join(ev_b.get("influence", []))
        )

    def _process_relation(self, idx_a: int, idx_b: int, ev_a: dict, ev_b: dict, ctx: RetrieveContext) -> dict:
        """Call LLM to determine the relation between two events."""
        prompt = self._build_relation_prompt(ev_a, ev_b)
        try:
            param = {'max_new_tokens': 3000}
            relation_text = ctx.llm_invoker(message=[{"role": "user", "content": prompt}], parameters=param)
        except Exception as e:
            logging.error(f"LLM call failed for pair ({idx_a}, {idx_b}): {e}")
            relation_text = None

        source_id = ev_a.get("id")
        target_id = ev_b.get("id")

        if source_id is None or target_id is None:
            logging.warning(f"Missing event IDs for pair indices ({idx_a}, {idx_b}). "
                            f"source_id={source_id}, target_id={target_id}")

        return {
            "source_id": source_id,
            "target_id": target_id,
            "similarity": float(ctx.candidate_matrix[idx_a, idx_b]),
            "relation_raw": relation_text,
            "retrieved_at": datetime.utcnow().isoformat(),
            "_source_idx": idx_a,
            "_target_idx": idx_b,
        }

    def run(self, ctx: RetrieveContext) -> RetrieveContext:
        logging.info("Starting EventRelationStep...")
        
        self.max_workers = getattr(ctx, "llm_worker_nodes", 1)

        events, row_id = self._load_events(ctx)
        if not events:
            logging.warning("No events found in context — skipping relation building.")
            ctx.event_relations = []
            return ctx

        if not hasattr(ctx, "embedding_invoker") or ctx.embedding_invoker is None:
            logging.error("No embedding_invoker found in context.")
            ctx.event_relations = []
            return ctx

        # Step 1: Build candidate similarity matrix
        candidate_matrix = self._build_candidate_matrix(ctx, events)
        CandidateMatrixPlotter(data_path=ctx.working_dir).plot_matrix(candidate_matrix, title="Event Influence Similarity")
        ctx.candidate_matrix = candidate_matrix  # store for later

        # Step 2: Select Top N pairs
        top_pairs = self._get_top_n_pairs(candidate_matrix, events, self.top_n)
        logging.info(f"Selected top {len(top_pairs)} event pairs for relation extraction.")

        # Step 3: Process in parallel with LLM
        relations = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_relation, i, j, events[i], events[j], ctx): (i, j)
                for score, i, j in top_pairs
            }

            for future in as_completed(futures):
                i, j = futures[future]
                try:
                    rel = future.result()
                    relations.append(rel)
                except Exception as e:
                    logging.error(f"Error processing relation for pair ({i}, {j}): {e}")

        ctx.event_relations = relations

        # Step 4: Persist in DB if available
        if hasattr(ctx, "chunk_storage") and ctx.chunk_storage:
            try:
                ctx.chunk_storage.insert_relations(event_id=row_id, relations=relations)
                logging.info("Event relations saved to DB.")
            except Exception as e:
                logging.error(f"Failed to save relations: {e}")

        logging.info("EventRelationStep finished.")
        return ctx

    def _build_candidate_matrix(self, ctx: RetrieveContext, events: list[dict]) -> np.ndarray:
        """
        Builds a candidate similarity matrix comparing each event's summary to all other events' influence chains.
        """
        n = len(events)
        logging.info(f"Embedding summaries for {n} events...")

        # Step 1: Embed summaries
        summary_embeddings: list = []
        for ev in tqdm(events, desc="Embedding summaries", unit="event", total=n):
            summary_text = ev.get("summary", "") or ""
            if summary_text.strip():
                emb = ctx.embedding_invoker(sentences=summary_text)
                if isinstance(emb, list) and len(emb) == 1:
                    emb = emb[0]
                summary_embeddings.append(np.asarray(emb, dtype=np.float32))
            else:
                summary_embeddings.append(None)

        # Step 2: Embed influences per event
        influence_embeddings_per_event: list = []
        for ev in tqdm(events, desc="Embedding influence chains", unit="event", total=n):
            influences = ev.get("influence", []) or []
            emb_list = []
            for infl in influences:
                infl_text = infl if isinstance(infl, str) else str(infl)
                emb = ctx.embedding_invoker(sentences=infl_text)
                if isinstance(emb, list) and len(emb) == 1:
                    emb = emb[0]
                emb_list.append(np.asarray(emb, dtype=np.float32))
            influence_embeddings_per_event.append(emb_list)

        # Step 3: Compute candidate similarity matrix
        candidate_matrix = np.zeros((n, n), dtype=np.float32)
        for i in tqdm(range(n), desc="Building similarity matrix", unit="event", total=n):
            for j in range(n):
                if i == j:
                    candidate_matrix[i, j] = 0.0
                    continue
                sim_scores = [
                    self._cosine_similarity(summary_embeddings[i], infl_emb)
                    for infl_emb in influence_embeddings_per_event[j]
                    if infl_emb is not None
                ]
                candidate_matrix[i, j] = max(sim_scores) if sim_scores else 0.0

        return candidate_matrix

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
