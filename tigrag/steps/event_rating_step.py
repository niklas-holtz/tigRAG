from __future__ import annotations
import logging
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from ..steps.step import Step, RetrieveContext


class EventRatingStep(Step):
    """
    Computes a PageRank-like score for events based on directed relations between them.
    - Uses ctx.event_relations (produced earlier) with edges source_id -> target_id
    - Differentiates relation types via weights
    - Adds an 'outgoing bonus' term that rewards nodes with strong outgoing edges,
      especially 'contradiction' (configurable).
    - Writes per-event rating into event['rating'] and a summary list into ctx.event_ratings
    - Persists ratings to SQLite (events.ratings_json) using ctx.chunk_storage.insert_ratings(...)
    """

    def __init__(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        mix_weight: float = 0.7,  # final_score = mix_weight*pagerank + (1-mix_weight)*outgoing_bonus
        include_none_edges: bool = False,
        # Incoming weights (used to build the transition matrix)
        relation_weight_in: Optional[Dict[str, float]] = None,
        # Outgoing bonus per edge-type (summed per node and normalized)
        relation_weight_out: Optional[Dict[str, float]] = None,
    ):
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.mix_weight = float(mix_weight)
        self.include_none_edges = bool(include_none_edges)

        # Default weights (tune as needed)
        self.relation_weight_in = relation_weight_in or {
            "contradiction": 1.2,
            "causal": 1.0,
            "amplification": 0.8,
            "mitigation": 0.6,
            "coreference": 0.4,
            "none": 0.1,   # very low influence; set to 0.0 and include_none_edges=False to ignore
        }
        self.relation_weight_out = relation_weight_out or {
            # outgoing 'contradiction' gives strongest bonus
            "contradiction": 1.0,
            "causal": 0.6,
            "amplification": 0.4,
            "mitigation": 0.4,
            "coreference": 0.2,
            "none": 0.0,
        }

    def run(self, ctx: RetrieveContext) -> RetrieveContext:
        logging.info("Starting EventRatingStep...")

        events: List[Dict[str, Any]] = getattr(ctx, "events", []) or []
        relations: List[Dict[str, Any]] = getattr(ctx, "event_relations", []) or []

        if not events:
            logging.warning("No events found in context — skipping rating.")
            ctx.event_ratings = []
            return ctx

        # Ensure each event has an ID; if missing, assign sequential IDs (1..n)
        missing_ids = any(ev.get("id") is None for ev in events)
        if missing_ids:
            logging.warning("Some events are missing 'id'. Assigning sequential IDs (1..n).")
            for k, ev in enumerate(events, start=1):
                if ev.get("id") is None:
                    ev["id"] = k

        # Build id <-> index maps
        ids: List[int] = [int(ev["id"]) for ev in events]
        id_to_index: Dict[int, int] = {eid: i for i, eid in enumerate(ids)}
        n = len(ids)

        # Build weighted adjacency from relations
        edges: List[Tuple[int, int, str, float]] = []  # (src_idx, tgt_idx, rel_label, in_weight)

        def _parse_relation_label(rel: Dict[str, Any]) -> Optional[str]:
            # Try direct field
            label = rel.get("relation")
            if isinstance(label, str):
                return label.strip().lower()

            # Try to parse JSON from relation_raw
            raw = rel.get("relation_raw")
            if isinstance(raw, dict):
                lab = raw.get("relation")
                return lab.strip().lower() if isinstance(lab, str) else None

            if isinstance(raw, str):
                try:
                    obj = json.loads(raw)
                    lab = obj.get("relation")
                    return lab.strip().lower() if isinstance(lab, str) else None
                except Exception:
                    return None
            return None

        valid_types = set(self.relation_weight_in.keys())

        for rel in relations:
            src_id = rel.get("source_id")
            tgt_id = rel.get("target_id")
            if src_id is None or tgt_id is None:
                # Fallback: legacy indices mapping
                src_idx = rel.get("_source_idx")
                tgt_idx = rel.get("_target_idx")
                if isinstance(src_idx, int) and 0 <= src_idx < n:
                    src_id = events[src_idx].get("id")
                if isinstance(tgt_idx, int) and 0 <= tgt_idx < n:
                    tgt_id = events[tgt_idx].get("id")

            if src_id is None or tgt_id is None:
                logging.debug(f"Skipping relation without IDs: {rel}")
                continue

            label = _parse_relation_label(rel)
            if label is None:
                continue

            if (label == "none" and not self.include_none_edges):
                continue

            if label not in valid_types:
                continue

            if src_id not in id_to_index or tgt_id not in id_to_index:
                continue

            w_in = float(self.relation_weight_in.get(label, 0.0))
            if w_in <= 0.0:
                continue

            edges.append((id_to_index[int(src_id)], id_to_index[int(tgt_id)], label, w_in))

        # Build row-stochastic transition matrix (by source -> targets) with weights
        if n == 0:
            ctx.event_ratings = []
            return ctx

        M = np.zeros((n, n), dtype=np.float64)
        out_weight_sum = np.zeros(n, dtype=np.float64)

        for i, j, label, w_in in edges:
            M[i, j] += w_in
            out_weight_sum[i] += w_in

        for i in range(n):
            if out_weight_sum[i] > 0:
                M[i, :] /= out_weight_sum[i]

        # PageRank power iteration with damping + dangling handling
        d = self.damping
        v = np.ones(n, dtype=np.float64) / n
        r = np.ones(n, dtype=np.float64) / n

        for _ in range(self.max_iter):
            r_prev = r
            dangling = (out_weight_sum == 0).astype(np.float64)
            dangling_mass = (r_prev * dangling).sum()
            r = d * (r_prev @ M + dangling_mass * v) + (1 - d) * v
            if np.abs(r - r_prev).sum() < self.tol:
                break

        # Outgoing bonus
        bonus = np.zeros(n, dtype=np.float64)
        for i, j, label, _w_in in edges:
            bw = float(self.relation_weight_out.get(label, 0.0))
            if bw > 0:
                bonus[i] += bw
        if bonus.max() > 0:
            bonus = bonus / bonus.max()

        pr_norm = r / r.sum() if r.sum() > 0 else r
        final_score = self.mix_weight * pr_norm + (1.0 - self.mix_weight) * bonus

        # Write back
        id_to_score = {ids[i]: float(final_score[i]) for i in range(n)}
        id_to_pr = {ids[i]: float(pr_norm[i]) for i in range(n)}
        id_to_bonus = {ids[i]: float(bonus[i]) for i in range(n)}

        for ev in events:
            eid = int(ev["id"])
            ev.setdefault("rating", {})
            ev["rating"]["pagerank"] = id_to_pr.get(eid, 0.0)
            ev["rating"]["outgoing_bonus"] = id_to_bonus.get(eid, 0.0)
            ev["rating"]["score"] = id_to_score.get(eid, 0.0)

        # Sorted summary for convenience
        ratings = [
            {
                "id": int(ids[i]),
                "pagerank": float(pr_norm[i]),
                "outgoing_bonus": float(bonus[i]),
                "score": float(final_score[i]),
            }
            for i in range(n)
        ]
        ratings.sort(key=lambda x: x["score"], reverse=True)

        ctx.event_ratings = ratings

        # --- Persist to DB (save ratings for this query) ---
        try:
            event_row_id = getattr(ctx, "event_row_id", None)
            storage = getattr(ctx, "chunk_storage", None)
            if storage is not None and event_row_id is not None:
                storage.insert_ratings(event_row_id, ratings)
                logging.info("Saved %d event ratings to DB for event_row_id=%s.", len(ratings), event_row_id)
            else:
                if storage is None:
                    logging.warning("No chunk_storage in ctx — ratings not persisted.")
                if event_row_id is None:
                    logging.warning("No event_row_id in ctx — ratings not persisted.")
        except Exception as e:
            logging.error(f"Failed to persist event ratings: {e}")

        logging.info("EventRatingStep finished. Rated %d events.", n)
        return ctx
