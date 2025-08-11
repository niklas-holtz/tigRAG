from __future__ import annotations
from ..steps.step import Step, RetrieveContext
from typing import Optional
import logging
from tqdm import tqdm
import numpy as np

class ChunkSelectionStep(Step):
    """
    Selects relevant chunks from prepared clusters or directly from storage
    based on the current query in RetrieveContext.
    """

    def __init__(self):
        pass

    def run(self, ctx: RetrieveContext) -> RetrieveContext:
        logging.info("Starting ChunkSelectionStep...")

        if not hasattr(ctx, "clusters") or not ctx.clusters:
            logging.warning("No clusters found in context — skipping selection.")
            ctx.selected = []
            return ctx

        # Query-Embedding aus ctx.meta['q_emb'] ziehen (falls vorhanden)
        q_emb = None
        if getattr(ctx, "meta", None) and "q_emb" in ctx.meta:
            q_emb = self._to_np(ctx.meta["q_emb"])
            q_emb = self._l2norm(q_emb)

        top_n = 3
        lam = 0.7
        selected = []

        for cluster_idx, cluster in tqdm(
            enumerate(ctx.clusters, start=1),
            total=len(ctx.clusters),
            desc="Processing clusters",
            unit="cluster"
        ):
            logging.info(f"Processing cluster {cluster_idx} with {len(cluster)} chunks...")

            # Embeddings sammeln
            embs, items = [], []
            for chunk in cluster:
                e = chunk.get("embedding")
                if e is None:
                    continue
                e = self._l2norm(self._to_np(e))
                embs.append(e)
                items.append(chunk)

            if not embs:
                logging.info(f"Cluster {cluster_idx}: no embeddings — skipping.")
                continue

            E = np.vstack(embs)  # (N, D)

            if q_emb is None:
                # Fallback: sortiere nach mean similarity (repräsentativste zuerst)
                sims_mean = (E @ E.T).mean(axis=1)
                order = np.argsort(-sims_mean)[:top_n]
                for idx in order:
                    item = dict(items[idx])
                    item["mmr_score"] = float(sims_mean[idx])
                    item["cluster_id"] = cluster_idx
                    selected.append(item)
                continue

            # Cosine Similarities
            sims_q = E @ q_emb
            sim_mat = E @ E.T

            chosen = []
            candidates = list(range(E.shape[0]))
            k = min(top_n, len(candidates))

            for _ in range(k):
                if not chosen:
                    best_idx = int(np.argmax(sims_q[candidates]))
                    chosen.append(candidates.pop(best_idx))
                    continue
                best_score, best_cand_idx = -1e9, None
                for cand_pos, cand in enumerate(candidates):
                    div = max(sim_mat[cand, sel] for sel in chosen) if chosen else 0.0
                    score = lam * sims_q[cand] - (1.0 - lam) * div
                    if score > best_score:
                        best_score = score
                        best_cand_idx = cand_pos
                chosen.append(candidates.pop(best_cand_idx))

            for idx in chosen:
                item = dict(items[idx])
                div = max(sim_mat[idx, s] for s in chosen if s != idx) if len(chosen) > 1 else 0.0
                item["similarity"] = float(sims_q[idx])
                item["mmr_score"] = float(lam * sims_q[idx] - (1.0 - lam) * div)
                item["cluster_id"] = cluster_idx
                selected.append(item)

        ctx.selected = selected
        logging.info(f"ChunkSelectionStep finished. Selected {len(selected)} chunks.")
        return ctx

    @staticmethod
    def _to_np(x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        if v.ndim == 1:
            n = np.linalg.norm(v)
            return v if n == 0.0 else v / n
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0.0] = 1.0
        return v / n