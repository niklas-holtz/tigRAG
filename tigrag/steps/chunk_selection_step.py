from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import logging
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    # Fallback no-op tqdm
    def tqdm(it=None, *args, **kwargs):
        return it if it is not None else range(0)

from ..steps.step import Step, RetrieveContext


class ChunkSelectionStep(Step):
    """Global MMR selection with adaptive (cluster-aware) quotas and practical improvements.

    Key points
    ----------
    - The query embedding is generated at runtime via `ctx.embedding_invoker(sentences=[ctx.query])`.
    - No centroid fallback is used; relevance is always computed to the query.
    - Optional soft quotas, per-document caps, diagnostics, and final ordering controls.
    """

    def __init__(
        self,
        # Quotas & diversity
        lam: float = 0.7,
        alpha: float = 0.5,
        cluster_topk_for_weight: int = 3,
        weight_temperature: float = 0.5,
        # Hard vs. soft quotas
        use_soft_quotas: bool = False,
        overflow_penalty: float = 0.1,  # applied per unit over quota if soft quotas enabled
        per_doc_max: Optional[int] = None,  # e.g., max 3 chunks per document
        # Top-N control
        auto_topn: bool = True,
        min_n: int = 40,
        max_n: int = 256,
        top_n: int = 20,  # only used if auto_topn=False
        # Early stopping
        mmr_gain_window: int = 3,
        mmr_gain_drop: float = 0.15,
        mmr_min_gain_abs: float = 0.005,
        sim_floor_quantile: float = 0.70,
        sim_floor_abs: Optional[float] = None,
        stop_if_mmr_negative: bool = True,
        # Scoring tweaks
        zscore_relevance: bool = False,
        # Output ordering
        final_order: str = "mmr",  # "mmr" | "similarity" | "blend"
        final_blend_weight: float = 0.5,  # for "blend": w * sim + (1-w) * mmr
        # UX / Logging
        verbose: bool = False,
    ):
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.cluster_topk_for_weight = int(cluster_topk_for_weight)
        self.weight_temperature = float(weight_temperature)

        self.use_soft_quotas = bool(use_soft_quotas)
        self.overflow_penalty = float(overflow_penalty)
        self.per_doc_max = None if per_doc_max is None else int(per_doc_max)

        self.auto_topn = bool(auto_topn)
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.top_n = int(top_n)

        self.mmr_gain_window = int(mmr_gain_window)
        self.mmr_gain_drop = float(mmr_gain_drop)
        self.mmr_min_gain_abs = float(mmr_min_gain_abs)
        self.sim_floor_quantile = float(sim_floor_quantile)
        self.sim_floor_abs = None if sim_floor_abs is None else float(sim_floor_abs)
        self.stop_if_mmr_negative = bool(stop_if_mmr_negative)

        self.zscore_relevance = bool(zscore_relevance)

        assert final_order in {"mmr", "similarity", "blend"}
        self.final_order = final_order
        self.final_blend_weight = float(final_blend_weight)

        self.verbose = bool(verbose)

    # ---------------------------
    # Public API
    # ---------------------------
    def run(self, ctx: RetrieveContext) -> RetrieveContext:
        logging.info("Starting ChunkSelectionStep v2…")
        debug: Dict[str, object] = {}

        # 0) Validate input clusters
        if not hasattr(ctx, "clusters") or not ctx.clusters:
            logging.warning("No clusters in context — returning empty selection.")
            ctx.selected = []
            ctx.debug = {"stop_reason": "no_clusters"}
            return ctx

        # 1) Query embedding via ctx.embedding_invoker (REQUIRED)
        if not hasattr(ctx, "embedding_invoker"):
            raise ValueError("ctx.embedding_invoker is required to generate q_emb")
        q_text = getattr(ctx, "query", None)
        if not q_text:
            raise ValueError("ctx.query is required to generate q_emb")
        q_emb = self._to_np(ctx.embedding_invoker(sentences=[q_text])[0]).reshape(-1)

        # 2) Gather normalized embeddings/items per cluster
        cluster_embs: List[np.ndarray] = []
        cluster_items: List[List[dict]] = []
        cluster_sizes: List[int] = []

        for cluster in tqdm(ctx.clusters, total=len(ctx.clusters), desc="Reading clusters", unit="cluster", disable=not self.verbose):
            embs, items = [], []
            for chunk in cluster:
                e = chunk.get("embedding")
                if e is None:
                    continue
                e = self._l2norm(self._to_np(e))
                embs.append(e)
                items.append(chunk)
            E = np.vstack(embs).astype(np.float32, copy=False) if embs else np.zeros((0, 1), dtype=np.float32)
            cluster_embs.append(E)
            cluster_items.append(items)
            cluster_sizes.append(int(E.shape[0]))

        total_candidates = int(sum(cluster_sizes))
        if total_candidates == 0:
            logging.warning("No candidate embeddings — returning empty selection.")
            ctx.selected = []
            ctx.debug = {"stop_reason": "no_candidates"}
            return ctx

        # 3) Flatten for global selection
        E, idx2cluster, idx2local = self._flatten(cluster_embs)

        # 4) Compute relevance (cosine to q_emb)
        sims_q = self._compute_relevance(E, q_emb)

        # Optional: z-score normalize relevance to stabilize floors/fusion
        if self.zscore_relevance:
            mu, sigma = float(sims_q.mean()), float(sims_q.std() + 1e-6)
            sims_q = (sims_q - mu) / sigma
            debug.update({"zscore_mu": mu, "zscore_sigma": sigma})

        # 5) Similarity floor (initial)
        if self.sim_floor_abs is not None:
            sim_floor = float(self.sim_floor_abs)
        else:
            q = np.clip(self.sim_floor_quantile, 0.0, 1.0)
            sim_floor = float(np.quantile(sims_q, q)) if total_candidates > 0 else -1.0
        debug["initial_sim_floor"] = sim_floor

        # 6) Determine working N and quotas
        working_top_n = min(self.max_n if self.auto_topn else self.top_n, total_candidates)

        quotas = self._compute_cluster_quotas(
            sims_q=sims_q,
            idx2cluster=idx2cluster,
            cluster_sizes=np.asarray(cluster_sizes, dtype=np.int32),
            top_n=working_top_n,
            alpha=self.alpha,
            topk=self.cluster_topk_for_weight,
            temperature=self.weight_temperature,
        )
        debug["quotas"] = quotas.tolist()

        # 7) Global MMR with quotas
        selected_indices, mmr_at_pick, stop_reason, aux = self._select_with_global_mmr(
            E=E,
            sims_q=sims_q,
            idx2cluster=idx2cluster,
            quotas=quotas,
            top_n=working_top_n,
            lam=self.lam,
            auto_topn=self.auto_topn,
            min_n=self.min_n,
            max_n=working_top_n,
            mmr_gain_window=self.mmr_gain_window,
            mmr_gain_drop=self.mmr_gain_drop,
            mmr_min_gain_abs=self.mmr_min_gain_abs,
            sim_floor=sim_floor,
            stop_if_mmr_negative=self.stop_if_mmr_negative,
            use_soft_quotas=self.use_soft_quotas,
            overflow_penalty=self.overflow_penalty,
            per_doc_max=self.per_doc_max,
            cluster_items=cluster_items,
            verbose=self.verbose,
        )

        debug.update(aux)
        debug["stop_reason"] = stop_reason
        debug["mmr_gains"] = [float(x) for x in mmr_at_pick]

        # 8) Materialize
        selected: List[dict] = []
        for j, idx in enumerate(selected_indices):
            c_id = int(idx2cluster[idx])
            l_idx = int(idx2local[idx])
            item = dict(cluster_items[c_id][l_idx])
            item["cluster_id"] = c_id
            item["similarity"] = float(sims_q[idx])
            item["mmr_score"] = float(mmr_at_pick[j])
            selected.append(item)

        # 9) Final ordering
        if self.final_order == "mmr":
            # keep MMR pick order
            pass
        elif self.final_order == "similarity":
            selected.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        else:  # blend
            w = np.clip(self.final_blend_weight, 0.0, 1.0)
            for it in selected:
                it["final_score"] = w * it.get("similarity", 0.0) + (1.0 - w) * it.get("mmr_score", 0.0)
            selected.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        ctx.selected = selected
        ctx.debug = debug
        logging.info(f"ChunkSelectionStep v2 finished. Selected {len(selected)} chunks. Reason: {stop_reason}")
        return ctx

    # ---------------------------
    # Helper methods
    # ---------------------------

    @staticmethod
    def _to_np(x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        if v.ndim == 1:
            n = float(np.linalg.norm(v))
            return v if n == 0.0 else (v / n)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0.0] = 1.0
        return v / n

    @staticmethod
    def _flatten(cluster_embs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts, idx2cluster, idx2local = [], [], []
        for c_id, E in enumerate(cluster_embs):
            if E.size == 0:
                continue
            parts.append(E)
            idx2cluster.extend([c_id] * E.shape[0])
            idx2local.extend(list(range(E.shape[0])))
        E_all = np.vstack(parts).astype(np.float32, copy=False)
        return E_all, np.asarray(idx2cluster, dtype=np.int32), np.asarray(idx2local, dtype=np.int32)

    def _compute_relevance(
        self,
        E: np.ndarray,
        q_emb: np.ndarray,
    ) -> np.ndarray:
        """Cosine similarity to q_emb. q_emb is required; no centroid fallback."""
        q = self._l2norm(q_emb.reshape(-1))
        return E @ q

    def _compute_cluster_quotas(
        self,
        sims_q: np.ndarray,
        idx2cluster: np.ndarray,
        cluster_sizes: np.ndarray,
        top_n: int,
        alpha: float,
        topk: int,
        temperature: float,
    ) -> np.ndarray:
        """Compute per-cluster quotas blending relevance-proportional weights with a uniform prior.

        Steps:
          1) For each cluster, score = mean of its top-k sims (k clipped by cluster size).
          2) Softmax (temperature) over scores -> weights.
          3) Blend with uniform via alpha.
          4) Allocate integer quotas using Hamilton's method, capped by capacity.
        """
        C = int(cluster_sizes.shape[0])
        if C == 0 or top_n <= 0:
            return np.zeros_like(cluster_sizes)

        # 1) Top-k mean per cluster
        scores = np.full(C, -1e9, dtype=np.float32)
        for c in range(C):
            mask = (idx2cluster == c)
            if not np.any(mask):
                continue
            sims_c = sims_q[mask]
            k = int(min(topk, sims_c.shape[0]))
            if k <= 0:
                continue
            top_vals = np.partition(sims_c, -k)[-k:]
            scores[c] = float(np.mean(top_vals))

        # Valid clusters
        valid_mask = (scores > -1e9 / 2)
        non_empty = np.where(valid_mask & (cluster_sizes > 0))[0]
        if non_empty.size == 0:
            return np.zeros_like(cluster_sizes)

        # 2) Softmax with temperature
        s = scores[non_empty] / max(temperature, 1e-6)
        s = s - s.max()
        w = np.exp(s)
        w = w / w.sum()

        weights = np.zeros(C, dtype=np.float32)
        weights[non_empty] = w

        # 3) Blend with uniform prior
        uniform = np.zeros(C, dtype=np.float32)
        uniform[non_empty] = 1.0 / non_empty.size
        blended = alpha * weights + (1.0 - alpha) * uniform

        # 4) Hamilton allocation (largest remainder), capped by capacity
        fractional = top_n * blended
        base = np.floor(fractional).astype(np.int32)
        base = np.minimum(base, cluster_sizes)

        remaining = top_n - int(base.sum())
        if remaining > 0:
            remainders = fractional - base
            capacity_left = cluster_sizes - base
            # Sort by (remainder desc, score desc)
            order = np.lexsort((-scores, -remainders))
            for c in order:
                if remaining == 0:
                    break
                if capacity_left[c] > 0 and blended[c] > 0:
                    base[c] += 1
                    capacity_left[c] -= 1
                    remaining -= 1

        # Final safety pass if capacity left somewhere
        if remaining > 0:
            for c in non_empty:
                if remaining == 0:
                    break
                room = int(cluster_sizes[c] - base[c])
                if room > 0:
                    take = min(room, remaining)
                    base[c] += take
                    remaining -= take

        return base

    def _select_with_global_mmr(
        self,
        E: np.ndarray,
        sims_q: np.ndarray,
        idx2cluster: np.ndarray,
        quotas: np.ndarray,
        top_n: int,
        lam: float,
        *,
        auto_topn: bool,
        min_n: int,
        max_n: int,
        mmr_gain_window: int,
        mmr_gain_drop: float,
        mmr_min_gain_abs: float,
        sim_floor: float,
        stop_if_mmr_negative: bool,
        use_soft_quotas: bool,
        overflow_penalty: float,
        per_doc_max: Optional[int],
        cluster_items: List[List[dict]],
        verbose: bool,
    ) -> Tuple[List[int], List[float], str, Dict[str, object]]:
        """MMR with (soft) cluster quotas, optional per-document cap, and adaptive floor.

        Returns
        -------
        selected_indices, mmr_at_pick, stop_reason, aux_debug
        """
        N = E.shape[0]
        selected_indices: List[int] = []
        mmr_at_pick: List[float] = []
        picked_mask = np.zeros(N, dtype=bool)
        cluster_taken = np.zeros(quotas.shape[0], dtype=np.int32)
        max_sim_to_S = np.zeros(N, dtype=np.float32)
        doc_ids = self._gather_doc_ids(cluster_items)
        doc_taken: Dict[object, int] = {}

        eff_max_n = min(max_n, top_n, N) if auto_topn else min(top_n, N)
        eff_min_n = max(1, min_n) if auto_topn else min(top_n, N)

        gains_history: List[float] = []
        stop_reason = "max_n"

        pbar = tqdm(total=eff_max_n, desc="Selecting via global MMR", unit="pick", disable=not verbose)

        while len(selected_indices) < eff_max_n:
            # Allowed mask
            allowed = (~picked_mask)
            # Hard cluster quota gate (for candidate set); soft quota will add penalties later
            if not use_soft_quotas:
                allowed &= (cluster_taken[idx2cluster] < quotas[idx2cluster])

            # Per-document cap
            if per_doc_max is not None:
                # Block candidates whose doc already reached cap
                over_docs = np.array([doc_taken.get(d, 0) >= per_doc_max for d in doc_ids], dtype=bool)
                allowed &= ~over_docs

            allowed_idx = np.where(allowed)[0]
            if allowed_idx.size == 0:
                stop_reason = "quota_exhausted"
                break

            # Adaptive similarity floor: recompute on the currently allowed set (more precise late in the process)
            current_sim_floor = sim_floor
            if self.sim_floor_abs is None and allowed_idx.size > 0:
                q = np.clip(self.sim_floor_quantile, 0.0, 1.0)
                current_sim_floor = float(np.quantile(sims_q[allowed_idx], q))

            # Base MMR
            current_penalty = max_sim_to_S[allowed_idx]
            mmr_scores = lam * sims_q[allowed_idx] - (1.0 - lam) * current_penalty

            # Soft-quota overflow penalty
            if use_soft_quotas:
                overflow = np.maximum(0, (cluster_taken[idx2cluster[allowed_idx]] + 1) - quotas[idx2cluster[allowed_idx]])
                mmr_scores -= overflow_penalty * overflow.astype(np.float32)

            # Choose best candidate
            best_pos = int(np.argmax(mmr_scores))
            best_idx = int(allowed_idx[best_pos])
            best_gain = float(mmr_scores[best_pos])
            best_sim = float(sims_q[best_idx])

            # Early stop checks (after min_n)
            if auto_topn and len(selected_indices) >= eff_min_n:
                stop_rel = False
                if mmr_gain_window > 0 and mmr_gain_drop > 0 and len(gains_history) >= mmr_gain_window:
                    recent_avg = float(np.mean(gains_history[-mmr_gain_window:]))
                    baseline = abs(gains_history[0]) if gains_history else abs(best_gain) + 1e-6
                    stop_rel = (recent_avg <= mmr_gain_drop * baseline)

                stop_abs = (best_gain <= mmr_min_gain_abs)
                stop_neg = (best_gain <= 0.0) if stop_if_mmr_negative else False
                stop_sim = (best_sim <= current_sim_floor)

                if stop_rel or stop_abs or stop_neg or stop_sim:
                    stop_reason = (
                        "gain_drop" if stop_rel else
                        "min_gain" if stop_abs else
                        "negative" if stop_neg else
                        "sim_floor"
                    )
                    break

            # Commit the pick
            selected_indices.append(best_idx)
            mmr_at_pick.append(best_gain)
            picked_mask[best_idx] = True
            c = int(idx2cluster[best_idx])
            cluster_taken[c] += 1

            # Update per-doc counter
            d_id = doc_ids[best_idx]
            doc_taken[d_id] = doc_taken.get(d_id, 0) + 1

            # Update penalties with the newly selected vector
            v = E @ E[best_idx]
            np.maximum(max_sim_to_S, v, out=max_sim_to_S)

            # Track gain
            gains_history.append(best_gain)
            pbar.update(1)

        pbar.close()

        aux = {
            "cluster_taken": [int(x) for x in cluster_taken.tolist()],
            "doc_taken": {str(k): int(v) for k, v in doc_taken.items()},
            "final_allowed_floor": float(current_sim_floor) if 'current_sim_floor' in locals() else float(sim_floor),
        }
        return selected_indices, mmr_at_pick, stop_reason, aux

    @staticmethod
    def _gather_doc_ids(cluster_items: List[List[dict]]) -> np.ndarray:
        """Collect a flat list of doc_ids aligned with the flattened E. None if missing."""
        ids: List[object] = []
        for items in cluster_items:
            for it in items:
                ids.append(it.get("doc_id"))
        return np.asarray(ids, dtype=object)
