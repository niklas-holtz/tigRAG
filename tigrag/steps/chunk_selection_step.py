from __future__ import annotations
from ..steps.step import Step, RetrieveContext
from typing import Optional, Dict, List, Tuple
import logging
from tqdm import tqdm
import numpy as np

class ChunkSelectionStep(Step):
    """
    Selects top-N chunks using global MMR with an adaptive, cluster-aware quota.
    - Relevance: cosine similarity to q_emb
    - Diversity: MMR penalty vs. already selected items
    - Distribution: adaptive per-cluster quota so results don't come from a single cluster

    Token-agnostic:
    - Fixed top_n, or
    - auto_topn=True: stop by diminishing MMR gains + similarity floors (no token estimates).
    """

    def __init__(
        self,
        top_n: int = 20,
        lam: float = 0.7,
        alpha: float = 0.5,
        cluster_topk_for_weight: int = 3,
        weight_temperature: float = 0.5,
        # auto-topN purely by diminishing returns (no tokens)
        auto_topn: bool = True,
        min_n: int = 40,
        max_n: int = 256,
        mmr_gain_window: int = 3,
        mmr_gain_drop: float = 0.15,
        # additional robust stop criteria
        mmr_min_gain_abs: float = 0.005,      # stop if best MMR gain <= this (after min_n)
        sim_floor_quantile: float = 0.70,     # stop if next pick's sim <= quantile floor (after min_n)
        sim_floor_abs: Optional[float] = None,# optional absolute sim floor (overrides quantile if set)
        stop_if_mmr_negative: bool = True,    # stop if best MMR gain <= 0 (after min_n)
    ):
        """
        Args:
            top_n: total picks when auto_topn is False.
            lam: lambda in MMR.
            alpha: blend factor for per-cluster quota (0 uniform, 1 relevance-weighted).
            cluster_topk_for_weight: top-k to average per cluster when computing relevance weight.
            weight_temperature: softmax temperature for cluster weights.
            auto_topn: enable early stopping without tokens.
            min_n: minimum picks before any early stop.
            max_n: hard cap when auto_topn=True.
            mmr_gain_window: moving average window for recent MMR gains.
            mmr_gain_drop: stop when recent_avg <= mmr_gain_drop * first_gain.
            mmr_min_gain_abs: absolute MMR floor to stop.
            sim_floor_quantile: percentile of sims_q used as similarity floor.
            sim_floor_abs: absolute similarity floor (if provided).
            stop_if_mmr_negative: whether to stop when best MMR <= 0 (after min_n).
        """
        self.top_n = int(top_n)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.cluster_topk_for_weight = int(cluster_topk_for_weight)
        self.weight_temperature = float(weight_temperature)

        self.auto_topn = bool(auto_topn)
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.mmr_gain_window = int(mmr_gain_window)
        self.mmr_gain_drop = float(mmr_gain_drop)

        self.mmr_min_gain_abs = float(mmr_min_gain_abs)
        self.sim_floor_quantile = float(sim_floor_quantile)
        self.sim_floor_abs = None if sim_floor_abs is None else float(sim_floor_abs)
        self.stop_if_mmr_negative = bool(stop_if_mmr_negative)

    def run(self, ctx: RetrieveContext) -> RetrieveContext:
        logging.info("Starting ChunkSelectionStep (global MMR with adaptive cluster quotas)...")

        if not hasattr(ctx, "clusters") or not ctx.clusters:
            logging.warning("No clusters found in context — returning empty selection.")
            ctx.selected = []
            return ctx

        # Pull and normalize q_emb (query embedding)
        q_emb = None
        if getattr(ctx, "meta", None) and "q_emb" in ctx.meta:
            q_emb = self._l2norm(self._to_np(ctx.meta["q_emb"]))

        # Gather normalized embeddings and metadata from all clusters
        cluster_embs: List[np.ndarray] = []
        cluster_items: List[List[dict]] = []
        cluster_sizes: List[int] = []

        for cluster_idx, cluster in tqdm(
            enumerate(ctx.clusters, start=0),
            total=len(ctx.clusters),
            desc="Reading clusters",
            unit="cluster",
        ):
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
            cluster_sizes.append(E.shape[0])

        total_candidates = int(sum(cluster_sizes))
        if total_candidates == 0:
            logging.warning("No candidate embeddings found — returning empty selection.")
            ctx.selected = []
            return ctx

        # Flatten for global selection
        E, idx2cluster, idx2local = self._flatten(cluster_embs)

        # Relevance: sim to q_emb (cosine). If missing, fallback to centroid.
        if q_emb is None:
            logging.warning("q_emb not found — falling back to centroid similarity as relevance proxy.")
            centroid = self._l2norm(E.mean(axis=0))
            sims_q = E @ centroid
        else:
            sims_q = E @ q_emb

        # Compute similarity floor for early stopping
        if self.sim_floor_abs is not None:
            sim_floor = float(self.sim_floor_abs)
        else:
            q = np.clip(self.sim_floor_quantile, 0.0, 1.0)
            sim_floor = float(np.quantile(sims_q, q)) if total_candidates > 0 else -1.0

        # Determine working N for quota calculation (cap by candidates)
        working_top_n = min(self.max_n if self.auto_topn else self.top_n, total_candidates)

        # Compute adaptive per-cluster quotas
        quotas = self._compute_cluster_quotas(
            sims_q=sims_q,
            idx2cluster=idx2cluster,
            cluster_sizes=np.array(cluster_sizes, dtype=np.int32),
            top_n=working_top_n,
            alpha=self.alpha,
            topk=self.cluster_topk_for_weight,
            temperature=self.weight_temperature,
        )

        # Global MMR selection with cluster quotas (+ early stop options)
        selected_indices, mmr_at_pick = self._select_with_global_mmr(
            E=E,
            sims_q=sims_q,
            idx2cluster=idx2cluster,
            quotas=quotas,
            top_n=working_top_n,
            lam=self.lam,
            auto_topn=self.auto_topn,
            min_n=self.min_n,
            max_n=working_top_n,                 # hard cap never exceeds candidates
            mmr_gain_window=self.mmr_gain_window,
            mmr_gain_drop=self.mmr_gain_drop,
            mmr_min_gain_abs=self.mmr_min_gain_abs,
            sim_floor=sim_floor,
            stop_if_mmr_negative=self.stop_if_mmr_negative,
        )

        # Materialize selected items
        selected: List[dict] = []
        for j, idx in enumerate(selected_indices):
            c_id = idx2cluster[idx]
            l_idx = idx2local[idx]
            item = dict(cluster_items[c_id][l_idx])
            item["cluster_id"] = int(c_id)
            item["similarity"] = float(sims_q[idx])
            item["mmr_score"] = float(mmr_at_pick[j])
            selected.append(item)

        # Final presentation: sort by similarity to q_emb (descending)
        selected.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        ctx.selected = selected
        logging.info(f"ChunkSelectionStep finished. Selected {len(selected)} chunks.")
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
            n = np.linalg.norm(v)
            return v if n == 0.0 else v / n
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0.0] = 1.0
        return v / n

    @staticmethod
    def _flatten(cluster_embs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Flattens per-cluster embeddings to a single matrix E and returns
        mapping arrays idx2cluster and idx2local.
        """
        parts, idx2cluster, idx2local = [], [], []
        for c_id, E in enumerate(cluster_embs):
            if E.size == 0:
                continue
            parts.append(E)
            idx2cluster.extend([c_id] * E.shape[0])
            idx2local.extend(list(range(E.shape[0])))
        E_all = np.vstack(parts).astype(np.float32, copy=False)
        return E_all, np.asarray(idx2cluster, dtype=np.int32), np.asarray(idx2local, dtype=np.int32)

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
        """
        Computes per-cluster quotas using a blend of relevance-proportional weights and a uniform prior.
        Steps:
          1) For each cluster, score = mean of its top-k sims (k clipped by cluster size).
          2) Convert scores → softmax weights (temperature-scaled).
          3) Blend with uniform prior via alpha.
          4) Allocate integer quotas using Hamilton's method (largest remainders), capped by cluster capacity.
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

        # Use only non-empty clusters
        valid_mask = (scores > -1e9 / 2)
        non_empty = np.where(valid_mask & (cluster_sizes > 0))[0]
        if non_empty.size == 0:
            return np.zeros_like(cluster_sizes)

        # 2) Softmax with temperature over valid clusters
        s = scores[non_empty] / max(temperature, 1e-6)
        s = s - s.max()
        w = np.exp(s)
        w = w / w.sum()

        weights = np.zeros(C, dtype=np.float32)
        weights[non_empty] = w

        # 3) Blend with uniform prior over non-empty clusters
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

        # If still remaining due to capacity exhaustion, spread greedily
        if remaining > 0:
            for c in non_empty:
                if remaining == 0:
                    break
                room = cluster_sizes[c] - base[c]
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
    ) -> Tuple[List[int], List[float]]:
        """
        Global MMR with per-cluster quota constraints.
        Early stopping (token-free):
          - after reaching min_n, stop if ANY of these holds:
            (a) recent_avg_gain <= mmr_gain_drop * first_gain
            (b) best_gain <= mmr_min_gain_abs
            (c) stop_if_mmr_negative and best_gain <= 0
            (d) next candidate's similarity <= sim_floor
          - always enforce max_n cap when auto_topn=True
        """
        N = E.shape[0]
        selected_indices: List[int] = []
        mmr_at_pick: List[float] = []
        picked_mask = np.zeros(N, dtype=bool)
        cluster_taken = np.zeros(quotas.shape[0], dtype=np.int32)
        max_sim_to_S = np.zeros(N, dtype=np.float32)

        eff_max_n = min(max_n, top_n, N) if auto_topn else min(top_n, N)
        eff_min_n = max(1, min_n) if auto_topn else min(top_n, N)

        gains_history: List[float] = []
        pbar = tqdm(total=eff_max_n, desc="Selecting via global MMR", unit="pick")

        while len(selected_indices) < eff_max_n:
            allowed = (~picked_mask) & (cluster_taken[idx2cluster] < quotas[idx2cluster])
            allowed_idx = np.where(allowed)[0]
            if allowed_idx.size == 0:
                break

            # MMR(d) = λ * rel(d) - (1-λ) * max_{s∈S} sim(d, s)
            current_penalty = max_sim_to_S[allowed_idx]
            mmr_scores = lam * sims_q[allowed_idx] - (1.0 - lam) * current_penalty

            best_pos = int(np.argmax(mmr_scores))
            best_idx = int(allowed_idx[best_pos])
            best_gain = float(mmr_scores[best_pos])
            best_sim = float(sims_q[best_idx])

            # Commit the pick
            selected_indices.append(best_idx)
            mmr_at_pick.append(best_gain)
            picked_mask[best_idx] = True
            c = idx2cluster[best_idx]
            cluster_taken[c] += 1

            # Update penalties with the newly selected vector
            v = E @ E[best_idx]
            np.maximum(max_sim_to_S, v, out=max_sim_to_S)

            # Track gain
            gains_history.append(best_gain)
            pbar.update(1)

            # Early stopping checks after min_n
            if auto_topn and len(selected_indices) >= eff_min_n:
                stop_rel = False
                if mmr_gain_window > 0 and mmr_gain_drop > 0 and len(gains_history) >= mmr_gain_window + 1:
                    W = mmr_gain_window
                    recent_avg = float(np.mean(gains_history[-W:]))
                    baseline = abs(gains_history[0]) if gains_history[0] != 0 else 1e-6
                    stop_rel = (recent_avg <= mmr_gain_drop * baseline)

                stop_abs = (best_gain <= mmr_min_gain_abs)
                stop_neg = (best_gain <= 0.0) if stop_if_mmr_negative else False
                stop_sim = (best_sim <= sim_floor)

                if stop_rel or stop_abs or stop_neg or stop_sim:
                    logging.info(
                        "Early stop: rel=%s abs=%s neg=%s sim=%s "
                        "(recent_avg<=%.3f*first_gain | gain<=%.4f | gain<=0 | sim<=%.4f)",
                        stop_rel, stop_abs, stop_neg, stop_sim,
                        mmr_gain_drop, mmr_min_gain_abs, sim_floor
                    )
                    break

        pbar.close()
        return selected_indices, mmr_at_pick
