from __future__ import annotations
from .step import Step, RetrieveContext

from typing import List, Dict, Any, Optional, Sequence
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ..data_plotter.silhouette_curve_plotter import SilhouetteCurvePlotter


class ChunkClusteringStep(Step):
    """
    Loads chunk embeddings, selects k via silhouette score, performs KMeans,
    and stores clusters into ctx. Also plots silhouette score vs k if data_path is provided.
    """

    def __init__(self):
        self.last_silhouette = {"k": [], "score": []}

    def run(self, ctx: RetrieveContext) -> RetrieveContext:
        logging.info("Starting ChunkPreparationStep: running k-means clustering with silhouette-based k selection...")

        clusters: List[List[Dict[str, Any]]] = self.cluster_chunks_kmeans(
            k_min=10,
            k_max=60,
            normalize=True,
            n_init=8,
            max_iter=300,
            random_state=42,
            silhouette_sample_size=10000,
            plot_filename="silhouette_curve.png",
            ctx=ctx
        )

        logging.info(f"Clustering finished: {len(clusters)} clusters created.")

        ctx.clusters = clusters
        ctx.meta["chunk_preparation"] = {
            "n_clusters": len(clusters),
            "sizes": [len(c) for c in clusters],
            "silhouette_curve": self.last_silhouette,  # keep scores for later inspection
        }
        return ctx

    def cluster_chunks_kmeans(
        self,
        *,
        k_min: int = 4,
        k_max: int = 10,
        normalize: bool = True,
        n_init: int = 8,
        max_iter: int = 300,
        random_state: int = 42,
        silhouette_sample_size: int = 10000,
        plot_filename: str = "silhouette_curve.png",
        ctx: RetrieveContext = None
    ) -> List[List[Dict[str, Any]]]:
        """Cluster chunks using KMeans; choose k via silhouette; optionally plot the curve."""
        logging.info("Loading chunks from storage...")
        chunks: List[Dict[str, Any]] = ctx.chunk_storage.get_all_chunks()
        logging.info(f"Total chunks loaded: {len(chunks)}")

        # Collect embeddings and keep index mapping back to chunks
        idx_map: List[int] = []
        embs: List[np.ndarray] = []
        for i, ch in enumerate(chunks):
            e = ch.get("embedding")
            if e is None:
                continue
            if not isinstance(e, np.ndarray):
                e = np.asarray(e, dtype=np.float32)
            else:
                e = e.astype(np.float32, copy=False)
            embs.append(e)
            idx_map.append(i)

        logging.info(f"Chunks with embeddings: {len(embs)} / {len(chunks)}")

        if len(embs) == 0:
            logging.info("No chunks with embeddings found — returning empty list.")
            return []
        if len(embs) == 1:
            logging.info("Only one chunk with embedding found — single cluster returned.")
            return [[chunks[idx_map[0]]]]

        X = np.vstack(embs)

        # Normalize so Euclidean ≈ Cosine
        if normalize:
            logging.info("Normalizing embeddings (L2)...")
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            X = X / norms

        N = X.shape[0]
        k_max_eff = max(k_min, min(k_max, N - 1))
        if k_max_eff < 2:
            logging.info("Not enough points for more than one cluster — returning all in one cluster.")
            return [[chunks[i] for i in idx_map]]

        # Fixed subsample for comparable silhouette across k
        if N > silhouette_sample_size:
            rng = np.random.default_rng(random_state)
            sample_idx = rng.choice(N, size=silhouette_sample_size, replace=False)
            X_s = X[sample_idx]
            logging.info(f"Using silhouette subsample: {len(sample_idx)} of {N}")
        else:
            sample_idx = None
            X_s = X
            logging.info(f"Using all {N} points for silhouette scoring.")

        def _silhouette_safe(X_full, labels_full, X_sub, labels_sub) -> Optional[float]:
            """Compute silhouette score if valid; return None if invalid."""
            for Xc, lc, tag in ((X_sub, labels_sub, "subsample"), (X_full, labels_full, "full")):
                n = len(lc)
                if n < 3:
                    logging.info(f"Silhouette skipped on {tag}: n={n} < 3.")
                    continue
                n_labels = len(set(lc.tolist() if hasattr(lc, 'tolist') else lc))
                if n_labels < 2:
                    logging.info(f"Silhouette skipped on {tag}: only {n_labels} label(s).")
                    continue
                if n_labels > n - 1:
                    logging.info(f"Silhouette skipped on {tag}: labels={n_labels} > n-1={n - 1}.")
                    continue
                try:
                    return float(silhouette_score(Xc, lc, metric="euclidean"))
                except Exception as e:
                    logging.info(f"Silhouette failed on {tag}: {e}")
                    continue
            return None

        # Sweep k and compute silhouette
        best_k: Optional[int] = None
        best_score: float = -1.0
        self.last_silhouette = {"k": [], "score": []}  # reset for this run
        logging.info(f"Selecting best k in range [{k_min}, {k_max_eff}] using silhouette score...")

        for k in range(k_min, k_max_eff + 1):
            km = KMeans(
                n_clusters=k,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state,
            )
            labels_full = km.fit_predict(X)
            labels_sub = labels_full[sample_idx] if sample_idx is not None else labels_full

            score = _silhouette_safe(X, labels_full, X_s, labels_sub)
            if score is None:
                logging.info(f"k={k}: silhouette not computable — skipping (score = -inf).")
                score_val = float("-inf")
            else:
                score_val = score
                logging.info(f"k={k} silhouette score: {score_val:.4f}")

            # record for plotting
            self.last_silhouette["k"].append(k)
            self.last_silhouette["score"].append(score_val)

            if score_val > best_score:
                best_score = score_val
                best_k = k

        # Fallback if nothing valid
        if best_k is None or best_score == float("-inf"):
            best_k = max(2, min(k_max_eff, 2))
            logging.warning(f"No valid silhouette score across k — falling back to k={best_k}")

        # Log best
        sil_str = "nan" if best_score == float("-inf") else f"{best_score:.4f}"
        logging.info(f"Best k selected: {best_k} (silhouette={sil_str})")

        # Plot silhouette curve if a data path is available
        if len(self.last_silhouette["k"]) > 0:
            try:
                plotter = SilhouetteCurvePlotter(ctx.working_dir)
                path = plotter.plot_curve(
                    ks=self.last_silhouette["k"],
                    scores=self.last_silhouette["score"],
                    best_k=best_k,
                    filename=plot_filename,
                    with_timestamp=True,
                    show=False,
                )
                logging.info(f"Saved silhouette curve plot to: {path}")
            except Exception as e:
                logging.warning(f"Could not save silhouette curve plot: {e}")

        # Final KMeans with best_k
        logging.info("Running final KMeans with best_k...")
        km_final = KMeans(
            n_clusters=best_k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        final_labels = km_final.fit_predict(X)

        clusters: List[List[Dict[str, Any]]] = [[] for _ in range(best_k)]
        for local_i, lab in enumerate(final_labels):
            orig_idx = idx_map[local_i]
            clusters[lab].append(chunks[orig_idx])

        logging.info("Final clusters built: " + ", ".join(f"{len(c)} items" for c in clusters))
        return clusters
