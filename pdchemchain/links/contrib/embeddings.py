"""
Dimensionality reduction / embedding links with external dependencies.

UMAPEmbedding requires: pip install umap-learn
tSNEEmbedding requires: pip install scikit-learn
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import DataStructs

# Numba emits extensive DEBUG-level output during JIT compilation.
# Suppress it here so pdchemchain's own DEBUG level doesn't flood the logs.
logging.getLogger("numba").setLevel(logging.WARNING)

from pdchemchain.base import Link
from pdchemchain.typing import InColumnName, Partitionable




def _fps_to_numpy(fps: list) -> np.ndarray:
    """Convert a list of RDKit ExplicitBitVect to a numpy uint8 array."""
    n_bits = fps[0].GetNumBits()
    X = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, X[i])
    return X


@dataclass
class UMAPEmbedding(Link):
    """2D UMAP embedding of molecular fingerprints

    Converts RDKit fingerprints to numpy arrays internally,
    then runs UMAP with Jaccard metric (appropriate for binary fingerprints).

    Requires the umap-learn package.

    Parameters
    ----------
    in_column
        Column containing RDKit ExplicitBitVect fingerprints
    out_column_prefix
        Prefix for the output columns (produces {prefix}_dim1 and {prefix}_dim2)
    n_neighbors
        UMAP n_neighbors parameter. Larger values capture more global structure.
    min_dist
        UMAP min_dist parameter. Smaller values create tighter clusters.
    random_state
        Random seed for reproducibility
    """

    _partitionable = Partitionable.NO

    in_column: InColumnName = "__MolFP__"
    out_column_prefix: str = "Umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: int = 42
    low_memory: bool = True

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "umap-learn is required for UMAPEmbedding. "
                "Install with: pip install umap-learn"
            ) from e

        fps = df[self.in_column].tolist()
        X = _fps_to_numpy(fps)

        self.logger.info(f"Running UMAP on {len(fps)} fingerprints ({X.shape[1]} bits)")
        embedding = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric="jaccard",
            random_state=self.random_state,
            low_memory=self.low_memory,
            verbose=False,
        ).fit_transform(X)

        df = df.copy()
        df[f"{self.out_column_prefix}_dim1"] = embedding[:, 0]
        df[f"{self.out_column_prefix}_dim2"] = embedding[:, 1]
        return df


@dataclass
class tSNEEmbedding(Link):
    """2D t-SNE embedding of molecular fingerprints

    Converts RDKit fingerprints to numpy arrays, computes a Tanimoto distance
    matrix using RDKit's fast BulkTanimotoSimilarity, then runs sklearn's
    t-SNE with metric='precomputed'.

    Requires scikit-learn.

    Parameters
    ----------
    in_column
        Column containing RDKit ExplicitBitVect fingerprints
    out_column_prefix
        Prefix for the output columns (produces {prefix}_dim1 and {prefix}_dim2)
    perplexity
        t-SNE perplexity parameter. Must be less than n_samples.
        Typical range 5-50. Larger values consider more global structure.
    random_state
        Random seed for reproducibility
    """

    _partitionable = Partitionable.NO

    in_column: InColumnName = "__MolFP__"
    out_column_prefix: str = "tSNE"
    perplexity: float = 30.0
    random_state: int = 42

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from sklearn.manifold import TSNE
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for tSNEEmbedding. "
                "Install with: pip install scikit-learn"
            ) from e

        fps = df[self.in_column].tolist()
        n = len(fps)

        # Compute Tanimoto distance matrix using RDKit (fast C++ implementation)
        self.logger.info(f"Computing {n}x{n} Tanimoto distance matrix")
        dist_matrix = np.zeros((n, n))
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            for j, s in enumerate(sims):
                dist_matrix[i, j] = 1.0 - s
                dist_matrix[j, i] = 1.0 - s

        perplexity = min(self.perplexity, n - 1)
        if perplexity != self.perplexity:
            self.logger.warning(
                f"Perplexity reduced from {self.perplexity} to {perplexity} "
                f"(must be less than n_samples={n})"
            )

        self.logger.info(f"Running t-SNE on {n} molecules (perplexity={perplexity})")
        embedding = TSNE(
            n_components=2,
            metric="precomputed",
            perplexity=perplexity,
            random_state=self.random_state,
            init="random",
        ).fit_transform(dist_matrix)

        df = df.copy()
        df[f"{self.out_column_prefix}_dim1"] = embedding[:, 0]
        df[f"{self.out_column_prefix}_dim2"] = embedding[:, 1]
        return df
