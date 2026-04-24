from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from pdchemchain.base import Link
from pdchemchain.typing import InColumnName, Partitionable




@dataclass
class ButinaClustering(Link):
    """Butina clustering using RDKit Tanimoto distances

    Uses RDKit's BulkTanimotoSimilarity for fast distance computation,
    then RDKit's Butina algorithm for Taylor-Butina clustering.
    Clusters smaller than min_cluster_size are reassigned to cluster -1 (noise).

    Parameters
    ----------
    in_column
        Column containing RDKit ExplicitBitVect fingerprints
    out_column
        Column to store cluster assignments
    distance_cutoff
        Tanimoto distance threshold for cluster membership.
        0.4 means molecules with Tanimoto similarity >= 0.6 cluster together.
    min_cluster_size
        Minimum number of members for a cluster. Smaller clusters
        are reassigned to -1 (noise/singletons).
    """

    _partitionable = Partitionable.NO

    in_column: InColumnName = "__MolFP__"
    out_column: str = "ButinaCluster"
    distance_cutoff: float = 0.4
    min_cluster_size: int = 5

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        fps = df[self.in_column].tolist()
        n = len(fps)

        # Condensed distance matrix using RDKit (Butina ordering: i from 1..n-1, j from 0..i-1)
        dists = []
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1.0 - s for s in sims])

        clusters = Butina.ClusterData(dists, n, self.distance_cutoff, isDistData=True)

        # Assign cluster labels, folding small clusters into -1
        labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        for members in clusters:
            if len(members) >= self.min_cluster_size:
                for idx in members:
                    labels[idx] = cluster_id
                cluster_id += 1

        n_clusters = cluster_id
        n_noise = (labels == -1).sum()
        self.logger.info(
            f"Butina clustering: {n_clusters} clusters, {n_noise} noise/singleton molecules"
        )

        df = df.copy()
        df[self.out_column] = labels
        return df
