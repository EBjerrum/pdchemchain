from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from rdkit.ML.Cluster import Butina

from pdchemchain.base import Link
from pdchemchain.typing import InColumnName, Partitionable




@dataclass
class CommonBitFilter(Link):
    """Remove fingerprint bits that are set in a high fraction of molecules.

    Core-growing de novo sets share scaffold bits, compressing Tanimoto
    distances. Zeroing universally-set bits before clustering restores
    meaningful distance variation between side chains.

    The fingerprint size is preserved (bits are zeroed, not removed),
    so downstream links expecting a specific bit vector length still work.

    Parameters
    ----------
    in_column
        Column containing RDKit ExplicitBitVect fingerprints.
    out_column
        Column to store filtered fingerprints. Defaults to overwriting
        the input column so MolToFingerprint + CommonBitFilter + ButinaClustering
        works without reconfiguring column names.
    threshold
        Remove bits set in >= this fraction of molecules.
        1.0 (default) removes only bits present in ALL molecules.

    Examples
    --------
    >>> chain = MolToFingerprint() + CommonBitFilter() + ButinaClustering()
    """

    _partitionable = Partitionable.NO

    in_column: InColumnName = "__MolFP__"
    out_column: str = "__MolFP__"
    threshold: float = 1.0

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        fps = df[self.in_column].tolist()
        n = len(fps)
        n_bits = fps[0].GetNumBits()

        # Count bit frequencies
        counts = np.zeros(n_bits, dtype=int)
        for fp in fps:
            for b in fp.GetOnBits():
                counts[b] += 1
        freq = counts / n

        # Identify bits to remove
        bits_to_remove = set(np.where(freq >= self.threshold)[0])
        n_removed = len(bits_to_remove)

        if n_removed == 0:
            self.logger.info("No bits exceed threshold — fingerprints unchanged")
            df = df.copy()
            if self.out_column != self.in_column:
                df[self.out_column] = df[self.in_column]
            return df

        # Build filtered fingerprints
        filtered = []
        for fp in fps:
            new_fp = ExplicitBitVect(n_bits)
            for b in fp.GetOnBits():
                if b not in bits_to_remove:
                    new_fp.SetBit(b)
            filtered.append(new_fp)

        mean_on = np.mean([fp.GetNumOnBits() for fp in filtered])
        self.logger.info(
            f"Removed {n_removed} bits (>= {self.threshold:.0%} frequency), "
            f"mean on-bits: {mean_on:.0f}"
        )

        df = df.copy()
        df[self.out_column] = filtered
        return df


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
