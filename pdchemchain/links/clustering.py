import heapq
from dataclasses import dataclass
from typing import List, Sequence

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


def _neighbour_lists(fps: Sequence, cutoff: float):
    """Sparse neighbour lists (indices + distances) for pairs within ``cutoff``.

    Row-wise BulkTanimotoSimilarity: the inner loop is RDKit C++, and only pairs below
    the threshold are retained, so memory is proportional to the number of edges rather
    than to n^2. A dense n x n float32 matrix is 262 MB at n=8091, and wide concatenated
    fingerprints make a BLAS approach expensive without block splitting.
    """
    n = len(fps)
    fps = list(fps)
    idx: List[np.ndarray] = [None] * n  # type: ignore[list-item]
    dist: List[np.ndarray] = [None] * n  # type: ignore[list-item]
    for i in range(n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps), dtype=np.float32)
        d = 1.0 - sims
        d[i] = np.inf  # never a neighbour of itself
        hit = np.flatnonzero(d <= cutoff)
        order = np.argsort(d[hit], kind="stable")
        idx[i] = hit[order].astype(np.int32)
        dist[i] = d[hit][order]
    return idx, dist


def capped_butina(
    fps: Sequence,
    cutoff: float = 0.40,
    max_size: int = 25,
    min_cluster_size: int = 3,
    seed_by: str = "tightest",
) -> np.ndarray:
    """Size-capped Butina cluster labels; -1 for clusters below ``min_cluster_size``.

    Exposed as a function as well as a Link so it can be used on a bare fingerprint list.
    See :class:`CappedButinaClustering` for the rationale and parameters.
    """
    n = len(fps)
    idx, dist = _neighbour_lists(fps, cutoff)

    assigned = np.zeros(n, dtype=bool)
    labels = np.full(n, -1, dtype=np.int32)

    # Seed keys, maintained lazily. `nn` is the distance to the nearest still-unassigned
    # neighbour, `cnt` the number of unassigned neighbours. Both go stale as molecules are
    # assigned, so they are recomputed on pop and the candidate re-queued if its key moved
    # -- the standard lazy-heap trick, which avoids rescanning the whole pool each
    # iteration (that would be O(n^3 / max_size) overall).
    def live(i):
        """(nearest unassigned distance, count of unassigned neighbours, those indices)."""
        m = ~assigned[idx[i]]
        if not m.any():
            return np.inf, 0, np.empty(0, dtype=np.int32)
        return float(dist[i][m][0]), int(m.sum()), idx[i][m]

    heap = []
    for i in range(n):
        nn, cnt, _ = live(i)
        heapq.heappush(heap, (nn if seed_by == "tightest" else -cnt, i))

    cid = 0
    while heap:
        key, i = heapq.heappop(heap)
        if assigned[i]:
            continue
        nn, cnt, cand = live(i)
        cur = nn if seed_by == "tightest" else -cnt
        if cand.size and cur > key + 1e-12:
            heapq.heappush(heap, (cur, i))  # key was stale, re-queue
            continue

        take = cand[: max(0, max_size - 1)]
        members = np.concatenate(([i], take)) if take.size else np.array([i])
        labels[members] = cid
        assigned[members] = True
        cid += 1

    # Demote clusters below the minimum size to unassigned, then renumber by descending
    # size so ids follow RDKit Butina's convention.
    keep, sizes = np.unique(labels[labels >= 0], return_counts=True)
    big = keep[sizes >= min_cluster_size]
    order = big[np.argsort(-sizes[np.isin(keep, big)], kind="stable")]
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap.get(lab, -1) for lab in labels], dtype=np.int32)


@dataclass
class CappedButinaClustering(Link):
    """Butina clustering with an upper bound on cluster size.

    Standard Butina picks the molecule with the most neighbours within the threshold as a
    centroid and absorbs ALL of them into one cluster. On a densely sampled de novo set
    that produces a mega-cluster: on a TYK2 pool one centroid absorbed 638 molecules
    (7.9% of the set) at cutoff 0.40, and 1644 (20.3%) at cutoff 0.45. Such a cluster
    spans hundreds of distinct generic scaffolds and is useless both for stratified
    picking (one quota for a fifth of the set) and for visual inspection.

    The capped variant::

        while molecules remain unassigned:
            seed    = the unassigned molecule whose NEAREST unassigned neighbour is
                      closest (seed_by="tightest"), or the one with the most
                      unassigned neighbours (seed_by="most_neighbours", Butina's own
                      criterion)
            cluster = seed + up to `max_size - 1` of its NEAREST unassigned neighbours
                      within `distance_cutoff`
            remove that cluster from the pool

    Below the cap this reduces to standard Butina behaviour, since "the nearest up to X
    within cutoff" is the whole neighbourhood when the neighbourhood is smaller than X.
    Seeding on the tightest pair grows clusters outward from the densest core, which
    gives tighter, more interpretable series than seeding on neighbour count.

    Parameters
    ----------
    in_column
        Column containing RDKit ExplicitBitVect fingerprints.
    out_column
        Column to store cluster assignments.
    distance_cutoff
        Tanimoto distance threshold for cluster membership. 0.4 means molecules with
        Tanimoto similarity >= 0.6 may cluster together.
    max_size
        Maximum members per cluster. This is the whole point of the link: it forces
        dense regions to split into several clusters instead of one mega-cluster.
    min_cluster_size
        Clusters smaller than this are reassigned to -1 (noise/singletons).
    seed_by
        ``"tightest"`` (default) seeds on the closest remaining pair;
        ``"most_neighbours"`` reproduces Butina's own seeding criterion.

    Examples
    --------
    >>> # Cluster for diversity picking, no cluster larger than 25
    >>> CappedButinaClustering(in_column="__MolFP__", distance_cutoff=0.4, max_size=25)
    """

    _partitionable = Partitionable.NO

    in_column: InColumnName = "__MolFP__"
    out_column: str = "ButinaCluster"
    distance_cutoff: float = 0.4
    max_size: int = 25
    min_cluster_size: int = 3
    seed_by: str = "tightest"

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.seed_by not in ("tightest", "most_neighbours"):
            raise ValueError(
                f"seed_by must be 'tightest' or 'most_neighbours', got {self.seed_by!r}"
            )
        labels = capped_butina(
            df[self.in_column].tolist(),
            cutoff=self.distance_cutoff,
            max_size=self.max_size,
            min_cluster_size=self.min_cluster_size,
            seed_by=self.seed_by,
        )
        n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
        sizes = np.unique(labels[labels >= 0], return_counts=True)[1]
        self.logger.info(
            f"Capped Butina clustering (cap {self.max_size}): {n_clusters} clusters, "
            f"{int((labels == -1).sum())} noise/singleton molecules, "
            f"median size {np.median(sizes) if sizes.size else 0:.0f}, "
            f"max size {sizes.max() if sizes.size else 0}"
        )

        df = df.copy()
        df[self.out_column] = labels
        return df
