import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.DataStructs import ExplicitBitVect

from pdchemchain.links.chemistry import MolToFingerprint
from pdchemchain.links.clustering import ButinaClustering
from pdchemchain.links.contrib.embeddings import UMAPEmbedding, tSNEEmbedding
from tests.basetest import BaseTest


# Larger fixture for clustering tests — need enough molecules for meaningful clusters
@pytest.fixture
def clustering_dataframe() -> pd.DataFrame:
    smiles_list = [
        # Cluster 1: benzene derivatives
        "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1",
        "c1ccc(F)cc1", "c1ccc(Cl)cc1",
        # Cluster 2: short aliphatics
        "CCCC", "CCCCC", "CCCCCC", "CCC(C)C", "CCCC(C)C",
        "CCCCCCC", "CCCCCCCC",
        # Cluster 3: amides/peptide-like
        "CC(=O)NC", "CC(=O)NCC", "CC(=O)NCCC", "CCC(=O)NC",
        "CC(=O)N(C)C", "CCC(=O)NCC",
    ]
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return pd.DataFrame({"Smiles": smiles_list, "ROMol": mols})


@pytest.fixture
def fp_dataframe(clustering_dataframe) -> pd.DataFrame:
    """Clustering dataframe with fingerprints pre-computed."""
    link = MolToFingerprint()
    return link(clustering_dataframe)


class TestMolToFingerprint(BaseTest):
    _Link = MolToFingerprint
    _classparams = {"radius": 2}
    _alt_classparams = {"radius": 1}

    def test_fingerprint_type(self, sample_dataframe):
        link = MolToFingerprint()
        result = link(sample_dataframe)
        assert "__MolFP__" in result.columns
        assert isinstance(result["__MolFP__"].iloc[0], ExplicitBitVect)

    def test_fingerprint_nbits(self, sample_dataframe):
        link = MolToFingerprint(n_bits=1024)
        result = link(sample_dataframe)
        assert result["__MolFP__"].iloc[0].GetNumBits() == 1024

    def test_unusual_radius_warns(self, caplog):
        """Unusual radius should warn but not raise."""
        import logging
        with caplog.at_level(logging.WARNING):
            import unittest.mock
            with unittest.mock.patch("time.sleep"):
                MolToFingerprint(radius=4)
        assert "Unusual Morgan radius=4" in caplog.text

    def test_different_radii_differ(self, sample_dataframe):
        r1 = MolToFingerprint(radius=1)(sample_dataframe)["__MolFP__"].iloc[0]
        r2 = MolToFingerprint(radius=2)(sample_dataframe)["__MolFP__"].iloc[0]
        # Radius 2 should have at least as many bits on as radius 1
        assert r2.GetNumOnBits() >= r1.GetNumOnBits()


class TestButinaClustering(BaseTest):
    _Link = ButinaClustering
    _classparams = {"distance_cutoff": 0.4}
    _alt_classparams = {"distance_cutoff": 0.6}

    @pytest.fixture
    def sample_dataframe(self, fp_dataframe):
        return fp_dataframe

    def test_cluster_column_created(self, fp_dataframe):
        link = ButinaClustering()
        result = link(fp_dataframe)
        assert "ButinaCluster" in result.columns
        assert result["ButinaCluster"].dtype == int

    def test_min_cluster_size(self, fp_dataframe):
        link = ButinaClustering(min_cluster_size=5)
        result = link(fp_dataframe)
        # All clusters should have >= 5 members, or be -1
        for cluster_id in result["ButinaCluster"].unique():
            if cluster_id >= 0:
                assert (result["ButinaCluster"] == cluster_id).sum() >= 5

    def test_noise_label(self, fp_dataframe):
        # With very tight cutoff, most things become noise
        link = ButinaClustering(distance_cutoff=0.05, min_cluster_size=5)
        result = link(fp_dataframe)
        assert -1 in result["ButinaCluster"].values

    def test_custom_out_column(self, fp_dataframe):
        link = ButinaClustering(out_column="my_cluster")
        result = link(fp_dataframe)
        assert "my_cluster" in result.columns


class TestUMAPEmbedding(BaseTest):
    _Link = UMAPEmbedding
    _classparams = {"n_neighbors": 5}
    _alt_classparams = {"n_neighbors": 10}

    @pytest.fixture
    def sample_dataframe(self, fp_dataframe):
        return fp_dataframe

    def test_embedding_columns(self, fp_dataframe):
        link = UMAPEmbedding(n_neighbors=5)
        result = link(fp_dataframe)
        assert "Umap_dim1" in result.columns
        assert "Umap_dim2" in result.columns
        assert len(result) == len(fp_dataframe)

    def test_custom_prefix(self, fp_dataframe):
        link = UMAPEmbedding(out_column_prefix="MyEmb", n_neighbors=5)
        result = link(fp_dataframe)
        assert "MyEmb_dim1" in result.columns
        assert "MyEmb_dim2" in result.columns

    def test_reproducibility(self, fp_dataframe):
        link = UMAPEmbedding(n_neighbors=5, random_state=42)
        r1 = link(fp_dataframe)
        r2 = link(fp_dataframe)
        np.testing.assert_array_equal(r1["Umap_dim1"].values, r2["Umap_dim1"].values)


class TesttSNEEmbedding(BaseTest):
    _Link = tSNEEmbedding
    _classparams = {"perplexity": 5.0}
    _alt_classparams = {"perplexity": 10.0}

    @pytest.fixture
    def sample_dataframe(self, fp_dataframe):
        return fp_dataframe

    def test_embedding_columns(self, fp_dataframe):
        link = tSNEEmbedding(perplexity=5.0)
        result = link(fp_dataframe)
        assert "tSNE_dim1" in result.columns
        assert "tSNE_dim2" in result.columns
        assert len(result) == len(fp_dataframe)

    def test_custom_prefix(self, fp_dataframe):
        link = tSNEEmbedding(out_column_prefix="MyTSNE", perplexity=5.0)
        result = link(fp_dataframe)
        assert "MyTSNE_dim1" in result.columns
        assert "MyTSNE_dim2" in result.columns

    def test_perplexity_auto_clamp(self, fp_dataframe):
        """Perplexity larger than n_samples should be clamped, not crash."""
        link = tSNEEmbedding(perplexity=100.0)
        result = link(fp_dataframe)
        assert "tSNE_dim1" in result.columns
