import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.DataStructs import ExplicitBitVect

from pdchemchain.links.chemistry import MolToFingerprint
from pdchemchain.links.clustering import ButinaClustering, CommonBitFilter
from pdchemchain.links.dataframe import GroupPick
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


class TestCommonBitFilter(BaseTest):
    _Link = CommonBitFilter
    _classparams = {"threshold": 1.0}
    _alt_classparams = {"threshold": 0.5}

    @pytest.fixture
    def sample_dataframe(self, fp_dataframe):
        return fp_dataframe

    def test_fp_size_preserved(self, fp_dataframe):
        original_nbits = fp_dataframe["__MolFP__"].iloc[0].GetNumBits()
        result = CommonBitFilter()(fp_dataframe)
        assert result["__MolFP__"].iloc[0].GetNumBits() == original_nbits

    def test_bits_removed_at_100_percent(self, fp_dataframe):
        """Bits set in ALL molecules should be zeroed at threshold=1.0."""
        fps = fp_dataframe["__MolFP__"].tolist()
        # Find bits set in all molecules via AND
        common = ExplicitBitVect(fps[0].GetNumBits())
        common |= fps[0]
        for fp in fps[1:]:
            common &= fp
        common_bits = set(common.GetOnBits())

        result = CommonBitFilter(threshold=1.0)(fp_dataframe)
        for fp in result["__MolFP__"]:
            # None of the universally-common bits should remain
            assert not common_bits.intersection(fp.GetOnBits())

    def test_threshold_partial(self, fp_dataframe):
        """Lower threshold should remove more bits than 1.0."""
        r100 = CommonBitFilter(threshold=1.0)(fp_dataframe)
        r50 = CommonBitFilter(threshold=0.5)(fp_dataframe)
        mean_on_100 = np.mean([fp.GetNumOnBits() for fp in r100["__MolFP__"]])
        mean_on_50 = np.mean([fp.GetNumOnBits() for fp in r50["__MolFP__"]])
        assert mean_on_50 <= mean_on_100

    def test_no_common_bits(self):
        """Diverse FPs with no universally-set bits should be unchanged."""
        n_bits = 64
        fps = []
        for i in range(5):
            fp = ExplicitBitVect(n_bits)
            fp.SetBit(i)  # each FP has a unique bit
            fps.append(fp)
        df = pd.DataFrame({"__MolFP__": fps})
        result = CommonBitFilter(threshold=1.0)(df)
        for orig, filt in zip(fps, result["__MolFP__"]):
            assert tuple(orig.GetOnBits()) == tuple(filt.GetOnBits())

    def test_custom_out_column(self, fp_dataframe):
        result = CommonBitFilter(out_column="__MolFP_filtered__")(fp_dataframe)
        assert "__MolFP_filtered__" in result.columns
        assert "__MolFP__" in result.columns  # original preserved


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


# --- GroupPick tests ---


class TestGroupPickTag:
    """Test tag mode: marks best row per group, keeps all rows."""

    @pytest.fixture
    def clustered_df(self):
        return pd.DataFrame({
            "ButinaCluster": [0, 0, 0, 1, 1, -1],
            "docking_score": [-8.5, -6.0, -7.2, -9.0, -5.5, -4.0],
            "Smiles": ["A", "B", "C", "D", "E", "F"],
        })

    def test_tags_best_min(self, clustered_df):
        link = GroupPick(mode="min")
        result = link(clustered_df)
        assert len(result) == 6  # all rows preserved
        assert "picked" in result.columns
        # Cluster 0: best is -8.5 (row 0)
        assert result.loc[0, "picked"] == True
        assert result.loc[1, "picked"] == False
        # Cluster 1: best is -9.0 (row 3)
        assert result.loc[3, "picked"] == True

    def test_tags_best_max(self, clustered_df):
        link = GroupPick(mode="max")
        result = link(clustered_df)
        # Cluster 0: max is -6.0 (row 1)
        assert result.loc[1, "picked"] == True
        assert result.loc[0, "picked"] == False

    def test_skips_noise_by_default(self, clustered_df):
        link = GroupPick(mode="min")
        result = link(clustered_df)
        # Noise cluster (-1) should not be picked
        assert result.loc[5, "picked"] == False

    def test_pick_noise_when_enabled(self, clustered_df):
        link = GroupPick(mode="min", pick_noise=True)
        result = link(clustered_df)
        assert result.loc[5, "picked"] == True

    def test_custom_out_column(self, clustered_df):
        link = GroupPick(out_column="selected")
        result = link(clustered_df)
        assert "selected" in result.columns
        assert "picked" not in result.columns


class TestGroupPickTagFilter:
    """Test tag mode with filter_column."""

    def test_filter_restricts_candidates(self):
        df = pd.DataFrame({
            "ButinaCluster": [0, 0, 0],
            "docking_score": [-9.0, -7.0, -5.0],
            "eligible": [False, True, True],
        })
        link = GroupPick(filter_column="eligible")
        result = link(df)
        # Best eligible is -7.0 (row 1), not -9.0 (row 0, ineligible)
        assert result.loc[0, "picked"] == False
        assert result.loc[1, "picked"] == True
        assert result.loc[2, "picked"] == False

    def test_no_eligible_rows(self):
        df = pd.DataFrame({
            "ButinaCluster": [0, 0],
            "docking_score": [-9.0, -7.0],
            "eligible": [False, False],
        })
        link = GroupPick(filter_column="eligible")
        result = link(df)
        assert not result["picked"].any()


class TestGroupPickCollapse:
    """Test collapse mode: drops non-best rows."""

    def test_collapses_to_one_per_group(self):
        df = pd.DataFrame({
            "__id__": [0, 0, 1, 1],
            "docking_score": [-8.5, -6.0, -7.0, -9.0],
            "variant": ["a", "b", "c", "d"],
        })
        link = GroupPick(group_by="__id__", action="collapse")
        result = link(df)
        assert len(result) == 2
        row0 = result[result["__id__"] == 0].iloc[0]
        assert row0["docking_score"] == -8.5
        row1 = result[result["__id__"] == 1].iloc[0]
        assert row1["docking_score"] == -9.0

    def test_error_only_group_preserved(self):
        df = pd.DataFrame({
            "__id__": [0, 0, 1],
            "docking_score": [-8.5, -6.0, np.nan],
            "__error__": [None, None, "LigPrep: failed"],
        })
        link = GroupPick(group_by="__id__", action="collapse")
        result = link(df)
        assert len(result) == 2
        error_row = result[result["__id__"] == 1].iloc[0]
        assert error_row["__error__"] == "LigPrep: failed"

    def test_drop_group_columns(self):
        df = pd.DataFrame({
            "__id__": [0, 1],
            "__enum_id__": [0, 1],
            "docking_score": [-8.5, -6.0],
        })
        link = GroupPick(group_by="__id__", action="collapse", drop_group_columns=True)
        result = link(df)
        assert "__id__" not in result.columns
        assert "__enum_id__" not in result.columns


class TestGroupPickNaN:
    """Test handling of all-NaN score groups."""

    def test_collapse_all_nan_group_gets_error(self):
        """Groups where all scores are NaN should get first row with __error__."""
        df = pd.DataFrame({
            "__id__": [0, 0, 1, 1],
            "docking_score": [np.nan, np.nan, -8.5, -6.0],
            "Smiles": ["A", "B", "C", "D"],
        })
        link = GroupPick(group_by="__id__", action="collapse")
        result = link(df)
        assert len(result) == 2
        # Group 0: all NaN, should have error
        row0 = result[result["__id__"] == 0].iloc[0]
        assert "no valid" in row0["__error__"]
        assert row0["Smiles"] == "A"  # first row selected
        # Group 1: valid, best score
        row1 = result[result["__id__"] == 1].iloc[0]
        assert row1["docking_score"] == -8.5

    def test_collapse_all_groups_nan(self):
        """All groups NaN — every group gets an error row."""
        df = pd.DataFrame({
            "__id__": [0, 1],
            "docking_score": [np.nan, np.nan],
        })
        link = GroupPick(group_by="__id__", action="collapse")
        result = link(df)
        assert len(result) == 2
        assert result["__error__"].notna().all()

    def test_tag_nan_group_not_picked(self):
        """In tag mode, all-NaN groups get no pick."""
        df = pd.DataFrame({
            "ButinaCluster": [0, 0, 1],
            "docking_score": [np.nan, np.nan, -8.0],
        })
        link = GroupPick()
        result = link(df)
        assert result.loc[0, "picked"] == False
        assert result.loc[1, "picked"] == False
        assert result.loc[2, "picked"] == True


class TestGroupPickValidation:
    """Test parameter validation."""

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            GroupPick(mode="average")

    def test_invalid_action(self):
        with pytest.raises(ValueError, match="action must be"):
            GroupPick(action="filter")
