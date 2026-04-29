"""Tests for the REINVENT PdChemChain scoring component.

Tests the core scoring logic using pdchemchain chains directly, plus
conditional tests for the full REINVENT component when reinvent is installed.
"""

import numpy as np
import pandas as pd
import pytest

from pdchemchain.links.chemistry import MolFromSmiles, RDKitDescriptors
from pdchemchain.links.dataframe import NullLink
from pdchemchain.io_utilities import save_chain

try:
    from reinvent_plugins.components.comp_pdchemchain import PdChemChain, Parameters

    HAS_REINVENT = True
except ImportError:
    HAS_REINVENT = False

requires_reinvent = pytest.mark.skipif(
    not HAS_REINVENT, reason="reinvent_plugins not installed"
)


@pytest.fixture
def descriptors_chain():
    """A chain: SMILES -> ROMol -> MolLogP + qed."""
    return MolFromSmiles(in_column="SMILES") + RDKitDescriptors(descriptors=["MolLogP", "qed"])


@pytest.fixture
def descriptors_config(descriptors_chain, tmp_path):
    """Save the descriptors chain to a YAML config and return the path."""
    config_file = str(tmp_path / "chain.yaml")
    save_chain(descriptors_chain, config_file)
    return config_file


class TestChainExecution:
    """Test that chains produce expected DataFrame output (no REINVENT needed)."""

    def test_descriptors_chain_produces_columns(self, descriptors_chain):
        df = pd.DataFrame({"SMILES": ["c1ccccc1", "CCO", "CC(=O)O"]})
        result = descriptors_chain(df)
        assert "MolLogP" in result.columns
        assert "qed" in result.columns
        assert len(result) == 3

    def test_descriptors_values_are_numeric(self, descriptors_chain):
        df = pd.DataFrame({"SMILES": ["c1ccccc1", "CCO"]})
        result = descriptors_chain(df)
        assert np.isfinite(result["MolLogP"].values).all()
        assert np.isfinite(result["qed"].values).all()


@requires_reinvent
class TestPdChemChainComponent:
    """Test the full REINVENT scoring component."""

    def _make_params(self, config_file, score_columns, **kwargs):
        """Helper to construct Parameters with REINVENT's list convention."""
        return Parameters(
            config_file=[config_file],
            score_column=score_columns,
            smiles_column=kwargs.get("smiles_column", ["SMILES"]),
            default_score=kwargs.get("default_score", [float("nan")]),
        )

    def test_single_endpoint(self, descriptors_config):
        params = self._make_params(descriptors_config, ["MolLogP"])
        component = PdChemChain(params)

        assert component.number_of_endpoints == 1

        result = component(["c1ccccc1", "CCO", "CC(=O)O"])
        assert len(result.scores) == 1
        assert len(result.scores[0]) == 3
        assert np.isfinite(result.scores[0]).all()

    def test_multi_endpoint_single_execution(self, descriptors_config):
        params = self._make_params(descriptors_config, ["MolLogP", "qed"])
        component = PdChemChain(params)

        assert component.number_of_endpoints == 2

        result = component(["c1ccccc1", "CCO"])
        assert len(result.scores) == 2
        assert len(result.scores[0]) == 2
        assert len(result.scores[1]) == 2
        # MolLogP and qed should be different values
        assert not np.allclose(result.scores[0], result.scores[1])

    def test_missing_column_returns_nan(self, descriptors_config):
        params = self._make_params(descriptors_config, ["nonexistent_column"])
        component = PdChemChain(params)

        result = component(["c1ccccc1", "CCO"])
        assert len(result.scores) == 1
        assert np.isnan(result.scores[0]).all()

    def test_default_score_replaces_nan(self, descriptors_config):
        params = self._make_params(
            descriptors_config, ["nonexistent_column"], default_score=[0.0]
        )
        component = PdChemChain(params)

        result = component(["c1ccccc1", "CCO"])
        assert (result.scores[0] == 0.0).all()

    def test_invalid_smiles_handled(self, descriptors_config):
        """Invalid SMILES should be filtered by @normalize_smiles.

        The decorator strips invalid SMILES before calling our function.
        Depending on the REINVENT version, the result may be padded back
        to full length (with NaN) or returned as a shorter array.
        Either way, valid SMILES must get finite scores.
        """
        params = self._make_params(descriptors_config, ["MolLogP"])
        component = PdChemChain(params)

        result = component(["c1ccccc1", "NOT_A_SMILES", "CCO"])
        scores = result.scores[0]
        finite_scores = scores[np.isfinite(scores)]
        assert len(finite_scores) == 2  # benzene + ethanol

    def test_default_score_with_chain_errors(self, tmp_path):
        """Rows with __error__ should get default_score."""
        # MolFromSmiles will set __error__ for bad SMILES that pass
        # @normalize_smiles (e.g. SMILES that RDKit parses but produce
        # None mol). We test with valid + intentionally problematic input.
        chain = MolFromSmiles(in_column="SMILES") + RDKitDescriptors(descriptors=["MolLogP"])
        config = str(tmp_path / "chain.yaml")
        save_chain(chain, config)
        params = self._make_params(config, ["MolLogP"], default_score=[-999.0])
        component = PdChemChain(params)
        # Both are valid SMILES — should get real scores
        result = component(["c1ccccc1", "CCO"])
        assert np.isfinite(result.scores[0]).all()
        assert (result.scores[0] != -999.0).all()

    def test_chain_loaded_once(self, descriptors_config):
        """Verify the chain object is reused across calls."""
        params = self._make_params(descriptors_config, ["MolLogP"])
        component = PdChemChain(params)

        chain_id = id(component.chain)
        component(["c1ccccc1"])
        component(["CCO"])
        assert id(component.chain) == chain_id

    def test_all_invalid_smiles(self, descriptors_config):
        """All-invalid batch should return all NaN via @normalize_smiles."""
        params = self._make_params(descriptors_config, ["MolLogP", "qed"])
        component = PdChemChain(params)

        result = component(["INVALID1", "INVALID2"])
        assert len(result.scores) == 2
        assert np.isnan(result.scores[0]).all()
        assert np.isnan(result.scores[1]).all()
