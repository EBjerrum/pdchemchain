import pandas as pd
from pdchemchain.links import RemoveStereoSmiles
from ...basetest import BaseErrorTest
import pytest
from rdkit import Chem


class TestRemoveStereoSmiles(BaseErrorTest):
    _Link = RemoveStereoSmiles
    _classparams = {"in_column": "Smiles", "out_column": "SmilesNoStereo"}
    _alt_classparams = {"in_column": "Smiles2", "out_column": "Smiles2"}
    _expected_errors = [
        False,
        False,
        False,
        False,
        True,
        True,
    ]  # The invalid text string 'ImNotASmiles' can still be stripped of @

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({"Smiles": ["C[C@](N)(O)Cl"]})

    def test_removestereosmiles(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "@" in sample_dataframe.Smiles.iloc[0]
        assert "@" not in df_o.SmilesNoStereo.iloc[0]
