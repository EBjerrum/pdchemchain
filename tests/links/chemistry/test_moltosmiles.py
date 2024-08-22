from pdchemchain.links import MolToSmiles
from ...basetest import BaseErrorTest
import pytest
from rdkit import Chem


class TestMolToSmiles(BaseErrorTest):
    _Link = MolToSmiles
    _classparams = {"in_column": "ROMol", "out_column": "MolToSmiles"}
    _alt_classparams = {"in_column": "ROMol2", "out_column": "MolToSmiles2"}

    def test_moltosmiles(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "MolToSmiles" in df_o

        type_is_correct = []
        for smiles in df_o.MolToSmiles:
            type_is_correct.append(isinstance(smiles, str))

        assert all(type_is_correct)
