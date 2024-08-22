from pdchemchain.links import MolFromSmiles
from ...basetest import BaseErrorTest
import pytest
from rdkit import Chem


class TestMolFromSmiles(BaseErrorTest):
    _Link = MolFromSmiles
    _classparams = {"in_column": "Smiles", "out_column": "MolFromSmiles"}
    _alt_classparams = {"in_column": "Smiles2", "out_column": "MolFromSmiles2"}

    def test_molfromsmiles(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "MolFromSmiles" in df_o

        type_is_correct = []
        for mol in df_o.MolFromSmiles:
            type_is_correct.append(isinstance(mol, Chem.Mol))

        assert all(type_is_correct)
