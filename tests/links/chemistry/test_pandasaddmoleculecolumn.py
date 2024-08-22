from pdchemchain.links import PandasAddMoleculeColumn
from ...basetest import BaseTest
import pytest
from rdkit import Chem


class TestPandasAddMoleculeColumn(BaseTest):
    _Link = PandasAddMoleculeColumn
    _classparams = {
        "includeFingerprints": False,
        "molCol": "PandasAddMoleculeColumn",
        "smilesCol": "Smiles",
    }
    _alt_classparams = {
        "includeFingerprints": True,
        "molCol": "PandasAddMoleculeColumn2",
        "smilesCol": "Smiles2",
    }

    def test_pandastools_add_column(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "PandasAddMoleculeColumn" in df_o

        type_is_correct = []
        for mol in df_o.PandasAddMoleculeColumn:
            type_is_correct.append(isinstance(mol, Chem.Mol))

        assert all(type_is_correct)
