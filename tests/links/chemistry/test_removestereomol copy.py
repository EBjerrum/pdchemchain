import pandas as pd
from pdchemchain.links import RemoveStereoMol
from ...basetest import BaseErrorTest
import pytest
from rdkit import Chem


class TestRemoveStereoMol(BaseErrorTest):
    _Link = RemoveStereoMol
    _classparams = {"in_column": "ROMol", "out_column": "ROMol"}
    _alt_classparams = {"in_column": "ROMol2", "out_column": "ROMol2"}

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({"ROMol": [Chem.MolFromSmiles("C[C@](N)(O)Cl")]})

    def test_removestereomol(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "@" in Chem.MolToSmiles(sample_dataframe.ROMol.iloc[0])
        assert "@" not in Chem.MolToSmiles(df_o.ROMol.iloc[0])
