from pdchemchain.links import RemoveAtomMapping
from ...basetest import BaseErrorTest
import pytest
import pandas as pd
from rdkit import Chem


class TestMolToInChI(BaseErrorTest):
    _Link = RemoveAtomMapping
    _classparams = {
        "in_column": "ROMol",
        "out_column": "ROMol",
    }

    _alt_classparams = {
        "in_column": "ROMol2",
        "out_column": "ROMol2",
    }

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({"ROMol": [Chem.MolFromSmiles("[CH4:0]")]})

    def test_removeatommapping(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "molAtomMapNumber" in sample_dataframe.ROMol.iloc[0].GetAtomWithIdx(0).GetPropsAsDict()
        assert "molAtomMapNumber" not in df_o.ROMol.iloc[0].GetAtomWithIdx(0).GetPropsAsDict()
