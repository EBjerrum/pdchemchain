from pdchemchain.links import HeteroAtomRatio
from ...basetest import BaseErrorTest
import pytest


class TestHeteroAtomRatio(BaseErrorTest):
    _Link = HeteroAtomRatio
    _classparams = {"in_column": "ROMol", "out_column": "HeteroAtomRatio"}
    _alt_classparams = {"in_column": "ROMol2", "out_column": "HeteroAtomRatio2"}

    def test_atomratio(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "HeteroAtomRatio" in df_o
        assert df_o.HeteroAtomRatio.values == pytest.approx([0.0, 1.0 / 2, 2.0 / 3])
