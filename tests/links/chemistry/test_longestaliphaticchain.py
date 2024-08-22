from pdchemchain.links import LongestAliphaticChain
from ...basetest import BaseErrorTest
import pytest


class TestLongestAliphaticChain(BaseErrorTest):
    _Link = LongestAliphaticChain
    _classparams = {
        "in_column": "ROMol",
        "out_column": "LongestAliphaticChain",
        "max_chain_length": 11,
    }
    _alt_classparams = {
        "in_column": "ROMol2",
        "out_column": "LongestAliphaticChain2",
        "max_chain_length": 10,
    }

    def test_atomratio(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "LongestAliphaticChain" in df_o
        # assert df_o.LongestAliphaticChain.values == pytest.approx([0.,1./2, 2./3])
