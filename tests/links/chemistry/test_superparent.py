import pandas as pd
from pdchemchain.links import SuperParent
from ...basetest import BaseErrorTest
import pytest
from rdkit import Chem


class TestSuperParent(BaseErrorTest):
    _Link = SuperParent
    _classparams = {"in_column": "ROMol", "maxTautomers": 0, "out_column": "ROMol2"}
    _alt_classparams = {
        "in_column": "ROMol2",
        "maxTautomers": 100,
        "out_column": "ROMol3",
    }
    #    _expected_errors = [False,False,False,False,True,True] #The invalid text string 'ImNotASmiles' can still be stripped of @

    def test_removestereosmiles(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "ROMol2" in df_o
