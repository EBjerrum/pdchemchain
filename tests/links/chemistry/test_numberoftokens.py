from pdchemchain.links import NumberOfTokens
from ...basetest import BaseErrorTest
import pytest
from rdkit import Chem


class TestNumberOfTokens(BaseErrorTest):
    _Link = NumberOfTokens
    _classparams = {"in_column": "Smiles", "out_column": "NumberOfTokens"}
    _alt_classparams = {"in_column": "Smiles2", "out_column": "NumberOfTokens2"}
    _expected_errors = [
        False,
        False,
        False,
        False,
        True,
        True,
    ]  # The RDKit imparsable SMILES string  in iloc[3] can still be tokenized

    def test_numberoftokens(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "NumberOfTokens" in df_o

        type_is_correct = []
        for number in df_o.NumberOfTokens:
            # raise ValueError (f"{smiles=}")
            type_is_correct.append(isinstance(number, int))

        assert all(type_is_correct)
