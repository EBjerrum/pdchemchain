from pdchemchain.links.contrib.reinvent import REInventTokenizer, TokenReturnType
from ...basetest import BaseErrorTest
import pytest
from rdkit import Chem


class TestREInventTokenizer(BaseErrorTest):
    _Link = REInventTokenizer
    _classparams = {"in_column": "Smiles", "out_column": "Tokens", "return_type": "count"}
    _alt_classparams = {"in_column": "Smiles2", "out_column": "Tokens2", "return_type": "list"}
    _expected_errors = [
        False,
        False,
        False,
        False,
        True,
        True,
    ]  # The RDKit imparsable SMILES string in iloc[3] can still be tokenized

    def test_reinvent_tokenizer_count(self, sample_dataframe):
        """Test that return_type='count' returns integer token counts"""
        link = REInventTokenizer(
            in_column="Smiles", out_column="Tokens", return_type="count"
        )
        df_o = link(sample_dataframe)
        assert "Tokens" in df_o

        type_is_correct = []
        for token_count in df_o.Tokens:
            type_is_correct.append(isinstance(token_count, int))

        assert all(type_is_correct)

    def test_reinvent_tokenizer_list(self, sample_dataframe):
        """Test that return_type='list' returns list of tokens"""
        link = REInventTokenizer(
            in_column="Smiles", out_column="Tokens", return_type="list"
        )
        df_o = link(sample_dataframe)
        assert "Tokens" in df_o

        type_is_correct = []
        for tokens in df_o.Tokens:
            type_is_correct.append(isinstance(tokens, list))

        assert all(type_is_correct)

    def test_reinvent_tokenizer_set(self, sample_dataframe):
        """Test that return_type='set' returns set of unique tokens"""
        link = REInventTokenizer(
            in_column="Smiles", out_column="Tokens", return_type="set"
        )
        df_o = link(sample_dataframe)
        assert "Tokens" in df_o

        type_is_correct = []
        for tokens in df_o.Tokens:
            type_is_correct.append(isinstance(tokens, set))

        assert all(type_is_correct)

    def test_invalid_return_type_on_init(self):
        """Test that invalid return_type raises ValueError on initialization"""
        with pytest.raises(ValueError, match="Invalid return_type"):
            REInventTokenizer(
                in_column="Smiles", out_column="Tokens", return_type="invalid"
            )

    def test_invalid_return_type_on_setattr(self):
        """Test that invalid return_type raises ValueError when set after instantiation"""
        link = REInventTokenizer(
            in_column="Smiles", out_column="Tokens", return_type="count"
        )
        with pytest.raises(ValueError, match="Invalid return_type"):
            link.return_type = "invalid"

    def test_dynamic_return_type_change(self, sample_dataframe):
        """Test that return_type can be changed dynamically after instantiation"""
        link = REInventTokenizer(
            in_column="Smiles", out_column="Tokens", return_type="count"
        )

        # First run with count
        df_count = link(sample_dataframe)
        assert all(isinstance(x, int) for x in df_count.Tokens)

        # Change to list and run again
        link.return_type = "list"
        df_list = link(sample_dataframe)
        assert all(isinstance(x, list) for x in df_list.Tokens)

        # Change to set and run again
        link.return_type = "set"
        df_set = link(sample_dataframe)
        assert all(isinstance(x, set) for x in df_set.Tokens)

    def test_token_values_consistency(self, sample_dataframe):
        """Test that token counts, lists, and sets are consistent"""
        link_count = REInventTokenizer(
            in_column="Smiles", out_column="Tokens", return_type="count"
        )
        link_list = REInventTokenizer(
            in_column="Smiles", out_column="Tokens", return_type="list"
        )
        link_set = REInventTokenizer(
            in_column="Smiles", out_column="Tokens", return_type="set"
        )

        df_count = link_count(sample_dataframe)
        df_list = link_list(sample_dataframe)
        df_set = link_set(sample_dataframe)

        for i in range(len(sample_dataframe)):
            # Count should equal length of list
            assert df_count.iloc[i].Tokens == len(df_list.iloc[i].Tokens)
            # Set size should be <= list length (due to duplicates)
            assert len(df_set.iloc[i].Tokens) <= len(df_list.iloc[i].Tokens)
            # All set elements should be in list
            assert df_set.iloc[i].Tokens.issubset(set(df_list.iloc[i].Tokens))
