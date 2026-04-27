"""Tests for AggregateByScore link."""

import numpy as np
import pandas as pd
import pytest

from pdchemchain.links.schrodinger import AggregateByScore
from tests.basetest import BaseTest


class TestAggregateByScore(BaseTest):
    _Link = AggregateByScore
    _classparams = {"group_by": "__id__", "score_column": "docking_score", "mode": "min"}
    _alt_classparams = {"group_by": "__enum_id__", "score_column": "glide_gscore", "mode": "max"}

    @pytest.fixture
    def sample_dataframe(self):
        """Override with expanded DataFrame that has __id__ and scores."""
        return pd.DataFrame(
            {
                "__id__": [0, 0, 0, 1, 1, 2],
                "__enum_id__": [0, 1, 2, 3, 4, 5],
                "Smiles": ["CCO", "CCO", "CCO", "c1ccccc1", "c1ccccc1", "CC"],
                "docking_score": [-8.5, -7.2, -6.1, -9.0, -8.0, -5.5],
                "letters1": ["a", "b", "c", "d", "e", "f"],
                "int1": [1, 2, 3, 4, 5, 6],
                "int2": [2, 3, 4, 5, 6, 7],
            }
        )


class TestAggregateMinMode:
    """Test that min mode selects the lowest score per group."""

    def test_selects_min_score(self):
        df = pd.DataFrame(
            {
                "__id__": [0, 0, 1, 1],
                "__enum_id__": [0, 1, 2, 3],
                "Smiles": ["CCO", "CCO", "CC", "CC"],
                "docking_score": [-8.5, -6.0, -7.0, -9.0],
                "variant": ["a", "b", "c", "d"],
            }
        )
        agg = AggregateByScore(mode="min")
        result = agg(df)

        assert len(result) == 2
        # Group 0: best is -8.5 (variant "a")
        row0 = result[result["__id__"] == 0].iloc[0]
        assert row0["docking_score"] == -8.5
        assert row0["variant"] == "a"
        # Group 1: best is -9.0 (variant "d")
        row1 = result[result["__id__"] == 1].iloc[0]
        assert row1["docking_score"] == -9.0
        assert row1["variant"] == "d"


class TestAggregateMaxMode:
    """Test that max mode selects the highest score per group."""

    def test_selects_max_score(self):
        df = pd.DataFrame(
            {
                "__id__": [0, 0, 1, 1],
                "__enum_id__": [0, 1, 2, 3],
                "docking_score": [-8.5, -6.0, -7.0, -9.0],
            }
        )
        agg = AggregateByScore(mode="max")
        result = agg(df)

        assert len(result) == 2
        row0 = result[result["__id__"] == 0].iloc[0]
        assert row0["docking_score"] == -6.0
        row1 = result[result["__id__"] == 1].iloc[0]
        assert row1["docking_score"] == -7.0


class TestAggregatePreservesColumns:
    """Test that all columns from the best row are preserved."""

    def test_preserves_all_columns(self):
        df = pd.DataFrame(
            {
                "__id__": [0, 0],
                "__enum_id__": [0, 1],
                "Smiles": ["CCO", "CCO"],
                "docking_score": [-8.5, -6.0],
                "extra_data": ["keep_this", "not_this"],
                "numeric_col": [42, 99],
            }
        )
        agg = AggregateByScore(mode="min")
        result = agg(df)

        assert len(result) == 1
        assert result.iloc[0]["extra_data"] == "keep_this"
        assert result.iloc[0]["numeric_col"] == 42


class TestAggregateErrorHandling:
    """Test error row handling during aggregation."""

    def test_error_only_group_preserved(self):
        """Groups with only error rows should get their error row back."""
        df = pd.DataFrame(
            {
                "__id__": [0, 0, 1],
                "__enum_id__": [0, 1, 2],
                "docking_score": [-8.5, -6.0, np.nan],
                "__error__": [None, None, "LigPrep: failed"],
            }
        )
        agg = AggregateByScore(mode="min")
        result = agg(df)

        assert len(result) == 2
        # Group 0: valid, best score
        valid_row = result[result["__id__"] == 0].iloc[0]
        assert valid_row["docking_score"] == -8.5
        # Group 1: error-only
        error_row = result[result["__id__"] == 1].iloc[0]
        assert error_row["__error__"] == "LigPrep: failed"

    def test_mixed_group_ignores_error_rows(self):
        """Within a group, error rows should be ignored in score comparison."""
        df = pd.DataFrame(
            {
                "__id__": [0, 0, 0],
                "__enum_id__": [0, 1, 2],
                "docking_score": [-8.5, np.nan, -6.0],
                "__error__": [None, "failed", None],
            }
        )
        agg = AggregateByScore(mode="min")
        result = agg(df)

        assert len(result) == 1
        assert result.iloc[0]["docking_score"] == -8.5


class TestAggregateDropColumns:
    """Test drop_group_columns option."""

    def test_drop_group_columns(self):
        df = pd.DataFrame(
            {
                "__id__": [0, 1],
                "__enum_id__": [0, 1],
                "docking_score": [-8.5, -6.0],
            }
        )
        agg = AggregateByScore(drop_group_columns=True)
        result = agg(df)

        assert "__id__" not in result.columns
        assert "__enum_id__" not in result.columns

    def test_keep_group_columns_by_default(self):
        df = pd.DataFrame(
            {
                "__id__": [0, 1],
                "__enum_id__": [0, 1],
                "docking_score": [-8.5, -6.0],
            }
        )
        agg = AggregateByScore()
        result = agg(df)

        assert "__id__" in result.columns
        assert "__enum_id__" in result.columns


class TestAggregateInvalidMode:
    """Test that invalid mode raises ValueError."""

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            AggregateByScore(mode="average")
