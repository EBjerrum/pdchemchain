"""Tests for score aggregation links."""
import math
import pytest
import pandas as pd

from pdchemchain.links.aggregators import WeightedSum, WeightedGeometricMean, ArithmeticMean
from tests.basetest import BaseTest


# ---------------------------------------------------------------------------
# BaseTest wiring
# ---------------------------------------------------------------------------

class TestWeightedSum(BaseTest):
    _Link = WeightedSum
    _classparams = {"columns": ["a", "b"], "weights": [0.6, 0.4], "out_column": "score"}
    _alt_classparams = {"columns": ["c", "d"], "weights": [0.5, 0.5], "out_column": "score2"}


class TestWeightedGeometricMean(BaseTest):
    _Link = WeightedGeometricMean
    _classparams = {"columns": ["a", "b"], "weights": [0.6, 0.4], "out_column": "score"}
    _alt_classparams = {"columns": ["c", "d"], "weights": [0.5, 0.5], "out_column": "score2"}


class TestArithmeticMean(BaseTest):
    _Link = ArithmeticMean
    _classparams = {"columns": ["a", "b"], "out_column": "score"}
    _alt_classparams = {"columns": ["c", "d"], "out_column": "score2"}


# ---------------------------------------------------------------------------
# WeightedSum behaviour
# ---------------------------------------------------------------------------

def _df(*cols_values):
    """Build a single-row DataFrame from keyword-like pairs: _df('a', 0.8, 'b', 0.4)."""
    cols = cols_values[0::2]
    vals = cols_values[1::2]
    return pd.DataFrame({c: [v] for c, v in zip(cols, vals)})


def test_weighted_sum_basic():
    df = _df("a", 0.8, "b", 0.4)
    result = WeightedSum(columns=["a", "b"], weights=[1.0, 1.0])(df)
    assert result["score"].iloc[0] == pytest.approx(0.6)


def test_weighted_sum_normalises_weights():
    # weights [2, 2] should behave like [1, 1]
    df = _df("a", 0.8, "b", 0.4)
    r1 = WeightedSum(columns=["a", "b"], weights=[1.0, 1.0])(df)
    r2 = WeightedSum(columns=["a", "b"], weights=[2.0, 2.0])(df)
    assert r1["score"].iloc[0] == pytest.approx(r2["score"].iloc[0])


def test_weighted_sum_nan_is_zero():
    df = _df("a", float("nan"), "b", 1.0)
    result = WeightedSum(columns=["a", "b"], weights=[0.5, 0.5])(df)
    assert result["score"].iloc[0] == pytest.approx(0.5)


def test_weighted_sum_default_equal_weights():
    df = _df("a", 0.8, "b", 0.4)
    result = WeightedSum(columns=["a", "b"])(df)
    assert result["score"].iloc[0] == pytest.approx(0.6)


def test_weighted_sum_length_mismatch():
    with pytest.raises(ValueError):
        WeightedSum(columns=["a", "b"], weights=[1.0])


# ---------------------------------------------------------------------------
# WeightedGeometricMean behaviour
# ---------------------------------------------------------------------------

def test_geometric_mean_equal_weights():
    df = _df("a", 0.8, "b", 0.5)
    result = WeightedGeometricMean(columns=["a", "b"], weights=[1.0, 1.0])(df)
    expected = math.sqrt(0.8 * 0.5)
    assert result["score"].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_geometric_mean_zero_input():
    df = _df("a", 0.0, "b", 1.0)
    result = WeightedGeometricMean(columns=["a", "b"], weights=[1.0, 1.0])(df)
    # a=0 → floored to 1e-6, so result is tiny but > 0
    assert result["score"].iloc[0] < 0.01


def test_geometric_mean_nan_is_zero():
    df = _df("a", float("nan"), "b", 1.0)
    result = WeightedGeometricMean(columns=["a", "b"], weights=[1.0, 1.0])(df)
    # NaN → 0.0 → geometric mean is near 0
    assert result["score"].iloc[0] < 0.01


def test_geometric_mean_ones():
    df = _df("a", 1.0, "b", 1.0)
    result = WeightedGeometricMean(columns=["a", "b"], weights=[1.0, 1.0])(df)
    assert result["score"].iloc[0] == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# ArithmeticMean behaviour
# ---------------------------------------------------------------------------

def test_arithmetic_mean_basic():
    df = _df("a", 0.8, "b", 0.4)
    result = ArithmeticMean(columns=["a", "b"])(df)
    assert result["score"].iloc[0] == pytest.approx(0.6)


def test_arithmetic_mean_nan_is_zero():
    df = _df("a", float("nan"), "b", 0.8)
    result = ArithmeticMean(columns=["a", "b"])(df)
    assert result["score"].iloc[0] == pytest.approx(0.4)


def test_arithmetic_mean_three_columns():
    df = _df("a", 1.0, "b", 0.5, "c", 0.0)
    result = ArithmeticMean(columns=["a", "b", "c"])(df)
    assert result["score"].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Serialisation round-trip (covers list[float] weights)
# ---------------------------------------------------------------------------

def test_weighted_sum_roundtrip():
    agg = WeightedSum(columns=["a", "b"], weights=[0.6, 0.4], out_column="total")
    from pdchemchain.base import Link
    agg2 = Link.from_params(agg.get_params())
    assert agg2.columns == agg.columns
    assert agg2.weights == pytest.approx(agg.weights)
    assert agg2.out_column == agg.out_column


def test_geometric_mean_roundtrip():
    agg = WeightedGeometricMean(columns=["x", "y"], weights=[1.0, 2.0])
    from pdchemchain.base import Link
    agg2 = Link.from_params(agg.get_params())
    assert agg2.columns == agg.columns
    assert agg2.weights == pytest.approx(agg.weights)
