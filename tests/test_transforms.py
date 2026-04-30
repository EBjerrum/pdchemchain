"""Tests for score transformation links."""
import math
import pytest
import pandas as pd

from pdchemchain.links.transforms import (
    StepTransform,
    LinearTransform,
    SigmoidTransform,
    GaussianTransform,
    PlateauTransform,
)
from tests.basetest import BaseTest


# ---------------------------------------------------------------------------
# BaseTest wiring
# ---------------------------------------------------------------------------

class TestStepTransform(BaseTest):
    _Link = StepTransform
    _classparams = {"in_column": "int1", "out_column": "score_t", "good_value": 3.0, "bad_value": 1.0}
    _alt_classparams = {"in_column": "int2", "out_column": "score_t2", "good_value": 2.0, "bad_value": 0.5}


class TestLinearTransform(BaseTest):
    _Link = LinearTransform
    _classparams = {"in_column": "int1", "out_column": "score_t", "good_value": 3.0, "bad_value": 1.0}
    _alt_classparams = {"in_column": "int2", "out_column": "score_t2", "good_value": 4.0, "bad_value": 2.0}


class TestSigmoidTransform(BaseTest):
    _Link = SigmoidTransform
    _classparams = {"in_column": "int1", "out_column": "score_t", "good_value": 3.0, "bad_value": 1.0}
    _alt_classparams = {"in_column": "int2", "out_column": "score_t2", "good_value": 4.0, "bad_value": 2.0}


class TestGaussianTransform(BaseTest):
    _Link = GaussianTransform
    _classparams = {"in_column": "int1", "out_column": "score_t", "good_value": 2.0, "bad_value": 4.0}
    _alt_classparams = {"in_column": "int2", "out_column": "score_t2", "good_value": 3.0, "bad_value": 5.0}


class TestPlateauTransform(BaseTest):
    _Link = PlateauTransform
    _classparams = {"in_column": "int1", "out_column": "score_t",
                    "good_value": (1.0, 3.0), "bad_value": (-1.0, 6.0)}
    _alt_classparams = {"in_column": "int2", "out_column": "score_t2",
                        "good_value": (1.5, 3.5), "bad_value": (-0.5, 7.0)}


# ---------------------------------------------------------------------------
# StepTransform behaviour
# ---------------------------------------------------------------------------

def test_step_ascending():
    t = StepTransform(good_value=0.7, bad_value=0.4)
    assert t._compute(0.5) == 1.0   # above bad → good
    assert t._compute(0.4) == 0.0   # equal to bad → 0.0 (strictly >)
    assert t._compute(0.3) == 0.0   # below bad → bad


def test_step_descending():
    t = StepTransform(good_value=2.0, bad_value=5.0)
    assert t._compute(3.0) == 1.0   # below bad → good
    assert t._compute(5.0) == 0.0   # equal to bad → 0.0 (strictly <)
    assert t._compute(6.0) == 0.0   # above bad → bad


# ---------------------------------------------------------------------------
# LinearTransform behaviour
# ---------------------------------------------------------------------------

def test_linear_ascending():
    t = LinearTransform(good_value=1.0, bad_value=0.0)
    assert t._compute(0.0) == pytest.approx(0.0)
    assert t._compute(0.5) == pytest.approx(0.5)
    assert t._compute(1.0) == pytest.approx(1.0)


def test_linear_descending():
    t = LinearTransform(good_value=400.0, bad_value=600.0)
    assert t._compute(600.0) == pytest.approx(0.0)
    assert t._compute(500.0) == pytest.approx(0.5)
    assert t._compute(400.0) == pytest.approx(1.0)


def test_linear_clipping():
    t = LinearTransform(good_value=1.0, bad_value=0.0)
    assert t._compute(-0.5) == 0.0   # below bad → clipped to 0
    assert t._compute(1.5) == 1.0   # above good → clipped to 1


# ---------------------------------------------------------------------------
# SigmoidTransform behaviour
# ---------------------------------------------------------------------------

def test_sigmoid_midpoint_is_half():
    t = SigmoidTransform(good_value=10.0, bad_value=0.0)
    midpoint = 5.0
    assert t._compute(midpoint) == pytest.approx(0.5, abs=1e-6)


def test_sigmoid_auto_k_targets():
    t = SigmoidTransform(good_value=10.0, bad_value=0.0)
    assert t._compute(10.0) == pytest.approx(0.98, abs=1e-4)
    assert t._compute(0.0) == pytest.approx(0.02, abs=1e-4)


def test_sigmoid_descending():
    t = SigmoidTransform(good_value=-10.0, bad_value=-4.0)
    assert t._compute(-10.0) >= 0.98
    assert t._compute(-4.0) <= 0.02
    assert t._compute(-7.0) == pytest.approx(0.5, abs=1e-4)


def test_sigmoid_explicit_k():
    t = SigmoidTransform(good_value=10.0, bad_value=0.0, k=1.0)
    # With explicit k the score at good/bad may not hit 0.98/0.02
    assert 0.0 < t._compute(5.0) < 1.0


# ---------------------------------------------------------------------------
# GaussianTransform behaviour
# ---------------------------------------------------------------------------

def test_gaussian_peak_is_one():
    t = GaussianTransform(good_value=2.5, bad_value=5.5)
    assert t._compute(2.5) == pytest.approx(1.0)


def test_gaussian_bad_value_near_zero():
    t = GaussianTransform(good_value=2.5, bad_value=5.5)
    # At 3-sigma distance score should be ~0.011
    assert t._compute(5.5) == pytest.approx(math.exp(-0.5 * 3**2), rel=1e-4)


def test_gaussian_symmetric():
    t = GaussianTransform(good_value=2.5, bad_value=5.5)
    delta = abs(t.good_value - t.bad_value)
    # Score at equal distance on both sides should be the same
    assert t._compute(2.5 + 1.0) == pytest.approx(t._compute(2.5 - 1.0), rel=1e-6)


# ---------------------------------------------------------------------------
# PlateauTransform behaviour
# ---------------------------------------------------------------------------

def test_plateau_inside_good_is_near_one():
    t = PlateauTransform(good_value=(1.0, 3.0), bad_value=(-1.0, 6.0))
    # Boundaries score ≈ 0.98 (epsilon target); centre scores higher
    assert t._compute(1.0) == pytest.approx(0.98, abs=1e-3)
    assert t._compute(3.0) == pytest.approx(0.98, abs=1e-3)
    assert t._compute(2.0) > t._compute(1.0)  # centre higher than boundary


def test_plateau_outside_bad_is_near_zero():
    t = PlateauTransform(good_value=(1.0, 3.0), bad_value=(-1.0, 6.0))
    assert t._compute(-1.0) == pytest.approx(0.02, abs=1e-4)
    assert t._compute(6.0) == pytest.approx(0.02, abs=1e-4)


def test_plateau_ascending_shoulder():
    t = PlateauTransform(good_value=(1.0, 3.0), bad_value=(-1.0, 6.0))
    # Between bad_lo and good_lo: score should increase
    assert t._compute(0.0) > t._compute(-0.5)
    assert t._compute(0.9) > t._compute(0.0)


def test_plateau_bad_must_enclose_good():
    with pytest.raises(ValueError):
        PlateauTransform(good_value=(1.0, 3.0), bad_value=(0.0, 2.0))  # bad_hi < good_hi


def test_plateau_tuple_coercion():
    # YAML round-trip returns lists; __post_init__ should coerce back to tuples
    t = PlateauTransform(good_value=[1.0, 3.0], bad_value=[-1.0, 6.0])
    assert isinstance(t.good_value, tuple)
    assert isinstance(t.bad_value, tuple)


# ---------------------------------------------------------------------------
# Extreme values (overflow guard)
# ---------------------------------------------------------------------------

def test_sigmoid_extreme_bad_side():
    """Extreme values on the bad side should return ~0.0, not OverflowError."""
    t = SigmoidTransform(good_value=-10.0, bad_value=-4.0)
    assert t._compute(1000.0) == pytest.approx(0.0, abs=1e-6)


def test_plateau_extreme_values():
    t = PlateauTransform(good_value=(1.0, 3.0), bad_value=(-1.0, 6.0))
    assert t._compute(-10000.0) == pytest.approx(0.0, abs=1e-6)
    assert t._compute(10000.0) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# DataFrame round-trip
# ---------------------------------------------------------------------------

def test_transform_on_dataframe():
    df = pd.DataFrame({"score": [0.0, 0.5, 1.0]})
    result = LinearTransform(good_value=1.0, bad_value=0.0)(df)
    assert list(result["score_t"]) == pytest.approx([0.0, 0.5, 1.0])


def test_transform_nan_propagates():
    df = pd.DataFrame({"score": [float("nan"), 0.5]})
    result = LinearTransform(good_value=1.0, bad_value=0.0)(df)
    assert math.isnan(result["score_t"].iloc[0])
    assert result["score_t"].iloc[1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Serialisation round-trip (covers the base.py from_params fix)
# ---------------------------------------------------------------------------

def test_sigmoid_roundtrip():
    t = SigmoidTransform(good_value=-10.0, bad_value=-4.0, k=2.0)
    from pdchemchain.base import Link
    t2 = Link.from_params(t.get_params())
    assert t2.good_value == t.good_value
    assert t2.bad_value == t.bad_value
    assert t2.k == t.k


def test_plateau_roundtrip():
    t = PlateauTransform(good_value=(1.0, 3.0), bad_value=(-1.0, 6.0))
    from pdchemchain.base import Link
    t2 = Link.from_params(t.get_params())
    assert tuple(t2.good_value) == t.good_value
    assert tuple(t2.bad_value) == t.bad_value
