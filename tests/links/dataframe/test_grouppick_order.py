"""GroupPick(action="collapse") must preserve the input's group ORDER, not just its count.

Regression test. The collapse path concatenates the picks, the all-NaN representatives and
the error-only representatives, which left failed groups bunched at the tail of the frame
while the row count stayed correct. Consumers that read columns positionally -- the REINVENT
PdChemChain scoring component does exactly that -- then attached every value to the wrong
row, with nothing raising.
"""
import numpy as np
import pandas as pd

from pdchemchain.links.dataframe import GroupPick


def _expanded_frame() -> pd.DataFrame:
    """A post-docking frame: one expanded group, one all-NaN group, one errored group.

    Group 1 has no usable score, group 3 errored upstream, and group 2 enumerated into two
    variants of which the second scores better. Groups are deliberately NOT in sorted
    order relative to their failure modes, so a sort-based "fix" would not pass.
    """
    return pd.DataFrame(
        {
            "__id__": [0, 1, 2, 2, 3, 4],
            "__enum_id__": [0, 0, 0, 1, 0, 0],
            "name": ["m0", "m1", "m2", "m2", "m3", "m4"],
            "docking_score": [-8.0, np.nan, -7.0, -9.0, np.nan, -6.0],
            "__error__": [None, None, None, None, "LigPrep failed", None],
        }
    )


def _collapse(df: pd.DataFrame) -> pd.DataFrame:
    return GroupPick(
        group_by="__id__", score_column="docking_score", mode="min", action="collapse"
    )(df)


def test_collapse_preserves_input_group_order():
    df = _expanded_frame()
    result = _collapse(df)

    assert result["__id__"].tolist() == [0, 1, 2, 3, 4]
    assert result["name"].tolist() == ["m0", "m1", "m2", "m3", "m4"]


def test_collapse_keeps_one_row_per_group():
    df = _expanded_frame()
    result = _collapse(df)

    assert len(result) == df["__id__"].nunique()


def test_collapse_keeps_failures_in_place_as_nan():
    """A group that could not be scored stays at its own position with a NaN score."""
    result = _collapse(_expanded_frame())
    scores = result.set_index("__id__")["docking_score"]

    assert np.isnan(scores[1]), "all-NaN group should remain NaN"
    assert np.isnan(scores[3]), "errored group should remain NaN"


def test_collapse_picks_the_best_variant():
    result = _collapse(_expanded_frame())
    scores = result.set_index("__id__")["docking_score"]

    assert scores[2] == -9.0, "mode='min' must keep the lowest-scoring variant"


def test_collapse_order_is_input_order_not_sorted():
    """Guard against a sort-based fix: shuffled input groups must come back shuffled."""
    df = _expanded_frame()
    df["__id__"] = df["__id__"].map({0: 40, 1: 10, 2: 30, 3: 20, 4: 0})
    result = _collapse(df)

    assert result["__id__"].tolist() == [40, 10, 30, 20, 0]
    assert result["name"].tolist() == ["m0", "m1", "m2", "m3", "m4"]


def test_collapse_when_every_row_errored():
    """A wholesale backend failure must still collapse to one row per group, in order.

    This is the Glide-failed-entirely case: without it the EXPANDED frame comes back (more
    rows out than groups in), which breaks positional consumers harder than reordering.
    """
    df = _expanded_frame()
    df["__error__"] = "GlideDock: no docking result"
    df["docking_score"] = np.nan

    result = _collapse(df)

    assert len(result) == df["__id__"].nunique() == 5
    assert result["__id__"].tolist() == [0, 1, 2, 3, 4]
    assert result["name"].tolist() == ["m0", "m1", "m2", "m3", "m4"]
    assert result["docking_score"].isna().all()


def test_tag_mode_is_unaffected():
    """Tag mode never reordered; it must still return every input row untouched."""
    df = _expanded_frame()
    result = GroupPick(
        group_by="__id__", score_column="docking_score", mode="min", action="tag"
    )(df)

    assert len(result) == len(df)
    assert result["__id__"].tolist() == df["__id__"].tolist()
    # Groups 0, 2 and 4 each get one pick; groups 1 and 3 have no usable score, so they
    # get none -- tag mode keeps their rows but leaves them unmarked.
    assert result["picked"].sum() == 3
    assert result.loc[result["picked"], "name"].tolist() == ["m0", "m2", "m4"]
