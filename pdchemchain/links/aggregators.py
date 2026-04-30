"""Score aggregation links: combine multiple [0, 1] score columns into a single score.

NaN values (e.g. from molecules that didn't reach an expensive scoring stage) are
treated as 0.0, which is the natural funnel-scoring behaviour: an unscored expensive
metric contributes nothing (or actively drags down geometric mean).
"""
import math
from dataclasses import dataclass, field

import pandas as pd

from pdchemchain.base import RowLink


@dataclass
class _WeightedAggregator(RowLink):
    """Internal base for aggregators with columns + weights."""

    columns: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    out_column: str = "score"

    def __post_init__(self):
        super().__post_init__()
        if not self.weights:
            self.weights = [1.0] * len(self.columns)
        if len(self.weights) != len(self.columns):
            raise ValueError(
                f"weights length ({len(self.weights)}) must match columns length ({len(self.columns)})"
            )
        total = sum(self.weights)
        self._norm_weights = [w / total for w in self.weights] if total else []

    def _values(self, row: pd.Series) -> list[float]:
        return [0.0 if pd.isna(row[col]) else float(row[col]) for col in self.columns]


@dataclass
class WeightedSum(_WeightedAggregator):
    """Weighted sum of score columns, normalised to [0, 1].

    Weights are normalised to sum to 1.0 internally.
    NaN values are treated as 0.0. Defaults to equal weights if omitted.

    Parameters
    ----------
    columns
        Score columns to aggregate.
    weights
        Per-column weights (any positive scale; normalised internally).
        Must have the same length as columns.
    out_column
        Column to store the aggregated score.

    Examples
    --------
    >>> agg = WeightedSum(columns=["qed_score", "sa_score_t", "mw_score"],
    ...                   weights=[1.0, 1.0, 1.0], out_column="cheap_score")
    """

    def _row_apply(self, row: pd.Series) -> pd.Series:
        values = self._values(row)
        row[self.out_column] = sum(w * v for w, v in zip(self._norm_weights, values))
        return row


@dataclass
class WeightedGeometricMean(_WeightedAggregator):
    """Weighted geometric mean of score columns.

    Formula: prod(x_i ^ w_i) ^ (1 / sum(w_i))
    NaN values are treated as 0.0. A single zero makes the whole product zero,
    which is the correct strict behaviour for funnel scoring.
    A small epsilon floor (1e-6) avoids log(0). Defaults to equal weights if omitted.

    Parameters
    ----------
    columns
        Score columns to aggregate.
    weights
        Per-column weights (any positive scale; normalised internally).
        Must have the same length as columns.
    out_column
        Column to store the aggregated score.

    Examples
    --------
    >>> agg = WeightedGeometricMean(columns=["qed_score", "dock_score"],
    ...                             weights=[0.4, 0.6], out_column="final_score")
    """

    def _row_apply(self, row: pd.Series) -> pd.Series:
        values = self._values(row)
        log_sum = sum(w * math.log(max(v, 1e-6)) for w, v in zip(self._norm_weights, values))
        row[self.out_column] = math.exp(log_sum)
        return row


@dataclass
class ArithmeticMean(RowLink):
    """Unweighted arithmetic mean of score columns.

    NaN values are treated as 0.0.

    Parameters
    ----------
    columns
        Score columns to aggregate.
    out_column
        Column to store the aggregated score.

    Examples
    --------
    >>> agg = ArithmeticMean(columns=["qed_score", "sa_score_t"], out_column="mean_score")
    """

    columns: list = field(default_factory=list)
    out_column: str = "score"

    def _row_apply(self, row: pd.Series) -> pd.Series:
        values = [0.0 if pd.isna(row[col]) else float(row[col]) for col in self.columns]
        row[self.out_column] = sum(values) / len(values) if values else 0.0
        return row
