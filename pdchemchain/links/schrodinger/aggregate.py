"""Score-based aggregation for expanded DataFrames."""

from dataclasses import dataclass

import pandas as pd

from pdchemchain.base import Link
from pdchemchain.typing import InColumnName, Partitionable


@dataclass
class AggregateByScore(Link):
    """Select the best-scoring row per group from an expanded DataFrame.

    Typically used after LigPrep + GlideDock to collapse back to one row per
    original compound. All columns from the best-scoring row are preserved,
    including docked poses and other metadata.

    Parameters
    ----------
    group_by : str
        Column to group by (default: ``__id__``).
    score_column : str
        Column containing scores to compare.
    mode : str
        ``"min"`` for docking scores (lower is better) or ``"max"`` for other scores.
    drop_group_columns : bool
        Whether to drop ``__id__`` and ``__enum_id__`` after aggregation.

    Examples
    --------
    >>> from pdchemchain.links.schrodinger import AggregateByScore
    >>> agg = AggregateByScore()
    >>> collapsed = agg(expanded_df)  # One row per __id__, best docking_score
    """

    group_by: InColumnName = "__id__"
    score_column: InColumnName = "docking_score"
    mode: str = "min"
    drop_group_columns: bool = False

    _partitionable = Partitionable.NO

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{self.mode}'")

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Separate rows with errors (no valid score to compare)
        has_error_col = "__error__" in df.columns
        if has_error_col:
            error_mask = df["__error__"].notna()
            error_df = df[error_mask]
            valid_df = df[~error_mask]
        else:
            error_df = pd.DataFrame(columns=df.columns)
            valid_df = df

        self.logger.info(
            f"AggregateByScore: {len(valid_df)} valid rows, "
            f"{len(error_df)} error rows, grouping by '{self.group_by}'"
        )

        if valid_df.empty:
            result = error_df
        else:
            # Select best row per group
            if self.mode == "min":
                best_idx = valid_df.groupby(self.group_by)[self.score_column].idxmin()
            else:
                best_idx = valid_df.groupby(self.group_by)[self.score_column].idxmax()

            best_df = valid_df.loc[best_idx]

            # Reinsert error-only groups: compounds where ALL rows have errors
            if has_error_col and not error_df.empty:
                valid_groups = set(best_df[self.group_by])
                error_only = error_df[~error_df[self.group_by].isin(valid_groups)]
                if not error_only.empty:
                    # Take first error row per group
                    error_representatives = error_only.groupby(self.group_by).first()
                    error_representatives = error_representatives.reset_index()
                    best_df = pd.concat(
                        [best_df, error_representatives], ignore_index=True
                    )

            result = best_df

        # Optionally drop group columns
        if self.drop_group_columns:
            for col in ["__id__", "__enum_id__"]:
                if col in result.columns:
                    result = result.drop(columns=[col])

        result = result.reset_index(drop=True)
        self.logger.info(f"AggregateByScore: {len(df)} rows → {len(result)} rows")
        return result
