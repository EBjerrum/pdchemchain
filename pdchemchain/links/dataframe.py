from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from pdchemchain.base import Link, RowLink
from pdchemchain.typing import InColumnName, Partitionable


@dataclass
class DfEval(Link):
    """Flexible application of operations using pandas .eval method"""

    _partitionable = Partitionable.MAYBE

    eval_str: str
    out_column: Optional[str] = ""

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method that runs an eval on the dataframe. Only use in trusted code, as it allow for potential injection
        Useful for computing e.g. ratios between named columns, i.e. eval='column_A/column_B'
        Moreover, the eval is not error handled, so one row with wrong input will make the entire apply fail"""
        # TODO, figure out a way to do proper error handling, or at least not work on rows with anything in __error__
        if self.out_column:
            df = df.copy()
            df[self.out_column] = df.eval(self.eval_str)
        else:
            df = df.eval(self.eval_str)
        self.logger.debug(f"Used {self.eval_str} expression on dataframe")

        return df


@dataclass
class DropColumns(Link):
    """Drops columns based on defined list of columns names"""

    columns: List[str] = field(
        default_factory=list
    )  # TODO, make assertation mechanism for multiple existing columns

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"Will drop {self.columns} columns from dataframe.")
        df_dropped = df.drop(self.columns, axis=1)
        self.logger.debug(f"Columns remaining in dataframe: {df_dropped.columns}")
        return df_dropped


@dataclass
class DropDuplicates(Link):
    """Drops duplicates from dataframe based on values in the columns list"""

    _partitionable = Partitionable.NO

    columns: List[str] = field(
        default_factory=list
    )  # TODO, make assertation mechanism for multiple existing columns

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns:
            df_no_duplicates = df.drop_duplicates(subset=self.columns)
            self.logger.debug(
                f"Dropped {len(df)-len(df_no_duplicates)} duplicates. Rows remaining: {len(df_no_duplicates)}"
            )
        else:
            df_no_duplicates = df
            self.logger.warning(
                f"No subset columns defined ({self.columns=}), returning dataframe unchanged. This may not be what you inteded."
            )
        return df_no_duplicates


@dataclass
class DropTable(Link):
    """Forwards a new empty dataframe

    useful if a link chain in e.g. UnionLink should not be merged back into the output
    (as it was saved or something)"""

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(
            "Dropping dataframe with {len(df)} rows. Forwarding an empty dataframe."
        )
        return pd.DataFrame()


@dataclass
class KeepColumns(Link):
    """Keep only the columns with the specified column names"""

    columns: List[str] = field(
        default_factory=list
    )  # TODO, make assertation mechanism for multiple existing columns

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filtered = df.loc[:, self.columns]
        self.logger.debug(
            f"Kept {df_filtered.columns} columns. Dropped {set(df.columns).difference(set(df_filtered.columns))}."
        )
        return df_filtered


@dataclass
class NullLink(Link):
    """The NullLink does nothing to the dataframe

    Parameters
    ----------
    name : str
        A custom name for the nulllink
    """

    name: str = "NullLink"

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """This Link pass the dataframe unaltered (e.g. for demo purposes)"""
        self.logger.debug(f"Applying link {self.name}")
        return df


@dataclass
class RowEval(RowLink):
    """eval must be with columns as 'row.column_name', e.g. 'row.A + row.B'

    In comparison with DfEval, this one applies row_wise and with error handling"""

    eval_str: str
    out_column: str

    def _row_apply(self, row: pd.Series) -> pd.Series:
        row[self.out_column] = pd.eval(
            self.eval_str, target=row
        )  # Seems impossible to get row assignments of new columns using this
        self.logger.debug(f"Used {self.eval_str} expression on dataframe")

        return row


@dataclass
class Query(Link):
    """Filters a dataframe based on the query string"""

    query: str = ""

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.query:
            df_query = df.query(self.query)
            self.logger.debug(
                f"Rows before query: {len(df)}, rows after query: {len(df_query)}"
            )
        else:
            df_query = df
            self.logger.warning(
                f"No query defined ({self.query=}), returning dataframe unchanged. This may not be what you inteded."
            )
        return df_query


@dataclass
class RenameColumns(Link):
    """Renames columns in the dataframe based on a provided mapping"""

    columns: dict[str, str] = field(default_factory=dict)

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns:
            df_renamed = df.rename(columns=self.columns)
            renamed_cols = set(self.columns.keys()) & set(df.columns)
            self.logger.debug(
                f"Renamed {len(renamed_cols)} columns: {', '.join(renamed_cols)}"
            )
        else:
            df_renamed = df
            self.logger.warning(
                f"No column mapping defined ({self.columns=}), returning dataframe unchanged. This may not be what you intended."
            )
        return df_renamed


@dataclass
class GroupPick(Link):
    """Select the best-scoring row per group, either tagging or collapsing.

    Two modes of operation:

    - **action="tag"** (default): Adds a boolean column marking the best row per
      group. All rows are preserved — filter to picks with ``Query`` afterward.
    - **action="collapse"**: Drops non-best rows, returning one row per group.
      Error-only groups (all rows have ``__error__``) are preserved with a
      representative error row.

    An optional ``filter_column`` restricts which rows compete: only rows where
    that column is truthy are considered. Groups with no eligible rows get no pick
    (tag mode) or are dropped (collapse mode, unless error-only).

    Parameters
    ----------
    group_by : str
        Column to group by.
    score_column : str
        Column containing scores to compare.
    mode : str
        ``"min"`` (lower is better) or ``"max"`` (higher is better).
    action : str
        ``"tag"`` to mark picks in a boolean column, ``"collapse"`` to drop
        non-best rows.
    filter_column : str, optional
        Only rows where this column is truthy compete for best score.
    out_column : str
        Boolean column name for tag mode (ignored in collapse mode).
    pick_noise : bool
        Whether to pick from noise clusters (group value ``-1``).
    drop_group_columns : bool
        In collapse mode, drop ``__id__`` and ``__enum_id__`` columns afterward.

    Examples
    --------
    Tag best per cluster (keep all rows):

    >>> pick = GroupPick(group_by="ButinaCluster", score_column="docking_score")
    >>> result = pick(clustered_df)  # adds "picked" boolean column

    Collapse to one row per __id__ (AggregateByScore replacement):

    >>> agg = GroupPick(group_by="__id__", score_column="docking_score",
    ...                 action="collapse")
    >>> result = agg(expanded_df)  # one row per __id__
    """

    group_by: InColumnName = "ButinaCluster"
    score_column: InColumnName = "docking_score"
    mode: str = "min"
    action: str = "tag"
    filter_column: str = None
    out_column: str = "picked"
    pick_noise: bool = False
    drop_group_columns: bool = False

    _partitionable = Partitionable.NO

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{self.mode}'")
        if self.action not in ("tag", "collapse"):
            raise ValueError(
                f"action must be 'tag' or 'collapse', got '{self.action}'"
            )

    def _get_best_indices(self, df: pd.DataFrame) -> tuple[pd.Index, set]:
        """Find the index of the best-scoring row per group in df.

        Returns
        -------
        best_indices : pd.Index
            Index values of the best-scoring row per group (NaN-only groups excluded).
        nan_groups : set
            Group keys where all scores were NaN (no pick possible).
        """
        grouped = df.groupby(self.group_by)[self.score_column]
        if self.mode == "min":
            best = grouped.idxmin()
        else:
            best = grouped.idxmax()
        # idxmin/idxmax returns NaN for all-NaN groups
        nan_groups = set(best[best.isna()].index)
        best = best.dropna()
        return best, nan_groups

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.action == "tag":
            return self._apply_tag(df)
        else:
            return self._apply_collapse(df)

    def _apply_tag(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.out_column] = False

        # Determine eligible rows
        eligible = df
        if self.filter_column is not None:
            eligible = eligible[eligible[self.filter_column].astype(bool)]
        if not self.pick_noise:
            eligible = eligible[eligible[self.group_by] != -1]

        if eligible.empty:
            self.logger.warning("GroupPick: no eligible rows to pick from")
            return df

        best_idx, nan_groups = self._get_best_indices(eligible)
        if nan_groups:
            self.logger.warning(
                f"GroupPick(tag): {len(nan_groups)} groups with all-NaN scores, no pick"
            )
        df.loc[best_idx.values, self.out_column] = True

        n_picked = best_idx.shape[0]
        self.logger.info(
            f"GroupPick(tag): {n_picked} picks from {len(df)} rows "
            f"({eligible.shape[0]} eligible)"
        )
        return df

    def _apply_collapse(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Separate error rows
        has_error_col = "__error__" in df.columns
        if has_error_col:
            error_mask = df["__error__"].notna()
            error_df = df[error_mask]
            valid_df = df[~error_mask]
        else:
            error_df = pd.DataFrame(columns=df.columns)
            valid_df = df

        # Apply filter
        if self.filter_column is not None:
            valid_df = valid_df[valid_df[self.filter_column].astype(bool)]

        self.logger.info(
            f"GroupPick(collapse): {len(valid_df)} valid rows, "
            f"{len(error_df)} error rows, grouping by '{self.group_by}'"
        )

        if valid_df.empty:
            result = error_df
        else:
            best_idx, nan_groups = self._get_best_indices(valid_df)
            best_df = valid_df.loc[best_idx]

            # Groups where all scores were NaN — pick first row, mark as error
            if nan_groups:
                self.logger.warning(
                    f"GroupPick(collapse): {len(nan_groups)} groups with all-NaN "
                    f"scores, selecting first row as fallback"
                )
                nan_rows = valid_df[valid_df[self.group_by].isin(nan_groups)]
                nan_representatives = (
                    nan_rows.groupby(self.group_by).first().reset_index()
                )
                nan_representatives["__error__"] = (
                    f"GroupPick: no valid {self.score_column} to compare"
                )
                best_df = pd.concat(
                    [best_df, nan_representatives], ignore_index=True
                )

            # Reinsert error-only groups
            if has_error_col and not error_df.empty:
                covered_groups = set(best_df[self.group_by])
                error_only = error_df[~error_df[self.group_by].isin(covered_groups)]
                if not error_only.empty:
                    error_representatives = (
                        error_only.groupby(self.group_by).first().reset_index()
                    )
                    best_df = pd.concat(
                        [best_df, error_representatives], ignore_index=True
                    )

            result = best_df

        if self.drop_group_columns:
            for col in ["__id__", "__enum_id__"]:
                if col in result.columns:
                    result = result.drop(columns=[col])

        result = result.reset_index(drop=True)
        self.logger.info(f"GroupPick(collapse): {len(df)} rows -> {len(result)} rows")
        return result
