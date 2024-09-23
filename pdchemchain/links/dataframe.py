from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from pdchemchain.base import Link, RowLink


@dataclass
class DfEval(Link):
    """Flexible application of operations using pandas .eval method"""

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
