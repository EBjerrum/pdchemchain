from dataclasses import dataclass

import pandas as pd

from pdchemchain.base import (
    RowLink,
)

# The abstract class that gives us Link functionality
from pdchemchain.typing import InColumnName  # Incolumn name is a custom type


@dataclass  # Must be a dataclass
class LinearModelRow(RowLink):
    """Example of a custom link

    Calculates y based on a slope and bias and values in x column"""

    slope: float
    bias: float
    in_column: InColumnName = "x"  # InColumnName is a special typehint, that will ensure that the dataframe is checked for existence of the column name before processing and giving adequate error information
    out_column: str = "y"

    def _row_apply(
        self, row: pd.Series
    ) -> pd.Series:  # For RowLink we overload the _row_apply function
        x = float(
            row[self.in_column]
        )  # We simply index into the row with the column name, for a little bit extra robustness we try and cast the value as a float

        row[self.out_column] = (
            self.slope * x + self.bias
        )  # We simply assign the new value under the wanted column name

        return row  # We simply return the modified row
