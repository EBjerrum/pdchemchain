from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from pdchemchain.base import Link
from pdchemchain.io_utilities import df_process_to_csv


@dataclass
class FromFile(Link):
    """Reads dataframe from a CSV file"""

    filename: str
    pd_readcsv_options: Dict[str, any] = field(default_factory=lambda: {"sep": ","})

    def apply(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = pd.DataFrame()
        return super().apply(df)

    def __call__(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self.apply(df)

    def _apply(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = pd.DataFrame()
        if not df.empty:
            self.logger.warning(
                f"Input dataframe received in FromFile is not empty as expected but contained {len(df)} rows, possible error as it will get overwritten"
            )
        df = pd.read_csv(self.filename, **self.pd_readcsv_options)
        self.logger.info(
            f"Loaded dataframe from CSV file {self.filename}, dataframe has {len(df)} rows."
        )
        return df


@dataclass
class ToFile(Link):
    """Writes the dataframe to a CSV file"""

    filename: str
    pd_tocsv_options: Dict[str, any] = field(default_factory=lambda: {"sep": ","})

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_process_to_csv(df, self.filename, **self.pd_tocsv_options)
        self.logger.info(f"Saved dataframe with basename {self.filename}")
        return df
