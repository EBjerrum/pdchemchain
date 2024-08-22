from dataclasses import dataclass, field

import pandas as pd

from pdchemchain.base import Link
from pdchemchain.io_utilities import df_process_to_csv


@dataclass
class StripErrors(Link):
    """Strips row with errors from the dataframe

    After the action the property .errors_df contains the rows that had errors"""

    filename: str = None

    def __post_init__(self):
        super().__post_init__()
        self.error_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def has_errors(self) -> bool:
        return not self.error_df.empty

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "__error__" in df:
            noerror_mask = df.__error__.isna()
            error_df = df[~noerror_mask]
            noerrors_df = df[noerror_mask]
            noerrors_df = noerrors_df.drop(["__error__"], axis=1)
            self.logger.info(
                f"Stripped {len(error_df)} rows with errors. Available for inspection in .error_df attribute."
            )
        else:
            error_df = pd.DataFrame()
            noerrors_df = df
            self.logger.debug("No errors found")

        self.error_df = error_df

        if self.filename and not error_df.empty:
            self.logger.info(f"Saving rows with errors in {self.filename}.")
            df_process_to_csv(self.filename)

        return noerrors_df
