import sys
from io import StringIO
from traceback import format_exception
from typing import List, Union

import pandas as pd


def has_error(row: pd.Series) -> bool:
    error = row.get("__error__")

    if not error:
        return False
    elif not pd.notna(error):
        return False
    else:
        return True


def rows_with_errors(
    df: pd.DataFrame, aslist: bool = False
) -> Union[pd.Series, List[bool]]:
    if not isinstance(aslist, bool):
        raise ValueError("aslist must be a boolean.")

    if "__error__" not in df:
        result = [False] * len(df)
    else:
        isna = df.__error__.isna()  # None, NaN
        empties = ~df.__error__.astype(bool)  # Empty objects, e.g. empty strings
        result = ~(empties | isna)

    return list(result) if aslist else result


class ErrorContextManager:
    def __init__(self):
        self.errors = ""

    def __enter__(self):
        # capture std.error
        sys.stderr = self.error_output_buffer = StringIO()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        errors = ""
        # Save the captured output
        stderr_output = self.error_output_buffer.getvalue()

        if stderr_output:
            # Someone wrote to stderr (e.g. RDKit) #TODO, is this _always_ errors?
            errors = errors + stderr_output + "\n"

        if exc_type is not None:
            # An exception occurred within the context
            msg = "".join(
                format_exception(exc_type, exc_value, traceback)
            )  # TODO, do this end in \n?
            errors = errors + msg

        self.errors = errors

        # reset stderr to standard
        sys.stderr = sys.__stderr__

        return True  # Returns True as we are handling the errors


class RDKitErrorContextManager:
    def __init__(self):
        self.errors = ""

    def __enter__(self):
        # capture std.error
        sys.stderr = self.error_output_buffer = StringIO()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        errors = ""
        # Save the captured output
        stderr_output = self.error_output_buffer.getvalue()

        if stderr_output:
            # Someone wrote to stderr (e.g. RDKit) #TODO, is this _always_ errors?
            errors = errors + stderr_output + "\n"

        # if exc_type is not None:
        #     # An exception occurred within the context
        #     msg = "".join(format_exception(exc_type, exc_value, traceback)) #TODO, do this end in \n?
        #     errors = errors + msg

        self.errors = errors

        # reset stderr to standard
        sys.stderr = sys.__stderr__

        # return True #Returns True as we are handling the errors
