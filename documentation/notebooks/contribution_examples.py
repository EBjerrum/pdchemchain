# %%
from dataclasses import dataclass
from typing import Union
from pdchemchain.base import (
    RowLink,
)  # The abstract class that gives us Link functionality
from pdchemchain.typing import InColumnName  # Incolumn name is a custom type
import pandas as pd


@dataclass  # Must be a dataclass
class LinearModelRow(RowLink):
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


# %% A quick test on a dataframe to see the behaviour

df = pd.DataFrame({"x": [1, 2, 3, 4, 5, "ImNotANumber", None, "6"]})
lm = LinearModelRow(2, 4)
df_o = lm(df)
df_o
# %%
# We see that the datafame added the new column 'y' with the calculated values. Another column __error__ also appeared as two rows contained bad data.
# We can examine the errors and eventual fix them or filter them away (Tip: the StripErrors Link is available for removing error rows)
print(df_o.__error__[5])

# %% The Link doesn't need to be added to the framework as such, but can be made in a notebook for a one off use.
# The link_class will then be shown as belonging to __main__
lm.get_params()


# %% Implementing based on Link class may be computationally more efficient, but requires some more work with the error handling
from pdchemchain.base import Link
from typing import Union


@dataclass  # Must be a dataclass
class LinearModelDf(Link):
    slope: float
    bias: float
    in_column: InColumnName = "x"  # InColumnName is a special typehint, that will ensure that the dataframe is checked for existence of the column name before processing and giving adequate error information
    out_column: str = "y"

    def calculate_y(self, x: float) -> Union[float, str]:
        try:
            return self.slope * float(x) + self.bias
        except:
            return float("nan")  # We return NaN in case of an exception

    def _apply(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:  # With the Link base class we overload the _apply function. We get the dataframe directly for full control
        df = df.copy()  # In general we do not want to mutate the input dataframe
        df[self.out_column] = df[self.in_column].apply(
            self.calculate_y
        )  # We can apply on the whole dataframe

        # We have to handle updating of the error column manually
        error_mask = df[self.out_column].isna()
        if sum(error_mask) > 0:  # We had errors and must handle them
            errors = pd.Series(
                [float("nan")] * len(df)
            )  # Creation a NaN column the length of the dataframe
            errors[error_mask] = (
                f"Error in {self.in_column} row data"  # Adding custom error message to the column at places with errors
            )
            self.append_errors(
                df, errors
            )  # The append_errors function is convenient to create or update the __error__ column.

        return df


df = pd.DataFrame({"x": [1, 2, 3, 4, 5, "ImNotANumber", None, "6"]})
lmdf = LinearModelDf(2, 4)
df_o = lmdf(df)
df_o

# So getting the full control of the dataframe comes with a price in manual error handling. We also loose the ability to get the actual exception per row, unless we make the calculate_y function return both the value and a possible exception message.

# %% The two links was quite simple, and for these simple aritmetic cases, the RowEval or DfEval Links can easily be configured and used instead.
from pdchemchain.links.dataframe import RowEval

df = pd.DataFrame({"x": [1, 2, 3, 4, 5, "ImNotANumber", None, "6"]})
roweval = RowEval(
    "2 * row.x + 4", "y"
)  # With RowEval we have to specify row.column_name to get the content. We can't assign directly to a column in the eval string.
df_o = roweval(df)
df_o

# %% It works, except for the string that could have been cast as a float, but the dataframe eval method does not support the function float.

# DfEval works a bit differently, but is faster
from pdchemchain.links.dataframe import DfEval

df = pd.DataFrame(
    {"x": [1, 2, 3, 4, 5]}
)  # DfEval will unfortunately fail completely if any row is faulty
dfeval = DfEval(
    "y = 2 * x + 4"
)  # We do not need to specify the output column seperately with DfEval
df_o = dfeval(df)
df_o


# %% Integration into framework. The link class is added to any of the modules in the links directory. If a new module is created if the existing categories do not fit, be sure to add an import * statement in __init__.py for that file.

# Writing a pytest of the new Link Class
# If you are happy with the custom link and want to add it to the framework, it will be a good idea also to add a test. The project uses pytest.
# Pytests are found in the tests directory, and indivual tests are placed in a subdirectory under the links directory with a name corresponding to the module.

# Say we placed the LinearModelRow class in custom.py in the links directory we would make the file tests/links/custom/test_linearmodelrow.py and add the test module there

# In order to reuse code with as little code as possible some base classes with standard test and fixtures are provided, and customized for the specific link to be tested via
# class attributes


# from test_linearmodelrow.py
from pdchemchain.links.custom import LinearModelRow

# BaseTest and BaseErrorTest have tests for the expected Link behaviour. BaseErrorTest also has test for __error__ creation
from ...basetest import BaseErrorTest

import pandas as pd


class TestLinerModelRow(BaseErrorTest):
    # Three Attributes used by BaseTest and BaseErrorTest
    _Link = LinearModelRow  # This class attribute is used by the BaseTest pytest fixtures to create the link object to be tested

    # Dictionary with parameters for instantiation of link.
    # BaseTests use the sample_dataframe fixture from conftest.py in the tests directory, columnnames should match that
    _classparams = {"slope": 1, "bias": 2, "in_column": "int1", "out_column": "y"}

    # Alternative parameters that are NOT the same as the ones in _classparams
    _alt_classparams = {"slope": 2, "bias": 1, "in_column": "int2", "out_column": "y2"}

    # The following two are specific for BaseErrorTest
    _error_dataframe = pd.DataFrame(
        {"int1": [1, 2, 3, None, "ImNotANumber", "BAD"]}
    )  # A dataframe for testing, will be available for testfunctions as the fixture 'error_dataframe'
    _expected_errors = [
        False,
        False,
        False,
        True,
        True,
        True,
    ]  # The expected errors, we expect None and the strings not to work and __error__ will be filled for these rows.

    # Specific tests must start with 'test' for pytest to find them
    def test_calculation(
        self, link, sample_dataframe
    ):  # The fixtures with the link (from BaseTest) and sampledateframe (from conftest.py) are created and provided
        df_o = link(sample_dataframe)
        ys = list(
            df_o.y.values
        )  # We cast y column content as a list for the assert to work
        assert (
            ys == [3.0, 4.0, 5.0]
        )  # We check that [1,2,3] turns into [3.,4.,5.] with a slope 1 and bias 2 from the _class_params above


# A limitation of the class attribute approach to customization is that the parameters for the link instantiation can't depend on fixtures.
# In the case of that scenario to be relevant the fixtures of the baseclass themselves can be overloaded. An example is the tests/links/io/test_fromfile.py test, where a
# filepath to the temporary test csv file created by the fixture was needed for the configuration of the parameters for the link
