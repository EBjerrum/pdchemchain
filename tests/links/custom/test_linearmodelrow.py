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

    # Alternative parameters that are not the same as the ones in _classparams
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
        ys = list(df_o.y.values)  # We cast as a list for the assert to work
        assert (
            ys == [3.0, 4.0, 5.0]
        )  # We check that [1,2,3] turns into [3.,4.,5.] with a slope 1 and bias 2 from the _class_params above
