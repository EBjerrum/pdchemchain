import copy
from dataclasses import fields
import pytest

from pdchemchain import Link
from pdchemchain.errormanager import rows_with_errors
from pdchemchain.links.dataframe import NullLink
from pdchemchain import Chain
import pandas as pd
from pandas.testing import assert_frame_equal

from rdkit import Chem


class BaseTest:
    # May be a bit non-pytest-like, but using these class attributes to create the fixtures in the baseclass
    # saves a lot of code lines in the individual tests. If customization is needed where fixtures are needed (e.g. to put a tmpdir fixture in the clasparams)
    # the fixtures can be overloaded in the subclass

    _Link = NullLink
    _classparams = {"name": "link1"}
    _alt_classparams = {"name": "link2"}

    @pytest.fixture
    def classparams(self):
        return copy.deepcopy(self._classparams)

    @pytest.fixture
    def alt_classparams(self):
        return copy.deepcopy(self._alt_classparams)

    @pytest.fixture
    def LinkClass(self):
        return self._Link

    @pytest.fixture
    def link(self, LinkClass, classparams) -> Link:
        return LinkClass(**classparams)

    def test_return_dataframe(self, link, sample_dataframe):
        result = link(sample_dataframe)
        assert isinstance(result, pd.DataFrame)

    def test_add_operator(self, link):
        test_list = [
            (link, link),
            (link, Chain(links=[link])),
            (Chain(links=[link]), link),
            (Chain(links=[link]), (Chain(links=[link]))),
        ]  # TODO Consider making a fixture

        for link1, link2 in test_list:
            o_out = link1 + link2
            assert isinstance(o_out, Chain)

    def test_add_operator_order(self, LinkClass, classparams):
        link1 = LinkClass(**classparams)
        link2 = LinkClass(**classparams)

        chain = link1 + link2
        assert chain.links[0] is link1
        assert chain.links[0] is not link2

        assert chain.links[1] is not link1
        assert chain.links[1] is link2

    def test_sum_operator(self, LinkClass, classparams):
        link1 = LinkClass(**classparams)
        link2 = LinkClass(**classparams)
        link3 = LinkClass(**classparams)

        chain = sum([link1, link2, link3])
        assert isinstance(chain, Chain)
        assert chain.links[0] is link1
        assert chain.links[1] is link2
        assert chain.links[2] is link3

    def test_get_params(self, link):
        params = link.get_params()
        assert isinstance(params, dict)

        parameters_as_tuples = [(f.name, getattr(link, f.name)) for f in fields(link)]
        for name, value in parameters_as_tuples:
            sub_parameters = params[name]
            if isinstance(sub_parameters, dict) and "__class__" in sub_parameters:
                value2 = Link.from_params(
                    sub_parameters
                )  # This will only work for a single nesting
                assert value == value2
            elif isinstance(
                sub_parameters, (list, tuple)
            ):  # A list that may contain Link objects
                for i, sub_value in enumerate(value):
                    if isinstance(sub_value, Link):
                        value[i] = (
                            sub_value.get_params()
                        )  # TODO, here we compare dict to dict, in the previous it was object to object
                value == params[name]
            else:
                assert value == params[name]

    def test_cloning(self, link, LinkClass):
        params = link.get_params()
        link2 = LinkClass.from_params(params)

        parameters_as_tuples = [(f.name, getattr(link, f.name)) for f in fields(link)]

        for name, value in parameters_as_tuples:
            assert (
                getattr(link, name) == getattr(link2, name)
            )  # We are evt. comparing objects, but if NullLinks have the same name they are considered equal in bool context, so it works

    def test_cloning_negative(self, link, LinkClass, alt_classparams):
        params = link.get_params()

        for key, value in alt_classparams.items():
            params[key] = value

        link2 = LinkClass.from_params(params)

        assertations = []
        for name, value in alt_classparams.items():
            assertations.append(getattr(link, name) != getattr(link2, name))

        assert all(assertations)

    def test_input_not_mutated(self, link, sample_dataframe):
        df1 = sample_dataframe
        df2 = sample_dataframe.copy(deep=True)

        df_o = link(df1)
        assert_frame_equal(df1, df2)


class BaseErrorTest(BaseTest):
    _Link = NullLink
    _classparams = {"name": "link1"}
    _alt_classparams = {"name": "link2"}
    _error_dataframe = pd.DataFrame(
        {
            "int1": [1, 2, 3, 4, 5, 6],
            "int2": [2, 3, 4, 5, 6, 7],
            "letters1": ["a", "b", "c", "d", "e", "f"],
            "mixed_types": ["a", "b", "c", 1, None, 1.0],
            "Smiles": ["c1ccccc1", "c1ccccc1CN", "c1ccccc1.C", "ImNotASmiles", 1, None],
            "ROMol": [
                Chem.MolFromSmiles("c1ccccc1"),
                Chem.MolFromSmiles("c1ccccc1CN"),
                Chem.MolFromSmiles("c1ccccc1.C"),
                "a",
                1,
                None,
            ],
        }
    )
    _expected_errors = [False, False, False, True, True, True]

    @pytest.fixture
    def error_dataframe(self):
        return self._error_dataframe.copy(deep=True)

    @pytest.fixture
    def expected_errors(self):
        return copy.deepcopy(self._expected_errors)

    def test_error_creation(self, link, error_dataframe, expected_errors):
        df_o = link(error_dataframe)
        errors = rows_with_errors(df_o, aslist=True)
        assert errors == expected_errors
