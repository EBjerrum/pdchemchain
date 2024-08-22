from __future__ import annotations

import copy
import importlib.metadata
import inspect
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pydoc import locate
from typing import List, Tuple, Union

import pandas as pd

from pdchemchain.errormanager import has_error
from pdchemchain.io_utilities import load_chain, save_chain
from pdchemchain.logging import logger, logging
from pdchemchain.typing import InColumnName


@dataclass
class SelfConfigurable:
    # Sets both configuration on logging, and on objects instantiated logger. Synchronized to avoid confusion, but local changes in level are then not doable.
    def set_log_level(self, level_str: str = "debug"):
        level = logging.getLevelName(level_str.upper())
        logging.basicConfig(level=level)
        self.logger.setLevel(level=level)

    def __post_init__(self):
        self.logger = logging.getLogger(
            f"{type(self).__module__}.{type(self).__name__}"
        )

    @property
    def tooltip(self):
        return self.__doc__.split("\n")[0]

    # Inspired from https://github.com/scikit-learn/scikit-learn/blob/714c50092/sklearn/base.py#L137
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "pdChemChain links should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    # Inspired from https://github.com/scikit-learn/scikit-learn/blob/714c50092/sklearn/base.py#L178
    def get_params(self, defaults=True, version=True, log_level=True):
        """
        Get parameters for this link

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this link and
            contained subobjects that are links

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()

        if version:
            out["__version__"] = this_version()
            version = False

        if log_level:
            out["__loglevel__"] = logging.getLevelName(self.logger.getEffectiveLevel())
            log_level = False

        out["__class__"] = f"{type(self).__module__}.{type(self).__name__}"

        for key in self._get_param_names():
            value = getattr(self, key)
            # Check if it's the default value, and don't include
            if self.__dataclass_fields__[key].default == value and not defaults:
                continue

            if isinstance(value, SelfConfigurable):
                out[key] = value.get_params(
                    defaults=defaults, version=version, log_level=log_level
                )
            elif isinstance(value, (list, tuple)):
                deep_values = []
                for deep_value in value:
                    if hasattr(deep_value, "get_params"):
                        deep_values.append(
                            deep_value.get_params(
                                defaults=defaults, version=version, log_level=log_level
                            )
                        )
                    else:
                        deep_values.append(deep_value)
                out[key] = type(value)(deep_values)
            else:
                out[key] = value

        return out

    @classmethod
    def from_params(self, params) -> Link:
        # Get the class reference from the string
        params_copy = copy.deepcopy(params)

        if "__version__" in params_copy:
            config_version = params_copy.pop("__version__")
            if config_version != this_version():
                logger.warning(
                    f"version in config {config_version} doesn not equal the installed version {this_version()}. Unexpected behaviour may ensue."
                )

        if "__loglevel__" in params_copy:
            log_level_str = params_copy.pop("__loglevel__")
        else:
            log_level_str = "undefined"

        classpath = params_copy.pop("__class__")
        klass = locate(classpath)
        # Alternative, but requires splitting of 'path' and 'name'
        # getattr(sys.modules["__main__"], "NullLink")(name='tester')

        # Recursive instantiation of nested links
        recursive_params = {}
        for key, value in params_copy.items():
            if (
                isinstance(value, dict) and "__class__" in value
            ):  # For directly nested Link
                recursive_params[key] = Link.from_params(value)
            elif isinstance(value, (list, tuple)):  # for evt. Links in a list
                recursive_list = []
                for item in value:
                    if "__class__" in item:
                        recursive_list.append(Link.from_params(item))
                    else:
                        recursive_list.append(item)
                recursive_params[key] = recursive_list
            else:
                recursive_params[key] = value

        obj = klass(**recursive_params)
        if log_level_str != "undefined":
            obj.set_log_level(log_level_str)

        return obj

    def to_config_file(self, filename, defaults=True, version=True, log_level=True):
        save_chain(
            self, filename, defaults=defaults, version=version, log_level=log_level
        )

    @classmethod
    def from_config_file(self, filename) -> Link:
        link_or_chain = load_chain(filename)
        return link_or_chain


@dataclass
class Link(ABC, SelfConfigurable):
    """Base class for all Links

    All links must be @dataclasses and overload the abstract method _apply(self, df: pd.DataFrame) -> pd.DataFrame
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Starting processing of dataframe with {len(df)} rows")
        self.assert_incolumns(df)
        # TODO, figure our a way to skip calculations on rows that contains errors.
        return self._apply(df)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.apply(df)

    @abstractmethod
    def _apply(
        self, df: pd.DataFrame
    ) -> (
        pd.DataFrame
    ):  # TODO Calling .apply directly will circumvvent the incolumn assertation.
        """Method that does something to the dataframe"""

    def __add__(self, other):
        if not other:
            return self  # This should take care of seeding with 0 needed for sum()
        if isinstance(other, Chain):
            return Chain([self] + other.links)
        elif isinstance(other, Link):
            return Chain([self, other])
        else:
            raise TypeError("Unsupported type for addition")

    __radd__ = __add__

    # Error handling for df level
    def _concatenate_strings(self, str1, str2) -> str:
        """Combine two variables that may be strings or None/float('nan), in a safe way"""
        if pd.notna(str1) and pd.notna(str2):
            return (
                f"{str1}\n{str2}"
                if isinstance(str1, str) and isinstance(str2, str)
                else None
            )
        return str1 if pd.notna(str1) else str2

    def append_errors(self, df: pd.DataFrame, errors: pd.Series) -> pd.DataFrame:
        """Append errors to existing error column, or create a new one."""
        if "__error__" not in df:
            df["__error__"] = errors
        else:
            df["__error__"] = df["__error__"].combine(errors, self._concatenate_strings)
        return df

    def assert_incolumns(self, dataframe: pd.DataFrame):
        """Assert that the dataframe contains the fields that are typehinted as InColumnName in the dataclass definition"""

        in_column_properties = [
            field.name for field in fields(self) if field.type == InColumnName
        ]
        expected_columns = [getattr(self, prop) for prop in in_column_properties]
        if expected_columns:
            self.logger.debug(f"Asserting expected input columns: {expected_columns}")
        assert set(
            expected_columns
        ).issubset(
            set(dataframe.columns)
        ), f"DataFrame is missing expected input columns (in_columns): {set(expected_columns).difference(set(dataframe.columns))}"


class RowLink(Link):
    """Base class for all links that process dataframes row-by-row

    All subclasses must be @dataclasses and overload the _row_apply(self, row: pd.Series) -> pd.Series method

    rows are pd.Series and values can be indexed via the column name, e.g. row[self.in_column]

    new values for new columns can be directly set like example row[self.out_column] = new_value
    """

    def _apply(
        self, df
    ):  # TODO, the class hiearachy probably need to be modified, so that RowLink are not inheriting Link, but rather a common BaseLink class
        raise RuntimeError("This function should not be called in the RowLink class")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.assert_incolumns(df)
        self.logger.info(f"Processing dataframe with {len(df)} rows, row by row")
        # This should be reasonable efficient: https://www.learndatasci.com/solutions/how-iterate-over-rows-pandas/
        return df.apply(self._safe_row_apply, axis=1)

    def _create_or_append(
        self, row: pd.Series, data: str, column_name: str = "__error__"
    ) -> pd.Series:
        if (column_name not in row) or (
            not isinstance(row[column_name], str)
        ):  # TODO This will overwrite all that is not str
            row[column_name] = data
        else:
            row[column_name] += f"\n{data}"  # TODO assumes content is str type
        return row

    def _safe_row_apply(self, row: pd.Series) -> pd.Series:
        row = copy.deepcopy(row)

        if not has_error(
            row
        ):  # TODO, consider making it globally configurable if error rows should be skipped or appended. Synchronise with DF wide applys.
            try:
                row = self._row_apply(row)
            except Exception as e:
                return self._create_or_append(
                    row, traceback.format_exc()
                )  # str(e) is less chatty, TODO make it configurable
        return row

    @abstractmethod
    def _row_apply(self, row: pd.Series) -> pd.Series:
        # Do something with the row, e.g. row["mol"] = Chem.MolFromSmiles(row["Smiles"])
        return row


@dataclass
class Chain(Link):
    """Runs links sequentially, one after the other and return the processed dataframe."""

    links: Union[List[Link], Tuple[Link]]

    def set_log_level(self, level_str: str = "debug"):
        level = logging.getLevelName(level_str.upper())
        logging.basicConfig(level=level)
        self.logger.setLevel(level=level)
        for link in self.links:
            link.set_log_level(level_str)

    def __add__(self, other):
        if not other:
            return self
        if isinstance(other, Chain):
            return Chain(self.links + other.links)
        elif isinstance(other, Link):
            return Chain(self.links + [other])
        else:
            raise TypeError("Unsupported type for addition")

    __radd__ = __add__

    def _apply(self, df):
        # _apply method not used by special class Chain
        # TODO This is a code smell that something is not completely right with the class hierachy
        raise RuntimeError("This function should not be called in the Chain class")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the links independent of each other and join"""
        self.logger.debug(f"Starting sequential processing of {len(self.links)} Link")
        for link in self.links:
            df = link(df)
        self.logger.debug("Sequential processing done")
        return df


@dataclass
class UnionLink(Link):
    """Runs the dataframe through two seperate links and merges the result two dataframes into a single one.

    Merges using 'combine_first', which priorities output from internal Link # 1"""

    link1: Link
    link2: Link

    def set_log_level(self, level_str: str = "debug"):
        level = logging.getLevelName(level_str.upper())
        logging.basicConfig(level=level)
        self.logger.setLevel(level=level)
        for link in [self.link1, self.link2]:
            link.set_log_level(level_str)

    def _apply(self, df):
        # _apply method not used by special class Chain
        raise RuntimeError("This function should not be called in the UnionLink class")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the first Link to the original dataframe,
        then apply the second Link to the original dataframe,
        and join the resulting dataframes before returning the join.
        Join is using Pandas 'combine_first', which prioritizes value from the first dataframe, i.e. output of link1
        """
        self.logger.debug("Starting processing of seperate Links or Chains")
        self.logger.debug("Processing of first Link or Chain")
        df1 = self.link1(
            df.copy()
        )  # Apply first link to a copy of the original dataframe
        self.logger.debug("Processing of second Link or Chain")
        df2 = self.link2(
            df.copy()
        )  # Apply second link to a copy of the original dataframe
        self.logger.debug("Done processing, joining dataframes")
        return df1.combine_first(df2)


def this_version():
    return importlib.metadata.version("pdchemchain")
