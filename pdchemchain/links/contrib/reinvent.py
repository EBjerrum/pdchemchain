"""
Links that depend on REInvent

Requires reinvent to be installed: Please see installation instructions on https://github.com/MolecularAI/REINVENT4
"""

from dataclasses import dataclass
from enum import Enum
import pandas as pd

from pdchemchain.base import RowLink
from pdchemchain.typing import InColumnName


class TokenReturnType(str, Enum):
    """Enum for different token return types"""
    COUNT = "count"
    LIST = "list"
    SET = "set"


# @dataclass
# class NumberOfTokens(RowLink):
#     """Counts the number of tokens in the SMILES string
#
#     Uses the REINVENT tokenizer to tokenize and counts the number of tokens.
#     REINVENT must be installed for this Link to function.
#
#     Parameters
#     ----------
#     in_column
#         The label for the column containing the molecules to analyze
#     out_column
#         The label for the column that should store the token count
#
#     Raises
#     ------
#     ImportError
#         Raised if REINVENT is not installed, installation instructions are provided.
#     """
#
#     in_column: InColumnName = "Smiles"
#     out_column: str = "NumTokens"
#
#     def __post_init__(self):
#         super().__post_init__()
#         self.tokenizer = self._import_dependency()
#
#     def _import_dependency(self):
#         try:
#             from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer
#
#             return SMILESTokenizer()
#         except ImportError as e:
#             raise ImportError(
#                 "The 'reinvent' module is required for this class. "
#                 "Please using instructions found on https://github.com/MolecularAI/REINVENT4"
#             ) from e
#
#     def _row_apply(self, row: pd.Series) -> pd.Series:
#         """Number of tokens from the SMILES string, without start and end tokens"""
#         smiles = row[self.in_column]
#         tokens = self.tokenizer.tokenize(smiles, with_begin_and_end=False)
#         row[self.out_column] = len(tokens)
#
#         return row


@dataclass
class REInventTokenizer(RowLink):
    """Extracts tokens from SMILES strings in various formats

    Uses the REINVENT tokenizer to tokenize SMILES and returns tokens in a configurable format.
    REINVENT must be installed for this Link to function.

    Parameters
    ----------
    in_column
        The label for the column containing the molecules to analyze
    out_column
        The label for the column that should store the extracted tokens
    return_type
        The format for token output: "count" (int), "list" (list of tokens), or "set" (set of unique tokens).
        Defaults to "count".

    Raises
    ------
    ImportError
        Raised if REINVENT is not installed, installation instructions are provided.
    ValueError
        Raised if return_type is not one of the valid TokenReturnType values.
    """

    in_column: InColumnName = "Smiles"
    out_column: str = "Tokens"
    return_type: str = TokenReturnType.COUNT.value

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer = self._import_dependency()

    def _import_dependency(self):
        try:
            from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer

            return SMILESTokenizer()
        except ImportError as e:
            raise ImportError(
                "The 'reinvent' module is required for this class. "
                "Please install it using instructions from https://github.com/MolecularAI/REINVENT4"
            ) from e

    def _validate_return_type_value(self, value):
        valid_types = {t.value for t in TokenReturnType}
        if value not in valid_types:
            raise ValueError(
                f"Invalid return_type '{value}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}"
            )

    def __setattr__(self, name, value):
        if name == "return_type":
            self._validate_return_type_value(value)
        super().__setattr__(name, value)

    def _row_apply(self, row: pd.Series) -> pd.Series:
        """Extracts tokens from SMILES string in the specified format"""
        smiles = row[self.in_column]
        tokens = self.tokenizer.tokenize(smiles, with_begin_and_end=False)

        match self.return_type:
            case TokenReturnType.COUNT.value:
                row[self.out_column] = len(tokens)
            case TokenReturnType.LIST.value:
                row[self.out_column] = tokens
            case TokenReturnType.SET.value:
                row[self.out_column] = set(tokens)

        return row