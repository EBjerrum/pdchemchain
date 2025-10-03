"""
Links that depend on PySMILESUtils library

Requires pysmilesutils to be installed:
pip install git+ssh://git@github.com/EBjerrum/pysmilesutils.git
"""

from dataclasses import dataclass
import pandas as pd

from pdchemchain.base import RowLink
from pdchemchain.typing import InColumnName


@dataclass
class NumberOfTokens(RowLink):
    """Counts the number of tokens in the SMILES string

    Uses the PySMILESUtils to tokenize and counts the numbers of tokens.
    PySMILESUtils must be installed for this Link to function.

    Parameters
    ----------
    in_column
        The label for the column containing the molecules to analyze
    out_column
        The label for the column that should store the token count

    Raises
    ------
    ImportError
        Raised if PySMILESUtils is not installed, installation instructions are provided.
    """

    in_column: InColumnName = "Smiles"
    out_column: str = "NumTokens"

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer = self._import_dependency()

    def _import_dependency(self):
        try:
            from pysmilesutils.tokenize import SMILESAtomTokenizer as tokenizer

            return tokenizer(warn=False)
        except ImportError as e:
            raise ImportError(
                "The 'pysmilesutils' module is required for this class. "
                "Please install it using: pip install git+ssh://git@github.com/EBjerrum/pysmilesutils.git"
            ) from e

    def _row_apply(self, row: pd.Series) -> pd.Series:
        """Number of tokens from the smiles string, without start and end tokens"""
        smiles = row[self.in_column]
        tokens = self.tokenizer.tokenize([smiles], enclose=False)[0]
        row[self.out_column] = len(tokens)

        return row