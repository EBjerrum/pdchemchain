import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from pdchemchain.base import RowLink
from pdchemchain.typing import InColumnName
from pdchemchain.utilities import assert_import_dependency


@dataclass
class ScikitMolBase(RowLink, ABC):
    model_file: str
    out_column: str = "skmol_prediction"

    @property
    @abstractmethod
    def in_column(self):
        pass

    def __post_init__(self):
        super().__post_init__()
        assert_import_dependency("scikit_mol")
        self.model = pickle.load(open(self.model_file, "rb"))

    # Applying row-wise trade efficiency for error handling
    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol_or_smiles = row[self.in_column]
        row[self.out_column] = self.model.predict([mol_or_smiles])[
            0
        ]  # TODO, predict_proba?
        return row


@dataclass
class ScikitMolMolModel(ScikitMolBase):
    in_column: InColumnName = "ROMol"


@dataclass
class ScikitMolSmilesModel(ScikitMolBase):
    in_column: InColumnName = "Smiles"
