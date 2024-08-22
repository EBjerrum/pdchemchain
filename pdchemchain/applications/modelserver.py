from dataclasses import dataclass
from pdchemchain.base import SelfConfigurable
from pdchemchain.links.model import ScikitMolMolModel, ScikitMolSmilesModel

from typing import Literal
import pandas as pd


@dataclass
class ScikitMolModelServer(SelfConfigurable):
    model_file: str
    model_type: Literal["smiles", "mol"] = "smiles"

    def __post_init__(self):
        super().__post_init__()
        # In principle this could be inferred by inspection of the scikit-mol model, but better explicit than implicit
        if self.model_type == "smiles":
            self._model = ScikitMolSmilesModel(
                model_file=self.model_file, in_column="smiles_or_mol"
            )
        else:
            self._model = ScikitMolMolModel(
                model_file=self.model_file, in_column="smiles_or_mol"
            )

    def predict(self, smiles_or_mol_list):
        df = pd.DataFrame({"smiles_or_mol": smiles_or_mol_list})
        df_out = self._model(df)
        return df_out["skmol_prediction"].values
