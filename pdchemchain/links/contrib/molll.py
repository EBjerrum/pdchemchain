"""
Links that depend on MolLL library

Requires molecule_ll to be installed:
pip install molecule_ll
"""

from dataclasses import dataclass
import pandas as pd
import json
import os

from pdchemchain.base import RowLink
from pdchemchain.typing import InColumnName


@dataclass
class MolLL(RowLink):
    """Calculates molecular log-likelihood using MolLL or AtomLL models

    Uses the MolLL library to calculate log-likelihood scores for molecules.
    The molecule_ll package must be installed for this Link to function.

    Parameters
    ----------
    in_column
        The label for the column containing RDKit Mol objects
    out_column
        The label for the column that should store the log-likelihood scores
    model
        Model specification. Can be:
        - Pre-trained model name (e.g., 'LibInventMolLLr2', 'LibInventAtomLLr1')
        - Path to a saved JSON model file
        The model type (MolLL vs AtomLL) is auto-detected from save files.

    Raises
    ------
    ImportError
        Raised if molecule_ll is not installed, installation instructions are provided.
    ValueError
        Raised if the model parameter cannot be resolved to a valid model.
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "molll"
    model: str = "LibInventMolLLr2"

    # Known pre-trained models mapping
    _known_models = {
        'LibInventMolLLr1': 'LibInventMolLLr1',
        'LibInventMolLLr2': 'LibInventMolLLr2',
        'LibInventMolLLr3': 'LibInventMolLLr3',
        'LibInventAtomLLr1': 'LibInventAtomLLr1',
        'LibInventAtomLLr2': 'LibInventAtomLLr2',
        'LibInventAtomLLr3': 'LibInventAtomLLr3',
    }

    @classmethod
    def available_models(cls):
        """Return a list of available pre-trained models"""
        return list(cls._known_models.keys())

    def __post_init__(self):
        super().__post_init__()
        self.molll_model = self._load_model()

    def _import_dependencies(self):
        """Import required molll components with helpful error message"""
        try:
            import molll
            return molll
        except ImportError as e:
            raise ImportError(
                "The 'molecule_ll' package is required for this class. "
                "Please install it using: pip install molecule_ll"
            ) from e

    def _load_model(self):
        """Load the specified model, either pre-trained or from file"""
        molll = self._import_dependencies()
        
        # Check if it's a known pre-trained model
        if self.model in self._known_models:
            model_class = getattr(molll, self._known_models[self.model])
            return model_class()
        
        # Otherwise, treat as a file path
        if not os.path.exists(self.model):
            raise ValueError(
                f"Model '{self.model}' is not a known pre-trained model and "
                f"file does not exist. Known models: {self.available_models()}"
            )
        
        # Load from file - auto-detect model type
        try:
            with open(self.model, 'r') as f:
                save_data = json.load(f)
            
            model_type = save_data.get("Model")
            if model_type == "MolLL":
                model_instance = molll.MolLL()
            elif model_type == "AtomLL":
                model_instance = molll.AtomLL()
            else:
                raise ValueError(
                    f"Unknown model type '{model_type}' in save file. "
                    "Expected 'MolLL' or 'AtomLL'"
                )
            
            model_instance.load(self.model)
            return model_instance
            
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(
                f"Invalid model file '{self.model}': {e}"
            ) from e

    def _row_apply(self, row: pd.Series) -> pd.Series:
        """Calculate log-likelihood for the molecule"""
        mol = row[self.in_column]
        
        # Calculate log-likelihood score
        ll_score = self.molll_model.calculate_ll(mol)
        row[self.out_column] = ll_score
        
        return row