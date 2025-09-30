from dataclasses import dataclass, field
from typing import List
from abc import ABC, abstractmethod

import pandas as pd
from rdkit import Chem

from pdchemchain.base import RowLink
from pdchemchain.typing import InColumnName


@dataclass
class SMARTSFilterBase(RowLink, ABC):
    """Base class for SMARTS-based molecular filters
    
    This abstract base class provides common functionality for filters that use
    SMARTS patterns to identify problematic or undesirable molecular features.
    Subclasses should implement the `get_smarts_list` method to provide the
    specific SMARTS patterns for their filter type.
    
    The filter returns a score between 0 and 1, where:
    - 1.0 = molecule passes the filter (no problematic patterns found)
    - 0.0 = molecule fails the filter (contains problematic patterns)
    
    Parameters
    ----------
    in_column
        The label for the column containing the molecules to analyze
    out_column
        The label for the column that should store the filter score
    """
    
    in_column: InColumnName = "ROMol"
    out_column: str = "filter_score"
    
    def __post_init__(self):
        super().__post_init__()
        # Precompile SMARTS patterns for efficiency
        self.smarts_patterns = []
        for smarts in self.get_smarts_list():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None:
                    self.smarts_patterns.append(pattern)
                else:
                    self.logger.warning(f"Failed to parse SMARTS pattern: {smarts}")
            except Exception as e:
                self.logger.warning(f"Error parsing SMARTS pattern '{smarts}': {e}")
    
    @abstractmethod
    def get_smarts_list(self) -> List[str]:
        """Return the list of SMARTS patterns for this filter
        
        Returns
        -------
        List[str]
            List of SMARTS patterns as strings
        """
        pass
    
    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol = row[self.in_column]
        if isinstance(mol, Chem.Mol):
            # Check if molecule matches any problematic pattern
            has_match = any(
                mol.HasSubstructMatch(pattern) 
                for pattern in self.smarts_patterns
            )
            
            # Return inverted score: 1 - match (1.0 = pass, 0.0 = fail)
            filter_score = 0.0 if has_match else 1.0
            row[self.out_column] = filter_score
        else:
            raise ValueError(f"Seemingly not a Mol object: {mol} of type {type(mol)}")
        return row


@dataclass
class CustomAlerts(SMARTSFilterBase):
    """Custom molecular alerts filter
    
    Filters molecules based on a custom set of SMARTS patterns that identify
    potentially problematic molecular features. The default patterns include
    common structural alerts like large rings, peroxides, charged carbons, etc.
    
    Based on the CustomAlerts filter from Reinvent4.
    
    Parameters
    ----------
    in_column
        The label for the column containing the molecules to analyze
    out_column
        The label for the column that should store the filter score
    smarts_list
        Custom list of SMARTS patterns. If None, uses default alert patterns.
    """
    
    smarts_list: List[str] = field(default_factory=lambda: [
        '[*;r8]',  # 8-membered rings
        '[*;r9]',  # 9-membered rings
        '[*;r10]', # 10-membered rings
        '[*;r11]', # 11-membered rings
        '[*;r12]', # 12-membered rings
        '[*;r13]', # 13-membered rings
        '[*;r14]', # 14-membered rings
        '[*;r15]', # 15-membered rings
        '[*;r16]', # 16-membered rings
        '[*;r17]', # 17-membered rings
        '[#8][#8]', # Peroxides
        '[#6;+]',   # Charged carbons
        '[#16][#16]', # Disulfides
        '[#7;!n][S;!$(S(=O)=O)]', # N-S bonds (not sulfonamides)
        '[#7;!n][#7;!n]', # N-N bonds
        'C#C',      # Alkynes
        'C(=[O,S])[O,S]', # Certain carbonyl patterns
        '[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]', # N-C-S patterns
        '[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]',  # N-C-N patterns
        '[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]',  # N-C-O patterns
        '[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]', # O-C-S patterns
        '[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]',  # O-C-O patterns
        '[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]' # S-C-S patterns
    ])
    out_column: str = "custom_alerts_score"
    
    def get_smarts_list(self) -> List[str]:
        """Return the SMARTS patterns for custom alerts"""
        return self.smarts_list