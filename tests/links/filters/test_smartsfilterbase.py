from pdchemchain.links.filters import SMARTSFilterBase
from ...basetest import BaseErrorTest
import pandas as pd
from rdkit import Chem
from dataclasses import dataclass
from typing import List
import pytest


@dataclass 
class TestSMARTSFilter(SMARTSFilterBase):
    """Test implementation of SMARTSFilterBase for testing"""
    out_column: str = "test_filter_score"
    
    def get_smarts_list(self) -> List[str]:
        # Simple test pattern - aromatic carbons
        return ["c"]


class TestSMARTSFilterBase(BaseErrorTest):
    _Link = TestSMARTSFilter
    _classparams = {"in_column": "ROMol", "out_column": "test_filter_score"}
    _alt_classparams = {"in_column": "ROMol2", "out_column": "test_filter_score2"}

    def test_abstract_base_class_cannot_be_instantiated(self):
        """Test that the abstract base class cannot be instantiated directly"""
        with pytest.raises(TypeError):
            SMARTSFilterBase()

    def test_filter_scoring_logic(self, link):
        """Test that the base filter logic works correctly"""
        # Test molecules: benzene (aromatic) vs methane (aliphatic)
        benzene = Chem.MolFromSmiles("c1ccccc1")  # Should fail (score = 0.0)
        methane = Chem.MolFromSmiles("C")         # Should pass (score = 1.0)
        
        test_df = pd.DataFrame({"ROMol": [benzene, methane]})
        result_df = link(test_df)
        
        scores = result_df["test_filter_score"].tolist()
        assert scores[0] == 0.0  # Benzene should fail (has aromatic carbon)
        assert scores[1] == 1.0  # Methane should pass (no aromatic carbon)

    def test_empty_smarts_list(self):
        """Test behavior with empty SMARTS list"""
        @dataclass
        class EmptySMARTSFilter(SMARTSFilterBase):
            out_column: str = "empty_filter_score"
            
            def get_smarts_list(self) -> List[str]:
                return []
        
        empty_filter = EmptySMARTSFilter()
        test_mol = Chem.MolFromSmiles("c1ccccc1")
        test_df = pd.DataFrame({"ROMol": [test_mol]})
        
        result_df = empty_filter(test_df)
        # With no patterns, all molecules should pass (score = 1.0)
        assert result_df["empty_filter_score"].iloc[0] == 1.0