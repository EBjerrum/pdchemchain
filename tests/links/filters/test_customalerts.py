from pdchemchain.links import CustomAlerts
from ...basetest import BaseErrorTest
import pandas as pd
from rdkit import Chem


class TestCustomAlerts(BaseErrorTest):
    _Link = CustomAlerts
    _classparams = {"in_column": "ROMol", "out_column": "custom_alerts_score"}
    _alt_classparams = {"in_column": "ROMol2", "out_column": "custom_alerts_score2"}

    def test_filter_calculation(self, link, sample_dataframe):
        """Test that custom alerts filter produces scores"""
        df_o = link(sample_dataframe)
        assert "custom_alerts_score" in df_o.columns
        
        # Check that all scores are either 0.0 or 1.0
        scores = df_o["custom_alerts_score"].dropna()
        assert all(score in [0.0, 1.0] for score in scores)

    def test_clean_molecules_pass(self, link):
        """Test that clean molecules pass the filter (score = 1.0)"""
        # Simple, clean molecules that should pass all filters
        clean_smiles = ["CCO", "c1ccccc1", "CC(C)O"]  # ethanol, benzene, isopropanol
        clean_mols = [Chem.MolFromSmiles(smi) for smi in clean_smiles]
        test_df = pd.DataFrame({"ROMol": clean_mols, "SMILES": clean_smiles})
        
        result_df = link(test_df)
        
        # All clean molecules should get score 1.0
        scores = result_df["custom_alerts_score"].tolist()
        assert all(score == 1.0 for score in scores), f"Expected all 1.0, got {scores}"

    def test_problematic_molecules_fail(self, link):
        """Test that molecules with alerts fail the filter (score = 0.0)"""
        # Molecules that should trigger alerts
        problematic_smiles = [
            "CC#CC",  # Contains alkyne (C#C)
            "COOC",   # Contains peroxide ([#8][#8])
            "C[C+]C", # Contains charged carbon ([#6;+])
        ]
        problematic_mols = [Chem.MolFromSmiles(smi) for smi in problematic_smiles if Chem.MolFromSmiles(smi)]
        
        if problematic_mols:  # Only test if we have valid molecules
            test_df = pd.DataFrame({"ROMol": problematic_mols})
            result_df = link(test_df)
            
            # All problematic molecules should get score 0.0
            scores = result_df["custom_alerts_score"].tolist()
            assert all(score == 0.0 for score in scores), f"Expected all 0.0, got {scores}"

    def test_custom_smarts_list(self):
        """Test that custom SMARTS lists work correctly"""
        # Create filter with only alkyne alert
        custom_filter = CustomAlerts(smarts_list=["C#C"])
        
        # Test molecules
        alkyne_mol = Chem.MolFromSmiles("CC#CC")  # Should fail
        alkane_mol = Chem.MolFromSmiles("CCCC")   # Should pass
        
        test_df = pd.DataFrame({"ROMol": [alkyne_mol, alkane_mol]})
        result_df = custom_filter(test_df)
        
        scores = result_df["custom_alerts_score"].tolist()
        assert scores[0] == 0.0  # Alkyne should fail
        assert scores[1] == 1.0  # Alkane should pass

    def test_invalid_smarts_handling(self):
        """Test that invalid SMARTS patterns are handled gracefully"""
        # Create filter with invalid SMARTS pattern
        invalid_filter = CustomAlerts(smarts_list=["[invalid_smarts]", "CCO"])
        
        # Should still work with valid molecules
        test_mol = Chem.MolFromSmiles("CCO")
        test_df = pd.DataFrame({"ROMol": [test_mol]})
        
        # Should not raise exception
        result_df = invalid_filter(test_df)
        assert "custom_alerts_score" in result_df.columns