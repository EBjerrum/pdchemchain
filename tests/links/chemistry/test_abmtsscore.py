from pdchemchain.links import ABMPSScore
from ...basetest import BaseErrorTest
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


class TestABMTSScore(BaseErrorTest):
    _Link = ABMPSScore
    _classparams = {"in_column": "ROMol", "out_column": "AB_MTS_Score"}
    _alt_classparams = {"in_column": "ROMol2", "out_column": "AB_MTS_Score2"}

    def test_abmts_calculation(self, link, sample_dataframe):
        """Test that AB-MTS score is calculated correctly"""
        df_o = link(sample_dataframe)
        assert "AB_MTS_Score" in df_o.columns
        
        # Check that values are numeric (float or int)
        assert all(pd.api.types.is_numeric_dtype(type(val)) for val in df_o["AB_MTS_Score"] if pd.notna(val))

    def test_abmts_manual_verification(self, link):
        """Test AB-MTS calculation against manual calculation for known molecules"""
        # Create test dataframe with known molecules
        test_smiles = ["CCO", "c1ccccc1", "CC(C)O"]  # ethanol, benzene, isopropanol
        test_mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]
        test_df = pd.DataFrame({"ROMol": test_mols, "SMILES": test_smiles})
        
        result_df = link(test_df)
        
        # Manually verify calculation for ethanol (CCO)
        ethanol_mol = Chem.MolFromSmiles("CCO")
        clogp = Descriptors.MolLogP(ethanol_mol)
        nar = Descriptors.NumAromaticRings(ethanol_mol)  
        nrb = Descriptors.NumRotatableBonds(ethanol_mol)
        expected_score = abs(clogp - 3) + nar + nrb
        
        calculated_score = result_df.loc[0, "AB_MTS_Score"]
        assert abs(calculated_score - expected_score) < 1e-6, f"Expected {expected_score}, got {calculated_score}"

    def test_abmts_score_components(self, link):
        """Test that AB-MTS score components are calculated correctly"""
        # Test with a more complex molecule
        test_mol = Chem.MolFromSmiles("CCN(CC)c1ccc(cc1)C(=O)O")  # para-substituted benzoic acid derivative
        test_df = pd.DataFrame({"ROMol": [test_mol]})
        
        result_df = link(test_df)
        score = result_df.loc[0, "AB_MTS_Score"]
        
        # Score should be positive (all components are >= 0)
        assert score >= 0
        
        # For this molecule, we expect:
        # - Some aromatic rings (NAR > 0)
        # - Some rotatable bonds (NRB > 0)  
        # - LogP contribution |LogP - 3|
        assert score > 0  # Should be greater than 0 for this complex molecule