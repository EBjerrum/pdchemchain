import tempfile
from pdchemchain.links import FromSDF

from ...basetest import BaseTest

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

import os, pytest


class TestFromSDF(BaseTest):
    _Link = FromSDF

    @pytest.fixture
    def sdf_filename(self, tmpdir):
        # Create a sample dataframe with molecules and properties
        df = pd.DataFrame(
            {
                "ROMol": [
                    Chem.MolFromSmiles("C"),
                    Chem.MolFromSmiles("CN"),
                    Chem.MolFromSmiles("OCN"),
                ],
                "int1": [1, 2, 1],
                "int2": [2, 3, 4],
                "letters1": ["a", "b", "c"],
                "Smiles": ["C", "CN", "OCN"],
            }
        )
        filename = str(tmpdir) + "/load_test1.sdf"
        PandasTools.WriteSDF(df, filename, molColName="ROMol", properties=list(df.columns))

        return filename

    @pytest.fixture
    def classparams(self, sdf_filename):
        return {"filename": sdf_filename, "mol_column": "ROMol"}

    @pytest.fixture
    def alt_classparams(self, tmpdir):
        return {
            "filename": str(tmpdir)
            + "/load_test2.sdf",  # This is never attempted loaded, so no need for it to exist
            "mol_column": "Molecule",
        }

    def test_loading_file(self, link, sdf_filename):
        """Test that the SDF file loads and contains expected data"""
        df = link()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "ROMol" in df.columns
        # Check that molecules are loaded correctly
        assert all(df["ROMol"].apply(lambda x: x is not None))

    def test_properties_loaded(self, link):
        """Test that SD file properties are loaded as columns"""
        df = link()
        assert "int1" in df.columns
        assert "int2" in df.columns
        assert "letters1" in df.columns
        # SDF properties are loaded as strings
        assert list(df["int1"]) == ["1", "2", "1"]
        assert list(df["letters1"]) == ["a", "b", "c"]
