import tempfile
from pdchemchain.links import ToSDF, Link

from ...basetest import BaseTest
import pytest

from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd

import os, copy


class TestToSDF(BaseTest):
    _Link = ToSDF

    @pytest.fixture
    def save_filename(self, tmpdir):
        return str(tmpdir) + "/save_test1.sdf"

    @pytest.fixture
    def classparams(self, save_filename):
        return {"filename": save_filename, "mol_column": "ROMol"}

    @pytest.fixture
    def alt_classparams(self, tmpdir):
        return {"filename": str(tmpdir) + "/save_test2.sdf", "mol_column": "Molecule"}

    def test_saving_file(self, link, sample_dataframe, save_filename):
        """Test that SDF file is created"""
        df = link(sample_dataframe)
        assert os.path.exists(save_filename)

    def test_roundtrip(self, link, sample_dataframe, save_filename):
        """Test that data can be written and read back"""
        # Write the file
        link(sample_dataframe)

        # Read it back
        df_loaded = PandasTools.LoadSDF(save_filename, molColName="ROMol")

        # Check that we have the same number of rows
        assert len(df_loaded) == len(sample_dataframe)

        # Check that molecules are present
        assert all(df_loaded["ROMol"].apply(lambda x: x is not None))

        # Check that properties were saved
        assert "int1" in df_loaded.columns
        assert "letters1" in df_loaded.columns

    def test_dunder_columns_excluded(self, sample_dataframe, save_filename):
        """Test that dunder columns (__error__, __log__) are not written to SDF"""
        # Add error columns
        df_with_errors = sample_dataframe.copy()
        df_with_errors["__error__"] = [None, "Some error", None]
        df_with_errors["__log__"] = ["log1", "log2", "log3"]

        # Write to SDF
        link = ToSDF(filename=save_filename)
        link(df_with_errors)

        # Read back and check dunder columns are not present
        df_loaded = PandasTools.LoadSDF(save_filename, molColName="ROMol")
        assert "__error__" not in df_loaded.columns
        assert "__log__" not in df_loaded.columns

    def test_selective_properties(self, sample_dataframe, save_filename):
        """Test writing only specific properties"""
        link = ToSDF(filename=save_filename, properties=["int1", "Smiles"])
        link(sample_dataframe)

        # Read back
        df_loaded = PandasTools.LoadSDF(save_filename, molColName="ROMol")

        # Check only specified properties are present
        assert "int1" in df_loaded.columns
        assert "Smiles" in df_loaded.columns
        # int2 and letters1 should not be present
        assert "int2" not in df_loaded.columns
        assert "letters1" not in df_loaded.columns
