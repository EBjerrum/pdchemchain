"""Tests for LigPrep link — unit tests only (no Schrodinger required)."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from pdchemchain.links.schrodinger.ligprep import (
    LigPrep,
    _is_mol_column,
    _parse_variant_tag,
    _write_smi_file,
    _write_sdf_file,
    _read_output_sdf,
)


class TestParseVariantTag:
    """Test s_lp_Variant tag parsing."""

    def test_simple_tag(self):
        title, variant = _parse_variant_tag("0-1")
        assert title == "0"
        assert variant == "1"

    def test_tag_with_hyphens_in_title(self):
        title, variant = _parse_variant_tag("my-molecule-1")
        assert title == "my-molecule"
        assert variant == "1"

    def test_tag_without_hyphen(self):
        title, variant = _parse_variant_tag("something")
        assert title == "something"
        assert variant == "0"

    def test_numeric_index(self):
        title, variant = _parse_variant_tag("42-3")
        assert title == "42"
        assert variant == "3"


class TestIsMolColumn:
    """Test RDKit Mol detection in pandas Series."""

    def test_mol_series(self):
        s = pd.Series([Chem.MolFromSmiles("C"), Chem.MolFromSmiles("CC")])
        assert _is_mol_column(s) is True

    def test_string_series(self):
        s = pd.Series(["C", "CC"])
        assert _is_mol_column(s) is False

    def test_empty_series(self):
        s = pd.Series([], dtype=object)
        assert _is_mol_column(s) is False

    def test_series_with_none(self):
        s = pd.Series([None, Chem.MolFromSmiles("C")])
        assert _is_mol_column(s) is True

    def test_all_none(self):
        s = pd.Series([None, None])
        assert _is_mol_column(s) is False


class TestWriteSmiFile:
    """Test .smi file writing."""

    def test_writes_correct_format(self, tmp_path):
        df = pd.DataFrame({"Smiles": ["CCO", "c1ccccc1", "CC"]})
        filepath = tmp_path / "test.smi"

        written = _write_smi_file(df, "Smiles", filepath)

        assert len(written) == 3
        content = filepath.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "CCO\t0"
        assert lines[1] == "c1ccccc1\t1"
        assert lines[2] == "CC\t2"

    def test_skips_none_smiles(self, tmp_path):
        df = pd.DataFrame({"Smiles": ["CCO", None, "CC"]})
        filepath = tmp_path / "test.smi"

        written = _write_smi_file(df, "Smiles", filepath)

        assert len(written) == 2
        assert "1" not in written

    def test_skips_error_rows(self, tmp_path):
        df = pd.DataFrame(
            {"Smiles": ["CCO", "CC", "CCC"], "__error__": [None, "some error", None]}
        )
        filepath = tmp_path / "test.smi"

        written = _write_smi_file(df, "Smiles", filepath)

        assert len(written) == 2
        assert "1" not in written


class TestWriteSdfFile:
    """Test .sdf file writing with RDKit Mol objects."""

    def test_writes_molecules(self, tmp_path):
        df = pd.DataFrame(
            {
                "ROMol": [
                    Chem.MolFromSmiles("CCO"),
                    Chem.MolFromSmiles("c1ccccc1"),
                ]
            }
        )
        filepath = tmp_path / "test.sdf"

        written = _write_sdf_file(df, "ROMol", filepath)

        assert len(written) == 2
        # Verify the SDF is readable and has correct _Name
        suppl = Chem.SDMolSupplier(str(filepath))
        mols = [m for m in suppl if m is not None]
        assert len(mols) == 2
        assert mols[0].GetProp("_Name") == "0"
        assert mols[1].GetProp("_Name") == "1"

    def test_skips_none_molecules(self, tmp_path):
        df = pd.DataFrame(
            {"ROMol": [Chem.MolFromSmiles("CCO"), None, Chem.MolFromSmiles("CC")]}
        )
        filepath = tmp_path / "test.sdf"

        written = _write_sdf_file(df, "ROMol", filepath)

        assert len(written) == 2
        assert "1" not in written


class TestReadOutputSdf:
    """Test LigPrep output SDF reading."""

    def test_reads_with_variant_tag(self, tmp_path):
        # Create a mock LigPrep output SDF
        filepath = tmp_path / "output.sdf"
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.RWMol(mol)
        mol.SetProp("_Name", "0-1")
        mol.SetProp("s_lp_Variant", "0-1")

        writer = Chem.SDWriter(str(filepath))
        writer.write(mol)
        writer.close()

        results = _read_output_sdf(filepath)
        assert len(results) == 1
        key, variant_id, parsed_mol = results[0]
        assert key == "0"
        assert variant_id == "1"

    def test_reads_multiple_variants(self, tmp_path):
        filepath = tmp_path / "output.sdf"
        writer = Chem.SDWriter(str(filepath))

        for i, smi in enumerate(["CCO", "CC(O)O"]):
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.RWMol(mol)
            mol.SetProp("_Name", f"0-{i+1}")
            mol.SetProp("s_lp_Variant", f"0-{i+1}")
            writer.write(mol)

        writer.close()

        results = _read_output_sdf(filepath)
        assert len(results) == 2
        assert results[0][0] == "0"  # Both from compound 0
        assert results[1][0] == "0"

    def test_fallback_to_name(self, tmp_path):
        """If s_lp_Variant is missing, fall back to _Name."""
        filepath = tmp_path / "output.sdf"
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.RWMol(mol)
        mol.SetProp("_Name", "42")

        writer = Chem.SDWriter(str(filepath))
        writer.write(mol)
        writer.close()

        results = _read_output_sdf(filepath)
        assert len(results) == 1
        assert results[0][0] == "42"

    def test_missing_file(self, tmp_path):
        filepath = tmp_path / "nonexistent.sdf"
        results = _read_output_sdf(filepath)
        assert results == []


class TestLigPrepLink:
    """Test LigPrep link construction and serialization."""

    def test_default_params(self):
        lp = LigPrep()
        assert lp.in_column == "Smiles"
        assert lp.epik is True
        assert lp.max_stereo == 32

    def test_get_params(self):
        lp = LigPrep(epik=False, max_stereo=16)
        params = lp.get_params()
        assert params["epik"] is False
        assert params["max_stereo"] == 16

    def test_from_params(self):
        lp = LigPrep(epik=False, ph=6.5)
        params = lp.get_params()
        lp2 = LigPrep.from_params(params)
        assert lp2.epik is False
        assert lp2.ph == 6.5

    def test_build_command_smiles(self):
        lp = LigPrep(schrodinger_path="/apps/schrodinger")
        cmd = lp._build_command(
            Path("/tmp/input.smi"), Path("/tmp/output.sdf"), is_mol_input=False
        )
        assert "/apps/schrodinger/ligprep" in cmd[0]
        assert "-ismi" in cmd
        assert "-WAIT" in cmd

    def test_build_command_sdf(self):
        lp = LigPrep(schrodinger_path="/apps/schrodinger")
        cmd = lp._build_command(
            Path("/tmp/input.sdf"), Path("/tmp/output.sdf"), is_mol_input=True
        )
        assert "-isd" in cmd

    def test_build_command_no_epik(self):
        lp = LigPrep(schrodinger_path="/apps/schrodinger", epik=False)
        cmd = lp._build_command(
            Path("/tmp/input.smi"), Path("/tmp/output.sdf"), is_mol_input=False
        )
        assert "-epik" not in cmd

    def test_build_expanded_df(self):
        """Test the DataFrame expansion logic with mock parsed data."""
        lp = LigPrep()
        df = pd.DataFrame(
            {"Smiles": ["CCO", "c1ccccc1"], "extra": ["a", "b"]}
        )

        # Simulate LigPrep output: compound 0 got 2 variants, compound 1 got 1
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("OCC")
        mol3 = Chem.MolFromSmiles("c1ccccc1")
        parsed = [
            ("0", "1", mol1),
            ("0", "2", mol2),
            ("1", "1", mol3),
        ]

        result = lp._build_expanded_df(df, ["0", "1"], parsed)

        assert len(result) == 3
        assert list(result["__id__"]) == [0, 0, 1]
        assert list(result["__enum_id__"]) == [0, 1, 2]
        # Original columns duplicated
        assert list(result["extra"]) == ["a", "a", "b"]
        # Original Smiles preserved
        assert list(result["Smiles"]) == ["CCO", "CCO", "c1ccccc1"]
        # Prepared SMILES in separate column
        assert "LigPrepSmiles" in result.columns
        # Prepared mols stored
        assert result["__ROMolLigPrep__"].notna().all()

    def test_build_expanded_df_with_failures(self):
        """Test that missing output produces error rows."""
        lp = LigPrep()
        df = pd.DataFrame({"Smiles": ["CCO", "INVALID", "CC"]})

        mol1 = Chem.MolFromSmiles("CCO")
        # Compound 1 missing from output (failed), compound 2 has output
        mol3 = Chem.MolFromSmiles("CC")
        parsed = [("0", "1", mol1), ("2", "1", mol3)]

        result = lp._build_expanded_df(df, ["0", "1", "2"], parsed)

        assert len(result) == 3
        # Row for compound 1 should have error
        error_row = result[result["__id__"] == 1].iloc[0]
        assert "no output produced" in error_row["__error__"]
