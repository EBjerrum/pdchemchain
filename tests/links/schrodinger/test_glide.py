"""Tests for GlideDock link — unit tests only (no Schrodinger required)."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from pdchemchain.links.schrodinger.glide import (
    GlideDock,
    _generate_in_file,
    _write_ligands_sdf,
    _read_docked_sdf,
    _find_output_sdf,
)


class TestGenerateInFile:
    """Test .in file generation and manipulation."""

    def test_minimal_in_file(self):
        content = _generate_in_file(
            grid_file="/path/to/grid.zip",
            ligand_file="/tmp/ligands.sdf",
            poses_per_lig=1,
        )
        assert "GRIDFILE  /path/to/grid.zip" in content
        assert "PRECISION  SP" in content
        assert "POSES_PER_LIG  1" in content
        assert "POSE_OUTTYPE  ligandlib_sd" in content
        assert "LIGANDFILE  /tmp/ligands.sdf" in content

    def test_minimal_with_overrides(self):
        content = _generate_in_file(
            grid_file="/path/to/grid.zip",
            ligand_file="/tmp/ligands.sdf",
            poses_per_lig=3,
            overrides={"PRECISION": "XP", "AMIDE_MODE": "trans"},
        )
        assert "PRECISION  XP" in content
        assert "AMIDE_MODE  trans" in content
        # Default SP precision should be replaced
        assert content.count("PRECISION") == 1

    def test_user_in_file(self, tmp_path):
        """Test that user .in file is read and modified correctly."""
        user_in = tmp_path / "user.in"
        user_in.write_text(
            "GRIDFILE  /old/grid.zip\n"
            "PRECISION  XP\n"
            "AMIDE_MODE  trans\n"
            "LIGANDFILE  /old/ligands.sdf\n"
            "EXPANDED_SAMPLING  True\n"
        )

        content = _generate_in_file(
            grid_file="/new/grid.zip",
            ligand_file="/tmp/new_ligands.sdf",
            poses_per_lig=2,
            in_file=str(user_in),
        )

        # Managed keys replaced
        assert "GRIDFILE  /new/grid.zip" in content
        assert "LIGANDFILE  /tmp/new_ligands.sdf" in content
        assert "POSES_PER_LIG  2" in content
        assert "POSE_OUTTYPE  ligandlib_sd" in content

        # User settings preserved
        assert "PRECISION  XP" in content
        assert "AMIDE_MODE  trans" in content
        assert "EXPANDED_SAMPLING  True" in content

        # Old managed values removed
        assert "/old/grid.zip" not in content
        assert "/old/ligands.sdf" not in content

    def test_user_in_file_with_overrides(self, tmp_path):
        """Test that overrides replace user .in file settings."""
        user_in = tmp_path / "user.in"
        user_in.write_text(
            "PRECISION  SP\n"
            "AMIDE_MODE  penal\n"
        )

        content = _generate_in_file(
            grid_file="/grid.zip",
            ligand_file="/ligands.sdf",
            poses_per_lig=1,
            in_file=str(user_in),
            overrides={"PRECISION": "XP"},
        )

        # Override takes effect
        assert "PRECISION  XP" in content
        # Only one PRECISION line
        assert content.count("PRECISION") == 1
        # Non-overridden setting preserved
        assert "AMIDE_MODE  penal" in content

    def test_preserves_constraint_blocks(self, tmp_path):
        """Test that [CONSTRAINT_GROUP] blocks are preserved."""
        user_in = tmp_path / "user.in"
        user_in.write_text(
            "PRECISION  SP\n"
            "GRIDFILE  /old/grid.zip\n"
            "\n"
            "[CONSTRAINT_GROUP:1]\n"
            "    USE_CONS   A:ASP:93:OD1(hbond):1,\n"
            "    NREQUIRED_CONS   ALL\n"
            "\n"
            "[FEATURE:1]\n"
            '    PATTERN1   "[#1][#7] 1 include"\n'
        )

        content = _generate_in_file(
            grid_file="/new/grid.zip",
            ligand_file="/ligands.sdf",
            poses_per_lig=1,
            in_file=str(user_in),
        )

        assert "[CONSTRAINT_GROUP:1]" in content
        assert "USE_CONS" in content
        assert "[FEATURE:1]" in content
        assert "PATTERN1" in content


class TestWriteLigandsSdf:
    """Test ligand SDF writing for Glide input."""

    def test_writes_molecules(self, tmp_path):
        df = pd.DataFrame(
            {
                "__ROMolLigPrep__": [
                    Chem.MolFromSmiles("CCO"),
                    Chem.MolFromSmiles("CC"),
                ]
            }
        )
        filepath = tmp_path / "ligands.sdf"
        written = _write_ligands_sdf(df, "__ROMolLigPrep__", filepath)

        assert len(written) == 2
        suppl = Chem.SDMolSupplier(str(filepath))
        mols = [m for m in suppl if m is not None]
        assert len(mols) == 2
        assert mols[0].GetProp("_Name") == "0"

    def test_skips_error_rows(self, tmp_path):
        df = pd.DataFrame(
            {
                "__ROMolLigPrep__": [Chem.MolFromSmiles("CCO"), None],
                "__error__": [None, "LigPrep failed"],
            }
        )
        filepath = tmp_path / "ligands.sdf"
        written = _write_ligands_sdf(df, "__ROMolLigPrep__", filepath)

        assert len(written) == 1


class TestReadDockedSdf:
    """Test Glide output SDF parsing."""

    def test_reads_scores(self, tmp_path):
        filepath = tmp_path / "output_lib.sdf"
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.RWMol(mol)
        mol.SetProp("_Name", "0")
        mol.SetProp("r_i_docking_score", "-8.5")
        mol.SetProp("r_i_glide_gscore", "-8.2")
        mol.SetProp("r_i_glide_emodel", "-55.3")

        writer = Chem.SDWriter(str(filepath))
        writer.write(mol)
        writer.close()

        results = _read_docked_sdf(filepath)
        assert len(results) == 1
        key, scores, parsed_mol = results[0]
        assert key == "0"
        assert scores["docking_score"] == pytest.approx(-8.5)
        assert scores["glide_gscore"] == pytest.approx(-8.2)
        assert scores["glide_emodel"] == pytest.approx(-55.3)

    def test_handles_missing_scores(self, tmp_path):
        filepath = tmp_path / "output_lib.sdf"
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.RWMol(mol)
        mol.SetProp("_Name", "0")
        # Only docking_score, missing others

        mol.SetProp("r_i_docking_score", "-8.5")

        writer = Chem.SDWriter(str(filepath))
        writer.write(mol)
        writer.close()

        results = _read_docked_sdf(filepath)
        assert len(results) == 1
        _, scores, _ = results[0]
        assert scores["docking_score"] == pytest.approx(-8.5)
        assert np.isnan(scores["glide_gscore"])
        assert np.isnan(scores["glide_emodel"])

    def test_multiple_poses(self, tmp_path):
        filepath = tmp_path / "output_lib.sdf"
        writer = Chem.SDWriter(str(filepath))

        for score in [-8.5, -7.2]:
            mol = Chem.MolFromSmiles("CCO")
            mol = Chem.RWMol(mol)
            mol.SetProp("_Name", "0")
            mol.SetProp("r_i_docking_score", str(score))
            writer.write(mol)

        writer.close()

        results = _read_docked_sdf(filepath)
        assert len(results) == 2
        assert results[0][0] == "0"
        assert results[1][0] == "0"

    def test_missing_file(self, tmp_path):
        results = _read_docked_sdf(tmp_path / "nonexistent.sdf")
        assert results == []


class TestFindOutputSdf:
    """Test Glide output file discovery."""

    def test_finds_lib_sdf(self, tmp_path):
        (tmp_path / "glide_job_lib.sdf").touch()
        result = _find_output_sdf(tmp_path)
        assert result is not None
        assert result.name == "glide_job_lib.sdf"

    def test_finds_lib_sdfgz(self, tmp_path):
        (tmp_path / "glide_job_lib.sdfgz").touch()
        result = _find_output_sdf(tmp_path)
        assert result is not None
        assert result.name == "glide_job_lib.sdfgz"

    def test_ignores_input_sdf(self, tmp_path):
        (tmp_path / "ligands.sdf").touch()
        result = _find_output_sdf(tmp_path)
        assert result is None

    def test_no_output(self, tmp_path):
        result = _find_output_sdf(tmp_path)
        assert result is None


class TestGlideDockLink:
    """Test GlideDock link construction and serialization."""

    def test_requires_grid_file(self):
        with pytest.raises(ValueError, match="grid_file is required"):
            GlideDock()

    def test_default_params(self):
        dock = GlideDock(grid_file="/path/to/grid.zip")
        assert dock.mol_column == "__ROMolLigPrep__"
        assert dock.poses_per_lig == 1
        assert dock.store_poses is True

    def test_get_params(self):
        dock = GlideDock(grid_file="/grid.zip", njobs=4)
        params = dock.get_params()
        assert params["grid_file"] == "/grid.zip"
        assert params["njobs"] == 4

    def test_from_params(self):
        dock = GlideDock(grid_file="/grid.zip", poses_per_lig=3)
        params = dock.get_params()
        dock2 = GlideDock.from_params(params)
        assert dock2.grid_file == "/grid.zip"
        assert dock2.poses_per_lig == 3

    def test_build_result_df(self):
        """Test result DataFrame building with mock parsed data."""
        dock = GlideDock(grid_file="/grid.zip")
        df = pd.DataFrame(
            {
                "__ROMolLigPrep__": [
                    Chem.MolFromSmiles("CCO"),
                    Chem.MolFromSmiles("CC"),
                ],
                "Smiles": ["CCO", "CC"],
            }
        )

        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CC")
        parsed = [
            ("0", {"docking_score": -8.5, "glide_gscore": -8.2, "glide_emodel": -55.0}, mol1),
            ("1", {"docking_score": -6.0, "glide_gscore": -5.8, "glide_emodel": -40.0}, mol2),
        ]

        result = dock._build_result_df(df, ["0", "1"], parsed)

        assert len(result) == 2
        assert result.iloc[0]["docking_score"] == -8.5
        assert result.iloc[1]["docking_score"] == -6.0
        assert "__ROMolDocked__" in result.columns

    def test_build_result_df_with_failures(self):
        """Test that missing docking results produce error rows."""
        dock = GlideDock(grid_file="/grid.zip")
        df = pd.DataFrame(
            {
                "__ROMolLigPrep__": [
                    Chem.MolFromSmiles("CCO"),
                    Chem.MolFromSmiles("CC"),
                ],
            }
        )

        # Only compound 0 docked successfully
        mol1 = Chem.MolFromSmiles("CCO")
        parsed = [
            ("0", {"docking_score": -8.5, "glide_gscore": -8.2, "glide_emodel": -55.0}, mol1),
        ]

        result = dock._build_result_df(df, ["0", "1"], parsed)

        assert len(result) == 2
        # Compound 1 should have error
        error_row = result.iloc[1]
        assert "no docking result" in str(error_row.get("__error__", ""))
        assert np.isnan(error_row["docking_score"])

    def test_build_result_df_multiple_poses(self):
        """Test expansion when poses_per_lig > 1."""
        dock = GlideDock(grid_file="/grid.zip", poses_per_lig=2)
        df = pd.DataFrame(
            {
                "__ROMolLigPrep__": [Chem.MolFromSmiles("CCO")],
            }
        )

        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")
        parsed = [
            ("0", {"docking_score": -8.5, "glide_gscore": -8.2, "glide_emodel": -55.0}, mol1),
            ("0", {"docking_score": -7.2, "glide_gscore": -7.0, "glide_emodel": -45.0}, mol2),
        ]

        result = dock._build_result_df(df, ["0"], parsed)

        assert len(result) == 2
        assert list(result["__id__"]) == [0, 0]
        assert result.iloc[0]["docking_score"] == -8.5
        assert result.iloc[1]["docking_score"] == -7.2
