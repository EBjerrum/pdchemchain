"""Tests for PrimeMMGBSA link — unit tests only (no Schrodinger required)."""

import zipfile

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from pdchemchain.links.schrodinger.mmgbsa import (
    PrimeMMGBSA,
    _parse_output_csv,
    _resolve_receptor_file,
    _write_ligands_sdf,
)


class TestWriteLigandsSdf:
    """Test ligand SDF writing for structcat input."""

    def test_writes_molecules(self, tmp_path):
        df = pd.DataFrame(
            {"__ROMolDocked__": [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC")]}
        )
        filepath = tmp_path / "ligands.sdf"
        written = _write_ligands_sdf(df, "__ROMolDocked__", filepath)

        assert len(written) == 2
        suppl = Chem.SDMolSupplier(str(filepath))
        mols = [m for m in suppl if m is not None]
        assert len(mols) == 2
        assert mols[0].GetProp("_Name") == "0"

    def test_skips_error_rows(self, tmp_path):
        df = pd.DataFrame(
            {
                "__ROMolDocked__": [Chem.MolFromSmiles("CCO"), None],
                "__error__": [None, "GlideDock failed"],
            }
        )
        filepath = tmp_path / "ligands.sdf"
        written = _write_ligands_sdf(df, "__ROMolDocked__", filepath)

        assert len(written) == 1

    def test_skips_none_mols(self, tmp_path):
        df = pd.DataFrame({"__ROMolDocked__": [Chem.MolFromSmiles("CCO"), None]})
        filepath = tmp_path / "ligands.sdf"
        written = _write_ligands_sdf(df, "__ROMolDocked__", filepath)

        assert len(written) == 1


class TestResolveReceptorFile:
    """Test receptor file resolution from .mae or Glide grid .zip."""

    def test_mae_file_passthrough(self, tmp_path):
        mae_path = str(tmp_path / "receptor.mae")
        result = _resolve_receptor_file(mae_path, tmp_path)
        assert result == mae_path

    def test_extracts_from_grid_zip(self, tmp_path):
        # Create a mock Glide grid .zip with a *_recep.mae inside
        zip_path = tmp_path / "grid.zip"
        recep_content = b"fake mae content"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("grid_recep.mae", recep_content)
            zf.writestr("grid.grd", b"grid data")

        result = _resolve_receptor_file(str(zip_path), tmp_path)

        assert result.endswith("grid_recep.mae")
        assert open(result, "rb").read() == recep_content

    def test_zip_without_recep_raises(self, tmp_path):
        zip_path = tmp_path / "bad_grid.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("grid.grd", b"grid data")

        with pytest.raises(ValueError, match="No \\*_recep.mae found"):
            _resolve_receptor_file(str(zip_path), tmp_path)


class TestParseOutputCsv:
    """Test MMGBSA output CSV parsing."""

    def _make_csv(self, tmp_path, rows):
        """Write a mock MMGBSA output CSV."""
        filepath = tmp_path / "test-out.csv"
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        return filepath

    def test_default_scores(self, tmp_path):
        filepath = self._make_csv(tmp_path, [
            {
                "title": "Reinvent_0",
                "r_psp_MMGBSA_dG_Bind": -81.5,
                "r_psp_Lig_Strain_Energy": 14.8,
                "r_psp_MMGBSA_dG_Bind_Coulomb": -122.3,
            },
        ])
        results = _parse_output_csv(filepath, all_scores=False)

        assert "Reinvent_0" in results
        scores = results["Reinvent_0"]
        assert scores["mmgbsa_dg_bind"] == pytest.approx(-81.5)
        assert scores["mmgbsa_lig_strain"] == pytest.approx(14.8)
        # Coulomb should NOT be included in default mode
        assert "mmgbsa_MMGBSA_dG_Bind_Coulomb" not in scores

    def test_all_scores(self, tmp_path):
        filepath = self._make_csv(tmp_path, [
            {
                "title": "Reinvent_0",
                "r_psp_MMGBSA_dG_Bind": -81.5,
                "r_psp_Lig_Strain_Energy": 14.8,
                "r_psp_MMGBSA_dG_Bind_Coulomb": -122.3,
            },
        ])
        results = _parse_output_csv(filepath, all_scores=True)

        scores = results["Reinvent_0"]
        assert scores["mmgbsa_MMGBSA_dG_Bind"] == pytest.approx(-81.5)
        assert scores["mmgbsa_Lig_Strain_Energy"] == pytest.approx(14.8)
        assert scores["mmgbsa_MMGBSA_dG_Bind_Coulomb"] == pytest.approx(-122.3)

    def test_multiple_ligands(self, tmp_path):
        filepath = self._make_csv(tmp_path, [
            {"title": "0", "r_psp_MMGBSA_dG_Bind": -81.5, "r_psp_Lig_Strain_Energy": 14.8},
            {"title": "1", "r_psp_MMGBSA_dG_Bind": -55.0, "r_psp_Lig_Strain_Energy": 5.3},
        ])
        results = _parse_output_csv(filepath, all_scores=False)

        assert len(results) == 2
        assert results["0"]["mmgbsa_dg_bind"] == pytest.approx(-81.5)
        assert results["1"]["mmgbsa_dg_bind"] == pytest.approx(-55.0)

    def test_numeric_titles_read_as_strings(self, tmp_path):
        """Ensure numeric-looking titles like '0' aren't mangled by pandas."""
        filepath = self._make_csv(tmp_path, [
            {"title": "0", "r_psp_MMGBSA_dG_Bind": -81.5, "r_psp_Lig_Strain_Energy": 14.8},
        ])
        results = _parse_output_csv(filepath, all_scores=False)

        assert "0" in results


class TestPrimeMMGBSALink:
    """Test PrimeMMGBSA link construction and serialization."""

    def test_requires_receptor_file(self):
        with pytest.raises(ValueError, match="receptor_file is required"):
            PrimeMMGBSA()

    def test_default_params(self):
        link = PrimeMMGBSA(receptor_file="/path/to/receptor.mae")
        assert link.mol_column == "__ROMolDocked__"
        assert link.flexdist == 3.0
        assert link.out_type == "LIGAND"
        assert link.all_scores is False
        assert link.output_file is None

    def test_get_params(self):
        link = PrimeMMGBSA(receptor_file="/recep.mae", njobs=4, flexdist=5.0)
        params = link.get_params()
        assert params["receptor_file"] == "/recep.mae"
        assert params["njobs"] == 4
        assert params["flexdist"] == 5.0

    def test_from_params(self):
        link = PrimeMMGBSA(
            receptor_file="/recep.mae", flexdist=5.0, all_scores=True
        )
        params = link.get_params()
        link2 = PrimeMMGBSA.from_params(params)
        assert link2.receptor_file == "/recep.mae"
        assert link2.flexdist == 5.0
        assert link2.all_scores is True

    def test_merge_scores(self):
        """Test score merging back into DataFrame."""
        link = PrimeMMGBSA(receptor_file="/recep.mae")
        df = pd.DataFrame(
            {
                "__ROMolDocked__": [
                    Chem.MolFromSmiles("CCO"),
                    Chem.MolFromSmiles("CC"),
                ],
                "Smiles": ["CCO", "CC"],
            }
        )

        scores_by_key = {
            "0": {"mmgbsa_dg_bind": -81.5, "mmgbsa_lig_strain": 14.8},
            "1": {"mmgbsa_dg_bind": -55.0, "mmgbsa_lig_strain": 5.3},
        }

        result = link._merge_scores(df, ["0", "1"], scores_by_key)

        assert len(result) == 2
        assert result.iloc[0]["mmgbsa_dg_bind"] == pytest.approx(-81.5)
        assert result.iloc[1]["mmgbsa_dg_bind"] == pytest.approx(-55.0)
        assert result.iloc[0]["mmgbsa_lig_strain"] == pytest.approx(14.8)

    def test_merge_scores_with_failures(self):
        """Test that missing MMGBSA results produce error rows."""
        link = PrimeMMGBSA(receptor_file="/recep.mae")
        df = pd.DataFrame(
            {
                "__ROMolDocked__": [
                    Chem.MolFromSmiles("CCO"),
                    Chem.MolFromSmiles("CC"),
                ],
            }
        )

        # Only compound 0 got a result
        scores_by_key = {
            "0": {"mmgbsa_dg_bind": -81.5, "mmgbsa_lig_strain": 14.8},
        }

        result = link._merge_scores(df, ["0", "1"], scores_by_key)

        assert len(result) == 2
        assert result.iloc[0]["mmgbsa_dg_bind"] == pytest.approx(-81.5)
        assert np.isnan(result.iloc[1]["mmgbsa_dg_bind"])
        assert "no result returned" in str(result.iloc[1].get("__error__", ""))
