"""Schrodinger Prime MM-GBSA binding energy estimation."""

import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from pdchemchain.base import Link
from pdchemchain.config import get_schrodinger_path, ensure_schrodinger_job_server
from pdchemchain.errormanager import has_error
from pdchemchain.typing import InColumnName, Partitionable


# Default score columns to import (primary binding energy + ligand strain)
_DEFAULT_SCORES = {
    "r_psp_MMGBSA_dG_Bind": "mmgbsa_dg_bind",
    "r_psp_Lig_Strain_Energy": "mmgbsa_lig_strain",
}


def _write_ligands_sdf(
    df: pd.DataFrame, mol_column: str, filepath: Path
) -> list[str]:
    """Write docked poses to SDF for structcat input.

    Returns list of written index keys (as strings).
    """
    written = []
    writer = Chem.SDWriter(str(filepath))
    for idx, row in df.iterrows():
        if has_error(row):
            continue
        mol = row[mol_column]
        if mol is None:
            continue
        key = str(idx)
        mol = Chem.RWMol(mol)
        mol.SetProp("_Name", key)
        writer.write(mol)
        written.append(key)
    writer.close()
    return written


def _build_pose_viewer_mae(
    receptor_file: str,
    ligand_sdf: Path,
    output_mae: Path,
    schrodinger_path: str,
) -> None:
    """Combine receptor .mae and ligand .sdf into a pose viewer .mae file.

    Uses Schrodinger's ``structcat`` utility for reliable format conversion.
    """
    structcat = str(Path(schrodinger_path) / "utilities" / "structcat")
    cmd = [
        structcat,
        "-imae", str(receptor_file),
        "-isd", str(ligand_sdf),
        "-omae", str(output_mae),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"structcat failed: {result.stderr}")


_JOBNAME = "mmgbsa_calc"


def _resolve_receptor_file(receptor_file: str, tmpdir: Path) -> str:
    """Resolve receptor file path, extracting from Glide grid .zip if needed.

    Parameters
    ----------
    receptor_file : str
        Path to receptor .mae file, or Glide grid .zip containing one.
    tmpdir : Path
        Temp directory for extraction.

    Returns
    -------
    str
        Path to the receptor .mae file.
    """
    if not receptor_file.endswith(".zip"):
        return receptor_file

    with zipfile.ZipFile(receptor_file, "r") as zf:
        recep_entries = [n for n in zf.namelist() if n.endswith("_recep.mae")]
        if not recep_entries:
            raise ValueError(
                f"No *_recep.mae found in {receptor_file}. "
                "Provide a .mae receptor file directly."
            )
        recep_name = recep_entries[0]
        extracted = tmpdir / recep_name
        zf.extract(recep_name, tmpdir)
        return str(extracted)


def _parse_output_csv(
    filepath: Path,
    all_scores: bool,
) -> dict[str, dict[str, float]]:
    """Parse MMGBSA output CSV into per-ligand score dicts.

    Parameters
    ----------
    filepath : Path
        Path to the *-out.csv file.
    all_scores : bool
        If True, import all r_psp_ columns. If False, only default scores.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of title -> {score_name: value}.
    """
    df_out = pd.read_csv(filepath, dtype={"title": str})

    if all_scores:
        # Import all r_psp_ columns, strip the prefix
        score_cols = [c for c in df_out.columns if c.startswith("r_psp_")]
        col_map = {c: c.replace("r_psp_", "mmgbsa_") for c in score_cols}
    else:
        # Only default scores
        score_cols = [c for c in _DEFAULT_SCORES if c in df_out.columns]
        col_map = {c: _DEFAULT_SCORES[c] for c in score_cols}

    results = {}
    for _, row in df_out.iterrows():
        title = str(row["title"])
        scores = {}
        for src_col, dst_col in col_map.items():
            try:
                scores[dst_col] = float(row[src_col])
            except (ValueError, TypeError):
                scores[dst_col] = np.nan
        results[title] = scores

    return results


@dataclass
class PrimeMMGBSA(Link):
    """Estimate binding free energy using Schrodinger Prime MM-GBSA.

    Runs ``prime_mmgbsa`` on docked poses with a receptor structure. Input
    is a pose viewer .mae file (receptor + ligand poses), written
    automatically from the DataFrame and receptor file.

    Typical pipeline position::

        LigPrep → GlideDock → AggregateByScore → PrimeMMGBSA

    Parameters
    ----------
    receptor_file : str
        Path to receptor .mae file or Glide grid .zip file (the receptor
        is extracted automatically). Using the same grid file as GlideDock
        ensures receptor consistency. Required.
    mol_column : str
        Input column containing docked RDKit Mol objects (3D poses).
    score_column : str
        Output column for MM-GBSA dG_Bind.
    strain_column : str
        Output column for ligand strain energy.
    flexdist : float
        Distance (Angstrom) around ligand for flexible residues.
    out_type : str
        Output type for prime_mmgbsa.
    all_scores : bool
        If True, import all MM-GBSA decomposition terms (prefixed ``mmgbsa_``).
    output_file : str, optional
        Path to save MMGBSA output structures (.maegz). If None, structures
        are not saved.
    njobs : int
        Number of parallel subjobs.
    overrides : dict
        Additional CLI flags as key-value pairs (e.g. ``{"OPLS_VERSION": "S-OPLS"}``).
    schrodinger_path : str, optional
        Explicit path to Schrodinger installation.
    keep_tempdir : bool
        Keep temporary directory after execution.

    Examples
    --------
    >>> from pdchemchain.links.schrodinger import PrimeMMGBSA
    >>> mmgbsa = PrimeMMGBSA(receptor_file="receptor.mae")
    >>> result = mmgbsa(docked_df)
    """

    receptor_file: str = None
    mol_column: InColumnName = "__ROMolDocked__"

    # Score output columns
    score_column: str = "mmgbsa_dg_bind"
    strain_column: str = "mmgbsa_lig_strain"

    # MMGBSA parameters
    flexdist: float = 3.0
    out_type: str = "LIGAND"
    all_scores: bool = False

    # Output
    output_file: str = None

    # Execution
    njobs: int = 1
    overrides: dict = field(default_factory=dict)
    schrodinger_path: str = None
    keep_tempdir: bool = False

    _partitionable = Partitionable.YES

    def __post_init__(self):
        super().__post_init__()
        if self.receptor_file is None:
            raise ValueError(
                "receptor_file is required — provide the path to a receptor .mae file"
            )

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ensure_schrodinger_job_server(self.schrodinger_path)
        df_result = df.copy()
        self.logger.info(f"PrimeMMGBSA: processing {len(df_result)} molecules")

        tmpdir = Path(tempfile.mkdtemp(prefix="pdchemchain_mmgbsa_"))
        try:
            return self._run_mmgbsa(df_result, tmpdir)
        finally:
            if not self.keep_tempdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            else:
                self.logger.info(f"PrimeMMGBSA temp directory kept at: {tmpdir}")

    def _run_mmgbsa(self, df: pd.DataFrame, tmpdir: Path) -> pd.DataFrame:
        schrodinger = get_schrodinger_path(self.schrodinger_path)

        # Resolve receptor (supports both .mae and Glide grid .zip)
        receptor = _resolve_receptor_file(self.receptor_file, tmpdir)

        # Write ligands to SDF, then combine with receptor into pose viewer .mae
        ligand_sdf = tmpdir / "ligands.sdf"
        written_keys = _write_ligands_sdf(df, self.mol_column, ligand_sdf)

        if not written_keys:
            self.logger.warning("PrimeMMGBSA: no valid molecules to process")
            errors = pd.Series(
                ["PrimeMMGBSA: no valid input" if not has_error(row) else None
                 for _, row in df.iterrows()],
                index=df.index,
            )
            if errors.notna().any():
                df = self.append_errors(df, errors)
            return df

        input_mae = tmpdir / f"{_JOBNAME}_pv.mae"
        _build_pose_viewer_mae(
            receptor, ligand_sdf, input_mae, schrodinger
        )

        # Build command with -jobname for predictable output naming
        prime_bin = str(Path(schrodinger) / "prime_mmgbsa")
        cmd = [
            prime_bin,
            str(input_mae),
            "-jobname", _JOBNAME,
            "-out_type", self.out_type,
            "-flexdist", str(self.flexdist),
            "-OVERWRITE",
            "-WAIT",
            "-HOST", "localhost",
            "-NJOBS", str(self.njobs),
        ]

        # Apply overrides (e.g. OPLS_VERSION=S-OPLS → -prime_opt OPLS_VERSION=S-OPLS)
        for key, value in self.overrides.items():
            cmd.extend(["-prime_opt", f"{key}={value}"])

        self.logger.info(f"PrimeMMGBSA command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmpdir))

        if result.returncode != 0:
            self.logger.error(
                f"prime_mmgbsa failed (rc={result.returncode}): {result.stderr}"
            )

        # Parse output CSV (predictable name from -jobname)
        output_csv = tmpdir / f"{_JOBNAME}-out.csv"
        if not output_csv.exists():
            self.logger.warning("PrimeMMGBSA: no output CSV found")
            scores_by_key = {}
        else:
            scores_by_key = _parse_output_csv(output_csv, self.all_scores)
            self.logger.info(
                f"PrimeMMGBSA: {len(written_keys)} input → "
                f"{len(scores_by_key)} results"
            )

        # Optionally copy output structures
        if self.output_file:
            out_mae = tmpdir / f"{_JOBNAME}-out.maegz"
            if out_mae.exists():
                shutil.copy2(out_mae, self.output_file)
                self.logger.info(f"PrimeMMGBSA: structures saved to {self.output_file}")
            else:
                self.logger.warning("PrimeMMGBSA: no output structure file found")

        # Map scores back to DataFrame
        return self._merge_scores(df, written_keys, scores_by_key)

    def _merge_scores(
        self,
        df: pd.DataFrame,
        written_keys: list[str],
        scores_by_key: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        errors = pd.Series([None] * len(df), index=df.index, dtype="object")

        # Determine which score columns to initialize
        if scores_by_key:
            all_score_names = set()
            for scores in scores_by_key.values():
                all_score_names.update(scores.keys())
        else:
            all_score_names = {self.score_column, self.strain_column}

        # Initialize score columns with NaN
        for col in all_score_names:
            df[col] = np.nan

        for idx, row in df.iterrows():
            key = str(idx)

            if has_error(row):
                continue

            if key not in written_keys:
                errors[idx] = "PrimeMMGBSA: no valid molecule"
                continue

            scores = scores_by_key.get(key)
            if scores is None:
                errors[idx] = "PrimeMMGBSA: no result returned"
                continue

            for col, value in scores.items():
                df.at[idx, col] = value

        if errors.notna().any():
            df = self.append_errors(df, errors)

        return df
