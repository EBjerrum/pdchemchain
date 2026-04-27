"""Schrodinger Glide docking integration."""

import gzip
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from pdchemchain.base import Link
from pdchemchain.config import ensure_schrodinger_job_server, get_schrodinger_path
from pdchemchain.errormanager import has_error
from pdchemchain.typing import InColumnName, Partitionable


# Keys always managed by the link — removed from user .in file before appending ours
_MANAGED_KEYS = {"LIGANDFILE", "POSE_OUTTYPE", "POSES_PER_LIG", "GRIDFILE"}

def _generate_in_file(
    grid_file: str,
    ligand_file: str,
    poses_per_lig: int,
    in_file: str = None,
    overrides: dict = None,
) -> str:
    """Generate Glide .in file content.

    Parameters
    ----------
    grid_file : str
        Path to Glide grid file.
    ligand_file : str
        Path to input ligand SDF.
    poses_per_lig : int
        Number of poses per ligand.
    in_file : str, optional
        Path to user-provided Maestro .in file.
    overrides : dict, optional
        Additional keyword overrides.

    Returns
    -------
    str
        Complete .in file content.
    """
    overrides = overrides or {}

    if in_file is None:
        # Minimal .in with overrides applied
        lines = [
            f"GRIDFILE  {grid_file}",
            f"PRECISION  SP",
            f"POSES_PER_LIG  {poses_per_lig}",
            f"POSE_OUTTYPE  ligandlib_sd",
            f"LIGANDFILE  {ligand_file}",
        ]
        # Apply overrides (replace matching keys)
        override_keys = set(overrides.keys())
        filtered = []
        for line in lines:
            key = line.split()[0] if line.strip() else ""
            if key not in override_keys:
                filtered.append(line)
        filtered.extend(f"{k}  {v}" for k, v in overrides.items())
        return "\n".join(filtered) + "\n"

    # Read user's .in file and modify
    with open(in_file) as f:
        user_lines = f.readlines()

    # Collect all keys to remove: managed + overrides
    keys_to_remove = _MANAGED_KEYS | set(overrides.keys())

    # Filter out lines whose key matches
    filtered_lines = []
    in_block = False
    for line in user_lines:
        stripped = line.strip()
        # Handle block sections like [CONSTRAINT_GROUP:1] — always keep
        if stripped.startswith("["):
            in_block = True
            filtered_lines.append(line)
            continue
        if in_block:
            # Block content lines are indented; unindented line ends block
            if stripped and not line[0].isspace():
                in_block = False
            else:
                filtered_lines.append(line)
                continue

        # Normal key-value line
        key = stripped.split()[0] if stripped else ""
        if key not in keys_to_remove:
            filtered_lines.append(line)

    # Append managed keys
    filtered_lines.append(f"GRIDFILE  {grid_file}\n")
    filtered_lines.append(f"POSES_PER_LIG  {poses_per_lig}\n")
    filtered_lines.append(f"POSE_OUTTYPE  ligandlib_sd\n")
    filtered_lines.append(f"LIGANDFILE  {ligand_file}\n")

    # Append user overrides
    for k, v in overrides.items():
        filtered_lines.append(f"{k}  {v}\n")

    return "".join(filtered_lines)


def _write_ligands_sdf(
    df: pd.DataFrame, mol_column: str, filepath: Path
) -> list[str]:
    """Write molecules to SDF for Glide input.

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


def _find_output_sdf(tmpdir: Path) -> Path | None:
    """Find Glide output SDF file in temp directory."""
    # Glide outputs {jobname}_lib.sdf or {jobname}_lib.sdfgz
    for pattern in ["*_lib.sdf", "*_lib.sdfgz"]:
        matches = list(tmpdir.glob(pattern))
        if matches:
            return matches[0]
    # Fallback: any .sdf file that's not our input
    for sdf in tmpdir.glob("*.sdf"):
        if sdf.name != "ligands.sdf":
            return sdf
    return None


def _read_docked_sdf(
    filepath: Path,
) -> list[tuple[str, dict[str, float], Chem.rdchem.Mol]]:
    """Read Glide output SDF.

    Returns list of (original_key, scores_dict, mol) tuples.
    """
    results = []

    # Handle gzipped output
    actual_path = filepath
    if filepath.suffix == ".sdfgz":
        decompressed = filepath.with_suffix(".sdf")
        with gzip.open(filepath, "rb") as f_in, open(decompressed, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        actual_path = decompressed

    if not actual_path.exists():
        return results

    suppl = Chem.SDMolSupplier(str(actual_path), removeHs=False)
    for mol in suppl:
        if mol is None:
            continue

        # Get original key from _Name
        if not mol.HasProp("_Name"):
            continue
        original_key = mol.GetProp("_Name")

        # Extract scores
        scores = {}
        for prop_name, score_key in [
            ("r_i_docking_score", "docking_score"),
            ("r_i_glide_gscore", "glide_gscore"),
            ("r_i_glide_emodel", "glide_emodel"),
        ]:
            if mol.HasProp(prop_name):
                try:
                    scores[score_key] = float(mol.GetProp(prop_name))
                except (ValueError, RuntimeError):
                    scores[score_key] = np.nan
            else:
                scores[score_key] = np.nan

        results.append((original_key, scores, mol))

    return results


@dataclass
class GlideDock(Link):
    """Dock molecules using Schrodinger Glide.

    Two usage modes:

    1. **Quick start**: provide just ``grid_file`` — uses sensible SP defaults.
    2. **Full control**: provide ``in_file`` from Maestro with custom settings.

    The link manages LIGANDFILE, POSE_OUTTYPE, GRIDFILE, and POSES_PER_LIG
    automatically. Additional Glide keywords can be overridden via the
    ``overrides`` dict.

    Parameters
    ----------
    grid_file : str
        Path to Glide grid file (.zip). Required.
    in_file : str, optional
        Path to Maestro .in file. If None, a minimal SP config is generated.
    mol_column : str
        Input column containing RDKit Mol objects (3D structures).
    score_column : str
        Output column name for the primary docking score.
    gscore_column : str
        Output column name for the GlideScore.
    emodel_column : str
        Output column name for the Emodel score.
    store_poses : bool
        Whether to store docked poses as RDKit Mol objects.
    pose_column : str
        Output column name for docked poses.
    poses_per_lig : int
        Number of docking poses to generate per ligand.
    overrides : dict
        Glide keyword overrides (e.g. ``{"PRECISION": "XP"}``).
    njobs : int
        Number of parallel Glide subjobs.
    schrodinger_path : str, optional
        Explicit path to Schrodinger installation.
    keep_tempdir : bool
        Keep temporary directory after execution.

    Examples
    --------
    >>> from pdchemchain.links.schrodinger import GlideDock
    >>> dock = GlideDock(grid_file="my_grid.zip")
    >>> result = dock(prepared_df)  # DataFrame with __ROMolLigPrep__ column
    """

    grid_file: str = None
    in_file: str = None
    mol_column: InColumnName = "__ROMolLigPrep__"

    # Score output columns
    score_column: str = "docking_score"
    gscore_column: str = "glide_gscore"
    emodel_column: str = "glide_emodel"

    # Pose storage
    store_poses: bool = True
    pose_column: str = "__ROMolDocked__"
    poses_per_lig: int = 1

    # .in file overrides
    overrides: dict = field(default_factory=dict)

    # Execution
    njobs: int = 1
    schrodinger_path: str = None
    keep_tempdir: bool = False

    _partitionable = Partitionable.YES

    def __post_init__(self):
        super().__post_init__()
        if self.grid_file is None:
            raise ValueError("grid_file is required — provide the path to a Glide grid file (.zip)")

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ensure_schrodinger_job_server(self.schrodinger_path)
        df_input = df.copy()
        self.logger.info(f"GlideDock: docking {len(df_input)} molecules")

        tmpdir = Path(tempfile.mkdtemp(prefix="pdchemchain_glide_"))
        try:
            return self._run_glide(df_input, tmpdir)
        finally:
            if not self.keep_tempdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            else:
                self.logger.info(f"GlideDock temp directory kept at: {tmpdir}")

    def _run_glide(self, df: pd.DataFrame, tmpdir: Path) -> pd.DataFrame:
        # Write ligands to SDF
        ligand_file = tmpdir / "ligands.sdf"
        written_keys = _write_ligands_sdf(df, self.mol_column, ligand_file)

        if not written_keys:
            self.logger.warning("GlideDock: no valid molecules to dock")
            errors = pd.Series(
                ["GlideDock: no valid input" if not has_error(row) else None
                 for _, row in df.iterrows()],
                index=df.index,
            )
            if errors.notna().any():
                df = self.append_errors(df, errors)
            return df

        # Generate .in file
        in_content = _generate_in_file(
            grid_file=self.grid_file,
            ligand_file=str(ligand_file),
            poses_per_lig=self.poses_per_lig,
            in_file=self.in_file,
            overrides=self.overrides,
        )
        in_file_path = tmpdir / "glide.in"
        with open(in_file_path, "w") as f:
            f.write(in_content)

        self.logger.debug(f"GlideDock .in file:\n{in_content}")

        # Run Glide
        schrodinger = get_schrodinger_path(self.schrodinger_path)
        glide_bin = str(Path(schrodinger) / "glide")
        cmd = [
            glide_bin,
            str(in_file_path),
            "-WAIT",
            "-HOST",
            "localhost",
            "-NJOBS",
            str(self.njobs),
        ]

        self.logger.info(f"GlideDock command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmpdir))

        if result.returncode != 0:
            self.logger.error(
                f"Glide failed (rc={result.returncode}): {result.stderr}"
            )

        # Find and parse output
        output_file = _find_output_sdf(tmpdir)
        if output_file is None:
            self.logger.warning("GlideDock: no output file found")
            parsed = []
        else:
            parsed = _read_docked_sdf(output_file)
            self.logger.info(
                f"GlideDock: {len(written_keys)} input → {len(parsed)} docked poses"
            )

        return self._build_result_df(df, written_keys, parsed)

    def _build_result_df(
        self,
        df: pd.DataFrame,
        written_keys: list[str],
        parsed: list[tuple[str, dict[str, float], Chem.rdchem.Mol]],
    ) -> pd.DataFrame:
        # Group parsed results by original key
        results_by_key: dict[
            str, list[tuple[dict[str, float], Chem.rdchem.Mol]]
        ] = {}
        for original_key, scores, mol in parsed:
            results_by_key.setdefault(original_key, []).append((scores, mol))

        # Track whether input already has __id__ (from LigPrep)
        has_id = "__id__" in df.columns

        expanded_rows = []
        enum_counter = 0

        for idx, row in df.iterrows():
            key = str(idx)

            # Pass through rows with existing errors
            if has_error(row):
                new_row = row.copy()
                if not has_id:
                    new_row["__id__"] = idx
                new_row["__enum_id__"] = enum_counter
                enum_counter += 1
                expanded_rows.append(new_row)
                continue

            # Row wasn't written (None mol)
            if key not in written_keys:
                new_row = row.copy()
                if not has_id:
                    new_row["__id__"] = idx
                new_row["__enum_id__"] = enum_counter
                new_row[self.score_column] = np.nan
                new_row[self.gscore_column] = np.nan
                new_row[self.emodel_column] = np.nan
                new_row["__error__"] = "GlideDock: no valid molecule for docking"
                enum_counter += 1
                expanded_rows.append(new_row)
                continue

            # Check for docking results
            poses = results_by_key.get(key, [])
            if not poses:
                new_row = row.copy()
                if not has_id:
                    new_row["__id__"] = idx
                new_row["__enum_id__"] = enum_counter
                new_row[self.score_column] = np.nan
                new_row[self.gscore_column] = np.nan
                new_row[self.emodel_column] = np.nan
                new_row["__error__"] = "GlideDock: no docking result"
                enum_counter += 1
                expanded_rows.append(new_row)
                continue

            # Add one row per pose
            for scores, mol in poses:
                new_row = row.copy()
                if not has_id:
                    new_row["__id__"] = idx
                new_row["__enum_id__"] = enum_counter
                new_row[self.score_column] = scores.get("docking_score", np.nan)
                new_row[self.gscore_column] = scores.get("glide_gscore", np.nan)
                new_row[self.emodel_column] = scores.get("glide_emodel", np.nan)
                if self.store_poses:
                    new_row[self.pose_column] = mol
                enum_counter += 1
                expanded_rows.append(new_row)

        result_df = pd.DataFrame(expanded_rows)
        result_df = result_df.reset_index(drop=True)
        return result_df
