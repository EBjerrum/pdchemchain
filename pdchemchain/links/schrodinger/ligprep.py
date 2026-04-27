"""Schrodinger LigPrep integration for ligand preparation."""

import gzip
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem

from pdchemchain.base import Link
from pdchemchain.config import ensure_schrodinger_job_server, get_schrodinger_path
from pdchemchain.errormanager import has_error
from pdchemchain.typing import InColumnName, Partitionable


def _parse_variant_tag(variant_tag: str) -> tuple[str, str]:
    """Parse s_lp_Variant tag to extract original title and variant number.

    Parameters
    ----------
    variant_tag : str
        LigPrep variant tag, format: ``{title}-{variant_number}``.
        Title may contain hyphens, so we split from the right.

    Returns
    -------
    tuple[str, str]
        (original_title, variant_number)
    """
    # Split on last hyphen: "my-mol-1" -> ("my-mol", "1")
    parts = variant_tag.rsplit("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    # Fallback: no hyphen found, use whole tag as title
    return variant_tag, "0"


def _is_mol_column(series: pd.Series) -> bool:
    """Check if a pandas Series contains RDKit Mol objects."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    return isinstance(non_null.iloc[0], Chem.rdchem.Mol)


def _write_smi_file(df: pd.DataFrame, column: str, filepath: Path) -> list[str]:
    """Write SMILES to .smi file with index as title.

    Returns list of written index keys (as strings).
    """
    written = []
    with open(filepath, "w") as f:
        for idx, row in df.iterrows():
            if has_error(row):
                continue
            smiles = row[column]
            if pd.isna(smiles) or not smiles:
                continue
            key = str(idx)
            f.write(f"{smiles}\t{key}\n")
            written.append(key)
    return written


def _write_sdf_file(df: pd.DataFrame, column: str, filepath: Path) -> list[str]:
    """Write RDKit Mol objects to .sdf file with index as _Name.

    Returns list of written index keys (as strings).
    """
    written = []
    writer = Chem.SDWriter(str(filepath))
    for idx, row in df.iterrows():
        if has_error(row):
            continue
        mol = row[column]
        if mol is None:
            continue
        key = str(idx)
        mol = Chem.RWMol(mol)
        mol.SetProp("_Name", key)
        writer.write(mol)
        written.append(key)
    writer.close()
    return written


def _read_output_sdf(filepath: Path) -> list[tuple[str, str, Chem.rdchem.Mol]]:
    """Read LigPrep output SDF and parse variant tags.

    Returns list of (original_key, variant_id, mol) tuples.
    """
    results = []

    # Handle gzipped output
    if filepath.suffix == ".sdfgz" or (
        filepath.with_suffix(".sdf.gz").exists() and not filepath.exists()
    ):
        gz_path = (
            filepath
            if filepath.suffix == ".sdfgz"
            else filepath.with_suffix(".sdf.gz")
        )
        sdf_path = filepath.with_suffix(".sdf")
        with gzip.open(gz_path, "rb") as f_in, open(sdf_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        filepath = sdf_path

    if not filepath.exists():
        return results

    suppl = Chem.SDMolSupplier(str(filepath), removeHs=False)
    for mol in suppl:
        if mol is None:
            continue

        # Parse s_lp_Variant to get original key
        if mol.HasProp("s_lp_Variant"):
            variant_tag = mol.GetProp("s_lp_Variant")
            original_key, variant_id = _parse_variant_tag(variant_tag)
        elif mol.HasProp("_Name"):
            # Fallback: use _Name directly
            original_key = mol.GetProp("_Name")
            variant_id = "0"
        else:
            continue

        results.append((original_key, variant_id, mol))

    return results


@dataclass
class LigPrep(Link):
    """Prepare ligands using Schrodinger LigPrep.

    Supports SMILES input (writes .smi file) or ROMol input (writes .sdf file).
    The input type is determined by the dtype of the ``in_column``: set it to a
    column containing SMILES strings or RDKit Mol objects.

    One-to-many expansion: a single input molecule may produce multiple prepared
    variants (tautomers, stereoisomers, ionization states).

    Parameters
    ----------
    in_column : str
        Input column name. If the column contains strings, a .smi file is written.
        If it contains RDKit Mol objects, a .sdf file is written.
    out_mol_column : str
        Column name for the 3D prepared molecule.
    out_smiles_column : str
        Column name for SMILES of the prepared variant. Original Smiles column
        is preserved unchanged.
    epik : bool
        Use Epik for ionization/tautomerization.
    max_stereo : int
        Maximum number of stereoisomers to generate.
    ph : float
        Target pH for Epik.
    ph_tolerance : float
        pH tolerance for Epik.
    njobs : int
        Number of parallel LigPrep subjobs.
    schrodinger_path : str, optional
        Explicit path to Schrodinger installation. If None, discovered via config.
    keep_tempdir : bool
        Keep temporary directory after execution (for debugging).

    Examples
    --------
    >>> from pdchemchain.links.schrodinger import LigPrep
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Smiles": ["CCO", "c1ccccc1"]})
    >>> prep = LigPrep()
    >>> result = prep(df)  # Returns expanded DataFrame with __id__, __enum_id__, __ROMolLigPrep__
    """

    in_column: InColumnName = "Smiles"
    out_mol_column: str = "__ROMolLigPrep__"
    out_smiles_column: str = "LigPrepSmiles"

    # LigPrep parameters
    epik: bool = True
    max_stereo: int = 32
    ph: float = 7.0
    ph_tolerance: float = 2.0

    # Execution
    njobs: int = 1
    schrodinger_path: str = None
    keep_tempdir: bool = False

    _partitionable = Partitionable.YES

    def _build_command(
        self, input_file: Path, output_file: Path, is_mol_input: bool
    ) -> list[str]:
        """Build the LigPrep command line."""
        schrodinger = get_schrodinger_path(self.schrodinger_path)
        ligprep_bin = str(Path(schrodinger) / "ligprep")

        input_flag = "-isd" if is_mol_input else "-ismi"
        cmd = [
            ligprep_bin,
            input_flag,
            str(input_file),
            "-osd",
            str(output_file),
            "-WAIT",
            "-HOST",
            "localhost",
            "-NJOBS",
            str(self.njobs),
            "-s",
            str(self.max_stereo),
        ]

        if self.epik:
            cmd.extend(["-epik", "-ph", str(self.ph), "-pht", str(self.ph_tolerance)])

        return cmd

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ensure_schrodinger_job_server(self.schrodinger_path)
        df_input = df.copy()

        # Determine input type from column dtype
        is_mol_input = _is_mol_column(df_input[self.in_column])
        self.logger.info(
            f"LigPrep: processing {len(df_input)} molecules "
            f"({'SDF' if is_mol_input else 'SMILES'} mode)"
        )

        # Create temp directory
        tmpdir = Path(tempfile.mkdtemp(prefix="pdchemchain_ligprep_"))
        try:
            return self._run_ligprep(df_input, tmpdir, is_mol_input)
        finally:
            if not self.keep_tempdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            else:
                self.logger.info(f"LigPrep temp directory kept at: {tmpdir}")

    def _run_ligprep(
        self, df: pd.DataFrame, tmpdir: Path, is_mol_input: bool
    ) -> pd.DataFrame:
        # Write input file
        if is_mol_input:
            input_file = tmpdir / "input.sdf"
            written_keys = _write_sdf_file(df, self.in_column, input_file)
        else:
            input_file = tmpdir / "input.smi"
            written_keys = _write_smi_file(df, self.in_column, input_file)

        if not written_keys:
            self.logger.warning("LigPrep: no valid molecules to process")
            df["__id__"] = range(len(df))
            df["__enum_id__"] = range(len(df))
            errors = pd.Series(
                ["LigPrep: no valid input" if not has_error(row) else None
                 for _, row in df.iterrows()],
                index=df.index,
            )
            if errors.notna().any():
                df = self.append_errors(df, errors)
            return df

        output_file = tmpdir / "output.sdf"

        # Run LigPrep
        cmd = self._build_command(input_file, output_file, is_mol_input)
        self.logger.info(f"LigPrep command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(tmpdir)
        )

        if result.returncode != 0:
            self.logger.error(f"LigPrep failed (rc={result.returncode}): {result.stderr}")

        # Parse output
        parsed = _read_output_sdf(output_file)
        self.logger.info(
            f"LigPrep: {len(written_keys)} input → {len(parsed)} output molecules"
        )

        # Build expanded DataFrame
        return self._build_expanded_df(df, written_keys, parsed)

    def _build_expanded_df(
        self,
        df: pd.DataFrame,
        written_keys: list[str],
        parsed: list[tuple[str, str, Chem.rdchem.Mol]],
    ) -> pd.DataFrame:
        # Group parsed results by original key
        results_by_key: dict[str, list[tuple[str, Chem.rdchem.Mol]]] = {}
        for original_key, variant_id, mol in parsed:
            results_by_key.setdefault(original_key, []).append((variant_id, mol))

        expanded_rows = []
        enum_counter = 0

        for idx, row in df.iterrows():
            key = str(idx)

            # Pass through rows that already had errors
            if has_error(row):
                new_row = row.copy()
                new_row["__id__"] = idx
                new_row["__enum_id__"] = enum_counter
                enum_counter += 1
                expanded_rows.append(new_row)
                continue

            # Check if this key was written (valid input)
            if key not in written_keys:
                new_row = row.copy()
                new_row["__id__"] = idx
                new_row["__enum_id__"] = enum_counter
                new_row["__error__"] = "LigPrep: invalid input (None or empty)"
                enum_counter += 1
                expanded_rows.append(new_row)
                continue

            # Check if LigPrep produced output for this key
            variants = results_by_key.get(key, [])
            if not variants:
                new_row = row.copy()
                new_row["__id__"] = idx
                new_row["__enum_id__"] = enum_counter
                new_row["__error__"] = "LigPrep: no output produced for this molecule"
                enum_counter += 1
                expanded_rows.append(new_row)
                continue

            # Expand: one row per variant
            for variant_id, mol in variants:
                new_row = row.copy()
                new_row["__id__"] = idx
                new_row["__enum_id__"] = enum_counter
                new_row[self.out_mol_column] = mol
                # Store prepared SMILES in separate column (original Smiles preserved)
                try:
                    new_row[self.out_smiles_column] = Chem.MolToSmiles(mol)
                except Exception:
                    pass
                enum_counter += 1
                expanded_rows.append(new_row)

        result_df = pd.DataFrame(expanded_rows)
        result_df = result_df.reset_index(drop=True)
        return result_df
