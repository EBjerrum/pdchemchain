import gzip
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from pdchemchain.base import Link
from pdchemchain.io_utilities import df_process_to_csv
from pdchemchain.typing import InColumnName


def enable_mol_property_pickling() -> None:
    """Make RDKit include molecule properties when pickling.

    RDKit's default pickle keeps atoms, bonds and conformers but DROPS molecule-level
    properties, so a round-trip silently loses exactly the SD tags worth keeping -- for a
    Glide pose, ``r_i_docking_score``, ``s_glide_core_smarts``, the per-constraint
    satisfaction flags. The loss is invisible: the Mol comes back valid, just bare.

    This is a GLOBAL RDKit setting, not a per-call argument, so any process that reads or
    writes a pickled frame of molecules has to call it -- including a worker on another
    machine. :class:`ToPickle` and :class:`FromPickle` call it for you.
    """
    from rdkit import Chem

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def _generate_romol(df: pd.DataFrame, smiles_column: str, logger) -> pd.DataFrame:
    """(Re)generate ROMol column from a SMILES column, overwriting any existing ROMol."""
    from pdchemchain.links.chemistry import MolFromSmiles
    logger.info(f"(Re)generating ROMol from SMILES column '{smiles_column}'")
    return MolFromSmiles(in_column=smiles_column)(df)


@dataclass
class FromFile(Link):
    """Reads dataframe from a CSV file

    Uses pandas read_csv to load tabular data from a CSV file into a DataFrame.
    This link expects no input DataFrame (or an empty one) and will overwrite any
    provided input. It serves as a starting point for data processing chains.

    Parameters
    ----------
    filename : str
        Path to the CSV file to read. Can be absolute or relative path.
    pd_readcsv_options : Dict[str, any]
        Additional keyword arguments passed to pandas.read_csv.
        Common options include: sep (delimiter), header (row number for column names),
        index_col (column to use as row labels), usecols (subset of columns to read),
        dtype (data type specifications), encoding (file encoding).
        Default: {"sep": ","}

    Examples
    --------
    >>> # Load CSV with default comma separator
    >>> link = FromFile("data.csv")
    >>> df = link()
    >>>
    >>> # Load tab-separated file with custom options
    >>> link = FromFile("data.tsv", pd_readcsv_options={"sep": "\t", "index_col": 0})
    >>> df = link()
    """

    filename: str
    mol_from_smiles_column: Optional[str] = None
    pd_readcsv_options: Dict[str, any] = field(default_factory=lambda: {"sep": ","})

    def apply(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = pd.DataFrame()
        return super().apply(df)

    def __call__(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self.apply(df)

    def _apply(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = pd.DataFrame()
        if not df.empty:
            self.logger.warning(
                f"Input dataframe received in FromFile is not empty as expected but contained {len(df)} rows, possible error as it will get overwritten"
            )
        df = pd.read_csv(self.filename, **self.pd_readcsv_options)
        self.logger.info(
            f"Loaded dataframe from CSV file {self.filename}, dataframe has {len(df)} rows."
        )
        if self.mol_from_smiles_column is not None:
            df = _generate_romol(df, self.mol_from_smiles_column, self.logger)
        return df


@dataclass
class ToFile(Link):
    """Writes dataframe to a CSV file

    Uses pandas to_csv to save the DataFrame as a CSV file. This link is a
    pass-through operation that returns the input DataFrame unchanged, allowing
    it to be placed anywhere in a processing chain for intermediate or final
    data export.

    Parameters
    ----------
    filename : str
        Path where the CSV file should be written. Can be absolute or relative path.
        Parent directories must exist.
    pd_tocsv_options : Dict[str, any]
        Additional keyword arguments passed to pandas.to_csv.
        Common options include: sep (delimiter), index (whether to write row names),
        header (whether to write column names), columns (subset of columns to write),
        mode ('w' for write, 'a' for append), encoding (file encoding).
        Default: {"sep": ","}

    Examples
    --------
    >>> # Save DataFrame at end of chain
    >>> chain = FromFile("input.csv") + ProcessingLink() + ToFile("output.csv")
    >>> df = chain()
    >>>
    >>> # Save intermediate results with custom options
    >>> chain = link1 + ToFile("checkpoint.csv", pd_tocsv_options={"index": False}) + link2
    >>> df = chain(input_df)
    """

    filename: str
    pd_tocsv_options: Dict[str, any] = field(default_factory=lambda: {"sep": ","})

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df_process_to_csv(df, self.filename, **self.pd_tocsv_options)
        self.logger.info(f"Saved dataframe with basename {self.filename}")
        return df


@dataclass
class ToPickle(Link):
    """Writes the dataframe to a pickle, molecule properties and all.

    The only format here that round-trips a dataframe FAITHFULLY: several Mol columns at
    once, 3D conformers, molecule-level SD properties, fingerprint bit vectors, NaNs and
    arbitrary Python objects. :class:`ToSDF` carries one Mol column and flattens the rest
    to SD tags; CSV loses every object column; parquet raises on a Mol unless each such
    column is hand-encoded to bytes first (and comes out roughly twice the size).

    That makes it the right format for handing a partition to another process or machine --
    see :class:`~pdchemchain.links.hpc.SlurmPartitionProcessor`, which appends one of
    these to the inner chain. It is a pass-through link, so it can sit anywhere in a chain.

    NOT AN ARCHIVE FORMAT. A pickle can break across pandas or RDKit versions, and loading
    one executes arbitrary code, so read only pickles your own job wrote. For a deliverable
    use :class:`ToSDF` or :class:`ToFile`.

    Parameters
    ----------
    filename : str
        Path to write. A ``.gz``/``.bz2``/``.zst`` suffix compresses (pandas infers it);
        gzip takes a molecule frame to about 15% of raw for a little CPU.
    pd_topickle_options : Dict[str, any]
        Extra keyword arguments for pandas ``to_pickle`` (e.g. ``compression``,
        ``protocol``).

    Examples
    --------
    >>> # Stage a partition for a worker on another machine
    >>> chain = inner_chain + ToPickle("work/shard_003.out.pkl")
    """

    filename: str
    pd_topickle_options: Dict[str, any] = field(default_factory=dict)

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        enable_mol_property_pickling()
        parent = os.path.dirname(self.filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_pickle(self.filename, **self.pd_topickle_options)
        self.logger.info(f"Pickled dataframe ({len(df)} rows) to {self.filename}")
        return df


@dataclass
class FromPickle(Link):
    """Reads a dataframe from a pickle written by :class:`ToPickle`.

    Sets RDKit's property-preserving pickle mode before reading, so molecule properties
    come back rather than being silently dropped. See :class:`ToPickle` for why this is the
    faithful format and why it is not an archive format.

    Parameters
    ----------
    filename : str
        Path to read. Compression is inferred from the suffix.
    pd_readpickle_options : Dict[str, any]
        Extra keyword arguments for pandas ``read_pickle``.

    Examples
    --------
    >>> chain = FromPickle("work/shard_003.pkl") + GlideDock(...) + ToPickle("out.pkl")
    """

    filename: str
    pd_readpickle_options: Dict[str, any] = field(default_factory=dict)

    def _apply(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        enable_mol_property_pickling()
        if df is not None and not df.empty:
            self.logger.warning(
                f"Input dataframe received in FromPickle is not empty as expected but "
                f"contained {len(df)} rows, possible error as it will get overwritten"
            )
        out = pd.read_pickle(self.filename, **self.pd_readpickle_options)
        self.logger.info(f"Read dataframe ({len(out)} rows) from {self.filename}")
        return out


@dataclass
class FromSDF(Link):
    """Reads dataframe from an SDF file with RDKit molecule objects

    Uses RDKit's PandasTools.LoadSDF to load molecules and their properties
    into a DataFrame. All SD file properties become DataFrame columns.

    Parameters
    ----------
    filename : str
        Path to the SDF file to read
    mol_column : str
        Column name for RDKit molecule objects (default: "ROMol")
    recognize_placeholders : bool
        If True, recognizes error placeholder molecules (created by ToSDF with handle_none)
        and converts them back to None. Placeholder molecules are identified by having:
        - Exactly 1 atom
        - Atomic number 0 (wildcard)
        - Molecule name "ERROR_NO_MOLECULE"
        Default: False
    infer_types : bool
        If True, attempts to cast string columns to numeric types after loading.
        SDF properties are always read as strings by RDKit; this restores numeric
        types for columns where all non-null values parse cleanly as numbers.
        Columns with any non-numeric values are left as strings. Default: True
    mol_from_smiles_column : str, optional
        When set, (re)generates the ROMol column from the named SMILES column,
        overwriting any molecule loaded from the SDF. Useful for stripping 3D
        coordinates before redocking, or for running SDF pipelines on CSV input.
        Default: None (use molecule as loaded)
    sdf_load_options : Dict[str, any]
        Additional keyword arguments passed to PandasTools.LoadSDF
        Common options: removeHs (bool), sanitize (bool), strictParsing (bool)

    Notes
    -----
    Use recognize_placeholders=True when loading SDF files that were written with
    ToSDF(handle_none=True) to restore None values for molecules that failed processing.
    This enables proper round-trip handling of error rows.
    """

    filename: str
    mol_column: str = "ROMol"
    recognize_placeholders: bool = True
    infer_types: bool = True
    mol_from_smiles_column: Optional[str] = None
    sdf_load_options: Dict[str, any] = field(default_factory=dict)

    def apply(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = pd.DataFrame()
        return super().apply(df)

    def __call__(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self.apply(df)

    def _apply(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = pd.DataFrame()
        if not df.empty:
            self.logger.warning(
                f"Input dataframe received in FromSDF is not empty as expected "
                f"but contained {len(df)} rows, possible error as it will get overwritten"
            )

        from rdkit.Chem import PandasTools

        df = PandasTools.LoadSDF(
            self.filename, molColName=self.mol_column, **self.sdf_load_options
        )

        # Recognize and convert error placeholder molecules back to None
        if self.recognize_placeholders and self.mol_column in df.columns:
            placeholder_count = 0

            for idx, mol in df[self.mol_column].items():
                if mol is not None:
                    # Check if this is an error placeholder molecule
                    if (
                        mol.GetNumAtoms() == 1
                        and mol.GetAtomWithIdx(0).GetAtomicNum() == 0
                        and mol.HasProp("_Name")
                        and mol.GetProp("_Name") == "ERROR_NO_MOLECULE"
                    ):
                        df.at[idx, self.mol_column] = None
                        placeholder_count += 1

            if placeholder_count > 0:
                self.logger.info(
                    f"Converted {placeholder_count} error placeholder molecules back to None"
                )

        if self.infer_types:
            for col in df.columns:
                if col == self.mol_column:
                    continue
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.isna().sum() == df[col].isna().sum():
                    df[col] = converted

        if self.mol_from_smiles_column is not None:
            df = _generate_romol(df, self.mol_from_smiles_column, self.logger)

        self.logger.info(
            f"Loaded dataframe from SDF file {self.filename}, "
            f"dataframe has {len(df)} rows."
        )
        return df


@dataclass
class ToSDF(Link):
    """Writes dataframe to an SDF file with RDKit molecule objects

    Uses RDKit's PandasTools.WriteSDF to write molecules and DataFrame
    columns as SD file properties. By default, all columns are written
    as properties.

    Parameters
    ----------
    filename : str
        Path to the SDF file to write. A ``.gz`` suffix (e.g. ``poses.sdf.gz``)
        writes gzip-compressed SDF. Parent directories are created if missing.
    mol_column : InColumnName
        Column name containing RDKit molecule objects (default: "ROMol")
    properties : Optional[List[str]]
        List of column names to write as SD properties.
        If None (default), writes all columns except the molecule column,
        dunder columns and other molecule columns (logged).
    increment_basename : bool
        If True, each call writes a new file with a zero-padded counter inserted
        before the extension (``poses.sdf.gz`` -> ``poses_00000.sdf.gz``,
        ``poses_00001.sdf.gz``, ...). Existing files are never overwritten, so a
        restarted run continues the numbering. Useful for dumping every batch of
        a long running chain, e.g. docking poses in a REINVENT run.
        Default: False
    handle_none : str
        How to handle None/missing molecules. Options:
        - False: Fail if None molecules are present (safe but strict behavior)
        - True or "warn": Replace None with error placeholder molecules and log warning (default)
        - "silent": Replace None with error placeholder molecules without warning
        Default: True
    sdf_write_options : Dict[str, any]
        Additional keyword arguments passed to PandasTools.WriteSDF
        Common options: allNumeric (bool)

    Notes
    -----
    When handle_none is True or "warn"/"silent", None molecules are replaced with
    wildcard atoms (*) labeled "ERROR" with MolWt=0. This allows saving error rows
    to SDF while making it obvious they are placeholders. Use FromSDF with
    recognize_placeholders=True to convert them back to None on loading.

    Examples
    --------
    >>> # Dump the docked poses of every REINVENT batch after aggregation
    >>> ToSDF("pose_dumps/poses.sdf.gz", mol_column="__ROMolDocked__",
    ...       increment_basename=True)
    """

    filename: str
    mol_column: InColumnName = "ROMol"
    properties: Optional[List[str]] = None
    handle_none: bool | str = True
    increment_basename: bool = False
    sdf_write_options: Dict[str, any] = field(default_factory=dict)

    def _next_filename(self) -> str:
        """Return the first unused numbered filename, starting after the last one written."""
        compressed = self.filename.endswith(".gz")
        base, ext = os.path.splitext(self.filename[:-3] if compressed else self.filename)
        if compressed:
            ext += ".gz"
        # ponytail: exists-check is not atomic, parallel writers to the same basename could collide
        index = getattr(self, "_batch_index", 0)
        while os.path.exists(f"{base}_{index:05d}{ext}"):
            index += 1
        self._batch_index = index + 1
        return f"{base}_{index:05d}{ext}"

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        from rdkit import Chem
        from rdkit.Chem import PandasTools

        df_copy = df.copy()

        # Handle None molecules if requested
        if self.handle_none:
            none_count = df_copy[self.mol_column].isna().sum()

            if none_count > 0:
                from pdchemchain.utilities import get_error_placeholder_molecule

                dummy_mol = get_error_placeholder_molecule()

                # Warn unless silent mode
                if self.handle_none != "silent":
                    self.logger.warning(
                        f"Replacing {none_count} None molecules with error placeholder molecules "
                        f"(wildcard atoms with 'ERROR' label)"
                    )

                # Replace None values with placeholder
                df_copy.loc[df_copy[self.mol_column].isna(), self.mol_column] = (
                    dummy_mol
                )

        # Determine which properties to write
        props = self.properties
        if props is None:
            # Write all columns except molecule column and dunder columns
            props = [
                col
                for col in df_copy.columns
                if col != self.mol_column and not col.startswith("__")
            ]
            # Other Mol columns would be written as "<rdkit.Chem.rdchem.Mol object at ...>"
            other_mol_columns = [
                col
                for col in props
                if df_copy[col].dtype == object
                and isinstance(next(iter(df_copy[col].dropna()), None), Chem.Mol)
            ]
            if other_mol_columns:
                self.logger.info(
                    f"Not writing other molecule columns as SD properties: {other_mol_columns}"
                )
                props = [col for col in props if col not in other_mol_columns]

        filename = self._next_filename() if self.increment_basename else self.filename
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        if filename.endswith(".gz"):
            with gzip.open(filename, "wt") as file_handle:
                PandasTools.WriteSDF(
                    df_copy,
                    file_handle,
                    molColName=self.mol_column,
                    properties=props,
                    **self.sdf_write_options,
                )
        else:
            PandasTools.WriteSDF(
                df_copy,
                filename,
                molColName=self.mol_column,
                properties=props,
                **self.sdf_write_options,
            )

        self.logger.info(
            f"Saved dataframe to SDF file {filename} " f"with {len(props)} properties"
        )
        return df
