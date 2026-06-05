from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from pdchemchain.base import Link
from pdchemchain.io_utilities import df_process_to_csv
from pdchemchain.typing import InColumnName


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
        Path to the SDF file to write
    mol_column : InColumnName
        Column name containing RDKit molecule objects (default: "ROMol")
    properties : Optional[List[str]]
        List of column names to write as SD properties.
        If None (default), writes all columns except the molecule column.
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
    """

    filename: str
    mol_column: InColumnName = "ROMol"
    properties: Optional[List[str]] = None
    handle_none: bool | str = True
    sdf_write_options: Dict[str, any] = field(default_factory=dict)

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
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

        PandasTools.WriteSDF(
            df_copy,
            self.filename,
            molColName=self.mol_column,
            properties=props,
            **self.sdf_write_options,
        )

        self.logger.info(
            f"Saved dataframe to SDF file {self.filename} " f"with {len(props)} properties"
        )
        return df
