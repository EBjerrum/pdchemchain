import copy
from dataclasses import dataclass, field
from typing import List

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from pdchemchain.base import Link, RowLink
from pdchemchain.errormanager import RDKitErrorContextManager
from pdchemchain.typing import InColumnName


@dataclass
class ElementsInList(RowLink):
    """Checks if a given molecule only has certain elements

    The Link will check if the molecule only contains the elements in the list
    and reports True or False in the output column

    Parameters
    ----------
    in_column
        The label for the column containing the molecules to check
    out_column
        The label for the column that should store the results of the check
    allowed_elements
        List of atomic numbers of elements to allow
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "ElementsAllowed"
    allowed_elements: List[int] = field(
        default_factory=lambda: [6, 7, 8, 9, 16, 17, 35]
    )  # How does it work with set?

    def _all_elements_in_allowed_set(self, mol: Chem.Mol) -> bool:
        mol_elements = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
        return mol_elements.issubset(self.allowed_elements)

    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol = row[self.in_column]
        row[self.out_column] = self._all_elements_in_allowed_set(mol)
        return row


@dataclass
class HeavyAtomCount(RowLink):
    """Counts the number of heavy atoms

    Parameters
    ----------
    in_column
        The label for the column containing the molecules to process
    out_column
        The label for the column that should store the results
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "HeavyAtomCount"

    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol = row[self.in_column]
        row[self.out_column] = Descriptors.HeavyAtomCount(mol)
        return row


@dataclass
class HeteroAtomRatio(RowLink):
    """Calculates the ratio of heteroatom to heavy atoms

    If the molecule contains no heavy atoms, the ratio is set to 0

    Parameters
    ----------
    in_column
        The label for the column containing the molecules to process
    out_column
        The label for the column that should store the results
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "HeteroAtomRatio"

    def _heteroatom_ratio(self, mol: Chem.Mol) -> float:
        heavy_atoms: int = Descriptors.HeavyAtomCount(mol)
        hetero_atoms: int = Descriptors.NumHeteroatoms(
            mol
        )  # sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [6, 1])  # Exclude C and H
        return (
            hetero_atoms / heavy_atoms if heavy_atoms > 0 else 0.0
        )  # Avoid division by zero

    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol = row[self.in_column]
        row[self.out_column] = self._heteroatom_ratio(mol)
        return row


@dataclass
class LongestAliphaticChain(RowLink):
    """Finds the length of the longest aliphatic carbon chain

    The longest aliphatic chain up to a length of max_chain_length are reported

    Parameters
    ----------
    max_chain_length
        The maximum chain length to search for and report
    in_column
        The label for the column containing the molecules to process
    out_column
        The label for the column that should store the results
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "LongestAliphaticChain"
    max_chain_length: int = 11

    def __post_init__(self):
        super().__post_init__()
        self.SMARTS_CHAINS = [
            Chem.MolFromSmarts("-".join(["[CR0H2]"] * i))
            for i in range(1, self.max_chain_length + 1)
        ]

    def _find_longest_aliphatic_chain_length(self, mol: Chem.Mol) -> int:
        for i, chain in enumerate(self.SMARTS_CHAINS, start=1):
            if mol.HasSubstructMatch(chain):
                continue
            return i - 1  # -1 to adjust for 0-based index

        return 0

    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol = row[self.in_column]
        row[self.out_column] = self._find_longest_aliphatic_chain_length(mol)
        return row


@dataclass
class MolFromSmiles(RowLink):
    """Converts SMILES strings to molecules row-wise

    SMILES strings are converted row-wise. Conversion errors is reported in the __error__ column

    Parameters
    ----------
    in_column
        The label for the column containing the SMILES strings to convert
    out_column
        The label for the column that should store the RDKit molecular objects
    """

    in_column: InColumnName = "Smiles"
    out_column: str = "ROMol"

    def _row_apply(self, row: pd.Series) -> pd.Series:
        e = RDKitErrorContextManager()
        with e:
            mol = Chem.MolFromSmiles(row[self.in_column])
        # if "pytest" in sys.modules and mol is None:
        #      raise ValueError ("SMILES parsing failed with unknown error. Seemingly running under pytest where error context manager is not fully functional.")
        if mol is None:
            raise ValueError(f"RDKit Error: {e.errors}")
        else:
            row[self.out_column] = mol
        return row


@dataclass
class MolToSmiles(RowLink):
    """Converts RDKit molecular objects to SMILES strings.

    Molecules are converted row-wise.

    Parameters
    ----------
    in_column
        The label for the column containing the molecules to convert
    out_column
        The label for the column that should store the SMILES strings
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "Smiles"

    def _row_apply(self, row: pd.Series) -> pd.Series:
        row[self.out_column] = Chem.MolToSmiles(row[self.in_column])
        return row


@dataclass
class NumberOfTokens(RowLink):
    """Counts the number of tokens in the SMILES string

    Uses the PySMILESUtils to tokenize and counts the numbers of tokens.
    PySMILESUtils must be installed for this Link to function.

    Parameters
    ----------
    in_column
        The label for the column containing the molecules to analyze
    out_column
        The label for the column that should store the token count

    Raises
    ------
    ImportError
        Raised if PySMILESUtils is not installed, installation instructions are provided.
    """

    in_column: InColumnName = "Smiles"
    out_column: str = "NumTokens"

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer = self._import_dependency()

    def _import_dependency(self):
        try:
            from pysmilesutils.tokenize import SMILESAtomTokenizer as tokenizer

            return tokenizer(warn=False)
        except ImportError as e:
            raise ImportError(
                "The 'pysmilesutils' module is required for this class. "
                "Please install it using: pip install git+ssh://git@github.com/EBjerrum/pysmilesutils.git"
            ) from e

    def _row_apply(self, row: pd.Series) -> pd.Series:
        """Number of tokens from the smiles string, without start and end tokens"""
        smiles = row[self.in_column]
        tokens = self.tokenizer.tokenize([smiles], enclose=False)[0]
        row[self.out_column] = len(tokens)

        return row


@dataclass
class PandasAddMoleculeColumn(Link):
    """Adds a molecule column using the PandasTools utility

    The PandasTools AddMoleculeColumnToFrame is used to add the molecule to the dataframe.
    SMILES failing conversion are reported as NaN in the column, and a note provided in the __error__ column.
    Unlike the MolFromSmiles Link, the reason is can't be provided.

    Parameters
    ----------
    smilesCol
        The label for the column containing the SMILES strings to convert
    molCol
        The label for the column that should store the RDKit molecular objects
    includeFingerprints
        Calculate fingerprints for the molecules that can be used for fast substructure matching
        (see PandasTools documentation http://rdkit.org/docs/source/rdkit.Chem.PandasTools.html)
    """

    smilesCol: InColumnName = "Smiles"
    molCol: str = "ROMol"
    includeFingerprints: bool = False

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()  # PandasTools mutates the dataframe
        PandasTools.AddMoleculeColumnToFrame(
            df,
            smilesCol=self.smilesCol,
            molCol=self.molCol,
            includeFingerprints=self.includeFingerprints,
        )  # Should maybe work on a copy of the dataframe, we mutate the input.

        # Example on error handling on the df level
        error_mask = df[self.molCol].isna()
        if sum(error_mask) > 0:
            errors = pd.Series([None] * len(df))
            errors[error_mask] = "Error in PandasTools SMILES conversion"
            df = self.append_errors(df, errors)
            self.logger.warning(f"{len(errors)} SMILES failed in conversion")
        return df


@dataclass
class RDKitDescriptors(RowLink):
    """Use the RDKit MolecularDescriptor calculator to add descriptor columns

    Calculates the descriptors and properties of the molecules using RDKit. Columns are named after the descriptor list.

    Parameters
    ----------
    in_column
        The label for the column containing the molecules to analyze
    descriptors
        A list of RDKit descriptors to calculate
    """

    in_column: InColumnName = "ROMol"
    descriptors: List[str] = field(
        default_factory=lambda: ["MolWt", "MolLogP", "NumHAcceptors", "NumHDonors"]
    )
    # TODO, add functions to list available descriptos, check if descriptors in list are correct, and update descriptors on object (and calculator)
    # TODO, possibility to add prefix or suffix to descriptor names, as well as provide custom list
    # TODO, check that we don't overwrite existing columns

    def __post_init__(self):
        super().__post_init__()
        self.calculator = MolecularDescriptorCalculator(self.descriptors)

    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol = row[self.in_column]
        if isinstance(mol, Chem.Mol):
            for desc_name, value in zip(
                self.descriptors, self.calculator.CalcDescriptors(mol)
            ):  # TODO, Calculator return -666 or -666.0 if no mol, and maybe other erros as well?, maybe use to catch errors?
                row[desc_name] = value
        else:
            raise ValueError(f"Seemingly not a Mol object: {mol} of type {type(mol)}")
        return row


@dataclass
class RemoveStereoMol(RowLink):
    """Remove stereo information from the molecular object

    The out_column label can be the same as the in_column label,
    in which case the input molecules will get substituted with the converted ones

    Parameters
    in_column
        The label for the column containing the molecules to strip stereo information from
    out_column
        The label for the column that should store the converted molecules

    Returns
    -------
    _type_
        _description_
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "ROMol"

    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol = copy.deepcopy(
            row[self.in_column]
        )  # TODO, whats most efficient, working on copies of the rows/objects or simply copy the dataframe?
        Chem.RemoveStereochemistry(mol)  # This mutates the input column?
        row[self.out_column] = mol
        return row


@dataclass
class RemoveStereoSmiles(RowLink):
    """Remove stereo information from the SMILES

    Uses string manipulation to remove stereo information from SMILES by stripping it of @ charachters

    Parameters
    ----------
    in_column
        The label for the column containing the SMILES strings to convert
    out_column
        The label for the column to store the converted SMILES strings
    """

    in_column: InColumnName = "Smiles"
    out_column: str = "Smiles"

    def _row_apply(self, row: pd.Series) -> pd.Series:
        row[self.out_column] = row[self.in_column].replace("@", "")
        return row


@dataclass
class SuperParent(RowLink):
    """Standardizes the molecule to the super parent structure

    The super parent is the fragment, charge, isotope, stereo, and tautomer parent of the molecule.
    However, the tautomer standardization is switched off by default by setting maxTautomers to generate to 0 (default 1000 in RDKit)

    Parameters
    ----------
    in_column
        The label for the column containing the molecules to standardize
    out_column
        The label for the column that should store the standardized molecules
    maxTautomers
        The maximum number of tautormers to generate during standardization attempts
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "ROMol"
    maxTautomers: int = 0

    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol = copy.deepcopy(row[self.in_column])
        if (
            self.maxTautomers != 1000
        ):  # TODO, what's the overhead of reinstating the params each time?
            params = rdMolStandardize.CleanupParameters()
            params.maxTautomers = self.maxTautomers
            mol = rdMolStandardize.SuperParent(mol, params=params)
        else:
            mol = rdMolStandardize.SuperParent(mol)
        row[self.out_column] = mol
        return row
