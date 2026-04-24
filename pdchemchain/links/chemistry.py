import copy
from dataclasses import dataclass, field
from typing import List, Optional
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools, RDConfig, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from pdchemchain.base import Link, RowLink
from pdchemchain.errormanager import RDKitErrorContextManager
from pdchemchain.typing import InColumnName, Partitionable


@dataclass
class ABMPSScore(RowLink):
    """Calculate AB-MPS score
    
    Calculates the AB-MPS score defined as: abs(cLogP - 3) + NumAromaticRings + NumRotatableBonds
    
    This score provides a simple metric for evaluate bRo5 chemical matter, 
    with AB-MPS values of ≤14 predicting a higher probability of success.
    
    Reference: Beyond the Rule of 5: Lessons Learned from AbbVie's Drugs and Compound Collection
    https://doi.org/10.1021/acs.jmedchem.7b00717
    
    Parameters
    ----------
    in_column
        The label for the column containing the molecules to analyze
    out_column
        The label for the column that should store the AB-MPS score
    """
    
    in_column: InColumnName = "ROMol"
    out_column: str = "AB_MPS_Score"
    
    def __post_init__(self):
        super().__post_init__()
        # Set up descriptor calculator for the three needed descriptors
        Warning("This class is currently NOT calculating the right ABMPSscore as it depends on logD not LogP as used here")
        descriptors = ["MolLogP", "NumAromaticRings", "NumRotatableBonds"]
        self.calculator = MolecularDescriptorCalculator(descriptors)
    
    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol = row[self.in_column]
        if isinstance(mol, Chem.Mol):
            # Calculate the three descriptors
            clogp, num_aromatic_rings, num_rotatable_bonds = self.calculator.CalcDescriptors(mol)
            
            # Calculate AB-MPS score: abs(cLogP - 3) + NAR + NRB
            ab_mts_score = abs(clogp - 3) + num_aromatic_rings + num_rotatable_bonds
            
            row[self.out_column] = ab_mts_score
        else:
            raise ValueError(f"Seemingly not a Mol object: {mol} of type {type(mol)}")
        return row


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
class MolToInChI(RowLink):
    """Converts RDKit molecular objects to InChI strings or keys

    https://en.wikipedia.org/wiki/International_Chemical_Identifier

    Molecules are converted row-wise.

    Parameters
    ----------
    in_column
        The label for the column containing the molecules to convert
    out_column
        The label for the column that should store the SMILES strings
    generate_keys=False
        Whether to generate the key
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "InChI"
    generate_keys: bool =False

    def _row_apply(self, row: pd.Series) -> pd.Series:
        if self.generate_keys:
            row[self.out_column] = Chem.inchi.MolToInchiKey(row[self.in_column])
        else:
            row[self.out_column] = Chem.inchi.MolToInchi(row[self.in_column])
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
        A list of RDKit descriptors to calculate. If none provided or empty list, all available descriptors will be added.
    """

    in_column: InColumnName = "ROMol"
    descriptors: List[str] = field(
        default_factory=lambda: ["MolWt", "MolLogP", "NumHAcceptors", "NumHDonors"]
    )

    @classmethod
    def get_available_descriptors(cls) -> List[str]:
        """List of names of all available descriptors"""
        return [descriptor[0] for descriptor in Descriptors._descList]
    
    @property
    def available_descriptors(self) -> List[str]:
        """List of names of all available descriptors"""
        return self.get_available_descriptors()
    
    def __post_init__(self):
        super().__post_init__()
        # Validate descriptors and set up calculator
        available_descriptors = self.available_descriptors
        if self.descriptors:
            unknown_descriptors = [
                desc_name
                for desc_name in self.descriptors
                if desc_name not in available_descriptors
            ]
            if unknown_descriptors:
                raise ValueError(f"Unknown descriptor names {unknown_descriptors} specified. Available descriptors can be found with RDKitDescriptors.get_available_descriptors()")
        else:
            # If no descriptors specified, use all available ones
            self.descriptors = available_descriptors
        
        self.calculator = MolecularDescriptorCalculator(self.descriptors)
    #TODO, make the calculator object reinstantiated if descriptors are updated

    #TODO, check that we don't overwrite existing columns

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
class RemoveAtomMapping(RowLink):
    """Remove atommap information from the molecular object

    The out_column label can be the same as the in_column label,
    in which case the input molecules will get substituted with the converted ones

    Parameters
    in_column
        The label for the column containing the molecules to strip stereo information from
    out_column
        The label for the column that should store the converted molecules
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "ROMol"

    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol = copy.deepcopy(
            row[self.in_column]
        ) 
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0) #Setting for 0 removed it
        
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


@dataclass
class TanimotoSimilarity(RowLink):
    target_smiles: str
    in_column: InColumnName = "ROMol"
    out_column: str = "TanimotoSimilarity"
    radius: int = 2
    #TODO make different fingerprints optionally

    def __post_init__(self):
        super().__post_init__()
        self.fingerprinter = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.target_fingerprint = self.fingerprinter.GetFingerprint(Chem.MolFromSmiles(self.target_smiles))

    
    def _row_apply(self, row: pd.Series) -> pd.Series:
        mol_fp = self.fingerprinter.GetFingerprint(row[self.in_column])
        row[self.out_column]  = Chem.DataStructs.TanimotoSimilarity(mol_fp, self.target_fingerprint) #OBS consider directionality for other sim-metrics if refactored
        return row


@dataclass
class MolToFingerprint(RowLink):
    """Compute Morgan fingerprint as RDKit ExplicitBitVect

    Generates Morgan (circular) fingerprints from RDKit molecule objects.
    The fingerprint is stored as an RDKit ExplicitBitVect in a dunder column
    for consumption by downstream clustering/embedding links.

    Parameters
    ----------
    in_column
        Column containing RDKit molecule objects
    out_column
        Column to store the fingerprint objects
    radius
        Morgan fingerprint radius (1, 2, or 3)
    n_bits
        Length of the bit vector
    """

    in_column: InColumnName = "ROMol"
    out_column: str = "__MolFP__"
    radius: int = 2
    n_bits: int = 2048

    def __post_init__(self):
        super().__post_init__()
        if self.radius not in (1, 2, 3):
            import time
            self.logger.warning(
                f"Unusual Morgan radius={self.radius}. "
                f"Typical values are 1, 2, or 3. Continuing in 10 seconds..."
            )
            time.sleep(10)
        self.fingerprinter = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits
        )

    def _row_apply(self, row: pd.Series) -> pd.Series:
        row[self.out_column] = self.fingerprinter.GetFingerprint(row[self.in_column])
        return row


@dataclass
class DockingEfficiency(RowLink):
    """Size-normalized docking efficiency scores

    Computes three docking efficiency metrics that correct for the
    heavy-atom-count bias in docking scores (larger molecules tend to
    score better simply because they make more contacts).

    Metrics produced:
    - {out_prefix}_sub: score - slope * HA  (removes linear size trend)
    - {out_prefix}_per_ha: score / HA  (classic ligand efficiency)
    - {out_prefix}_ratio: score / (slope * HA + intercept)  (ratio to frontier)

    The slope and intercept describe the best-score frontier vs molecule
    size. Use DockingEfficiencyAnalyzer to fit these from your dataset,
    or supply known values.

    Parameters
    ----------
    score_column
        Column containing docking scores (more negative = better)
    ha_column
        Column containing heavy atom counts
    slope
        Frontier regression slope (negative for docking scores)
    intercept
        Frontier regression intercept
    out_prefix
        Prefix for output column names
    """

    score_column: InColumnName = "Glide (raw)"
    ha_column: InColumnName = "HeavyAtomCount"
    slope: float = -0.264
    intercept: float = -5.11
    out_prefix: str = "dock_eff"

    def _row_apply(self, row: pd.Series) -> pd.Series:
        score = float(row[self.score_column])
        ha = float(row[self.ha_column])

        row[f"{self.out_prefix}_sub"] = score - self.slope * ha
        row[f"{self.out_prefix}_per_ha"] = score / ha if ha > 0 else float('nan')

        frontier_value = self.slope * ha + self.intercept
        row[f"{self.out_prefix}_ratio"] = score / frontier_value if frontier_value != 0 else float('nan')

        return row


@dataclass
class DockingEfficiencyAnalyzer(Link):
    """Fit frontier regression and compute docking efficiency metrics

    Analyzes the relationship between docking scores and heavy atom count
    by fitting a linear frontier (best score per HA bin). Automatically
    detects the linear region by sweeping HA cutoffs and selecting the
    widest range with good R-squared.

    After fitting, access the configured scorer via the .docking_efficiency
    property, or inspect fitted parameters via sklearn-style trailing-underscore
    attributes (slope_, intercept_, r_squared_, ha_range_).

    Diagnostic plots require matplotlib (pip install matplotlib).

    Parameters
    ----------
    score_column
        Column containing docking scores (more negative = better)
    ha_column
        Column containing heavy atom counts
    ha_min
        Minimum HA for frontier fitting
    ha_max
        Maximum HA for frontier fitting. None = auto-detect.
    r_squared_threshold
        Minimum R-squared for the auto-detected linear region (fallback)
    max_r2_drop
        Maximum allowed R-squared drop between consecutive HA cutoffs.
        The auto-detection picks the last cutoff before R² drops by
        more than this amount in a single step.
    min_frontier_points
        Minimum number of HA bins required for fitting
    out_prefix
        Prefix for output efficiency column names
    plot_dir
        Directory for saving diagnostic plots. None = no file output.
    """

    _partitionable = Partitionable.NO

    score_column: InColumnName = "Glide (raw)"
    ha_column: InColumnName = "HeavyAtomCount"
    ha_min: int = 10
    ha_max: Optional[int] = None
    r_squared_threshold: float = 0.7
    max_r2_drop: float = 0.05
    min_frontier_points: int = 5
    out_prefix: str = "dock_eff"
    plot_dir: Optional[str] = None

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Build frontier: best (min) score per HA bin
        frontier = df.groupby(self.ha_column)[self.score_column].min()

        # Determine HA range
        if self.ha_max is not None:
            effective_ha_max = self.ha_max
        else:
            effective_ha_max = self._auto_detect_ha_max(frontier)

        # Select frontier points in range
        frontier_in_range = frontier.loc[
            (frontier.index >= self.ha_min) & (frontier.index <= effective_ha_max)
        ]

        if len(frontier_in_range) < self.min_frontier_points:
            self.logger.warning(
                f"Only {len(frontier_in_range)} frontier points in HA range "
                f"[{self.ha_min}, {effective_ha_max}]. "
                f"Need at least {self.min_frontier_points}. "
                f"Returning DataFrame without efficiency columns."
            )
            return df

        # Fit linear regression on frontier
        coeffs = np.polyfit(frontier_in_range.index.astype(float),
                            frontier_in_range.values.astype(float), 1)
        self.slope_ = float(coeffs[0])
        self.intercept_ = float(coeffs[1])

        # R-squared
        y_pred = self.slope_ * frontier_in_range.index + self.intercept_
        ss_res = np.sum((frontier_in_range.values - y_pred) ** 2)
        ss_tot = np.sum((frontier_in_range.values - frontier_in_range.values.mean()) ** 2)
        self.r_squared_ = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        self.ha_range_ = (int(self.ha_min), int(effective_ha_max))

        # Log results
        self.logger.info(
            f"Frontier fit: slope={self.slope_:.4f}, intercept={self.intercept_:.2f}, "
            f"R²={self.r_squared_:.4f}, HA range={self.ha_range_}"
        )
        self.logger.info(
            f"DockingEfficiency(score_column='{self.score_column}', "
            f"ha_column='{self.ha_column}', "
            f"slope={self.slope_:.6f}, intercept={self.intercept_:.6f})"
        )

        # Compute efficiency columns
        ha = df[self.ha_column].astype(float)
        score = df[self.score_column].astype(float)
        df[f"{self.out_prefix}_sub"] = score - self.slope_ * ha
        df[f"{self.out_prefix}_per_ha"] = score / ha.replace(0, float('nan'))
        frontier_values = self.slope_ * ha + self.intercept_
        df[f"{self.out_prefix}_ratio"] = score / frontier_values.replace(0, float('nan'))

        # Generate plots
        self._frontier = frontier
        self._frontier_in_range = frontier_in_range
        self._generate_plots(df)

        return df

    def _auto_detect_ha_max(self, frontier: pd.Series) -> int:
        """Find the largest HA cutoff that maintains a good linear fit."""
        all_ha = frontier.index[frontier.index >= self.ha_min].values

        if len(all_ha) < self.min_frontier_points:
            self.logger.warning(
                f"Only {len(all_ha)} HA bins above ha_min={self.ha_min}. Using all."
            )
            return int(all_ha[-1]) if len(all_ha) > 0 else self.ha_min

        min_cutoff = int(all_ha[self.min_frontier_points - 1])
        max_cutoff = int(all_ha[-1])

        r_squared_by_cutoff = {}
        for cutoff in range(min_cutoff, max_cutoff + 1):
            pts = frontier.loc[
                (frontier.index >= self.ha_min) & (frontier.index <= cutoff)
            ]
            if len(pts) < 2:
                continue
            coeffs = np.polyfit(pts.index.astype(float), pts.values.astype(float), 1)
            y_pred = coeffs[0] * pts.index + coeffs[1]
            ss_res = np.sum((pts.values - y_pred) ** 2)
            ss_tot = np.sum((pts.values - pts.values.mean()) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            r_squared_by_cutoff[cutoff] = r2

        self._r_squared_by_cutoff = r_squared_by_cutoff

        # Walk cutoffs in order, stop before a step drops R² by more than max_r2_drop
        cutoffs_sorted = sorted(r_squared_by_cutoff.keys())
        best = cutoffs_sorted[0]
        for i in range(1, len(cutoffs_sorted)):
            prev_r2 = r_squared_by_cutoff[cutoffs_sorted[i - 1]]
            curr_r2 = r_squared_by_cutoff[cutoffs_sorted[i]]
            if prev_r2 - curr_r2 > self.max_r2_drop:
                break
            best = cutoffs_sorted[i]

        # Check if the selected range has acceptable R²
        best_r2 = r_squared_by_cutoff[best]
        if best_r2 >= self.r_squared_threshold:
            self.logger.info(
                f"Auto-detected ha_max={best} (R²={best_r2:.4f})"
            )
        else:
            self.logger.warning(
                f"Best ha_max={best} has R²={best_r2:.4f} "
                f"(below threshold {self.r_squared_threshold}). "
                f"Consider adjusting ha_min or inspecting the score-vs-HA scatter."
        )
        return best

    def _generate_plots(self, df: pd.DataFrame):
        """Generate diagnostic plots. Requires matplotlib."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning(
                "matplotlib not installed — skipping plots. "
                "Install with: pip install matplotlib"
            )
            self._last_figure = None
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Score vs HA with frontier
        ax = axes[0]
        ax.scatter(df[self.ha_column], df[self.score_column],
                   alpha=0.2, s=8, color='steelblue', label='All molecules')
        ax.scatter(self._frontier.index, self._frontier.values,
                   color='orange', s=25, zorder=5, label='Best per HA')
        ax.scatter(self._frontier_in_range.index, self._frontier_in_range.values,
                   color='red', s=35, zorder=6, label='Fitted range')

        ha_line = np.array([self._frontier_in_range.index.min(),
                            self._frontier_in_range.index.max()])
        ax.plot(ha_line, self.slope_ * ha_line + self.intercept_,
                'r-', linewidth=2,
                label=f'fit: {self.slope_:.3f}*HA + {self.intercept_:.2f}')

        ax.set_xlabel('Heavy Atom Count')
        ax.set_ylabel(self.score_column)
        ax.set_title(f'Docking Score vs HA (R²={self.r_squared_:.3f})')
        ax.legend(fontsize=8)

        # Panel 2: R² vs HA cutoff
        ax = axes[1]
        if hasattr(self, '_r_squared_by_cutoff') and self._r_squared_by_cutoff:
            cutoffs = sorted(self._r_squared_by_cutoff.keys())
            r2_values = [self._r_squared_by_cutoff[c] for c in cutoffs]
            ax.plot(cutoffs, r2_values, 'o-', markersize=4, color='steelblue')
            ax.axhline(y=self.r_squared_threshold, color='red', linestyle='--',
                       alpha=0.7, label=f'threshold={self.r_squared_threshold}')
            ax.axvline(x=self.ha_range_[1], color='green', linestyle='--',
                       alpha=0.7, label=f'selected max_HA={self.ha_range_[1]}')
            ax.set_xlabel('HA cutoff (max)')
            ax.set_ylabel('R²')
            ax.set_title('Linear Fit Quality vs HA Range')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, f'HA max={self.ha_max} specified\n(no auto-detection)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Segment Selection (skipped)')

        plt.tight_layout()

        if self.plot_dir:
            os.makedirs(self.plot_dir, exist_ok=True)
            filepath = os.path.join(self.plot_dir, 'docking_efficiency_diagnostic.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Diagnostic plot saved to {filepath}")

        self._last_figure = fig
        plt.close(fig)

    def _repr_png_(self):
        """Jupyter rich display: show the last diagnostic plot inline."""
        if not hasattr(self, '_last_figure') or self._last_figure is None:
            return None
        import io
        buf = io.BytesIO()
        self._last_figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return buf.read()

    @property
    def docking_efficiency(self) -> DockingEfficiency:
        """Return a configured DockingEfficiency link with fitted parameters.

        Raises AttributeError if the analyzer has not been applied yet.
        """
        if not hasattr(self, 'slope_'):
            raise AttributeError(
                "DockingEfficiencyAnalyzer has not been fitted yet. "
                "Call .apply(df) first."
            )
        return DockingEfficiency(
            score_column=self.score_column,
            ha_column=self.ha_column,
            slope=self.slope_,
            intercept=self.intercept_,
            out_prefix=self.out_prefix,
        )