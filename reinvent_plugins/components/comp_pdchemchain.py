"""REINVENT scoring component backed by pdchemchain pipelines.

Runs a pdchemchain YAML pipeline and returns selected columns as scoring
endpoints.  The chain is loaded once at initialization; each ``__call__``
creates a fresh DataFrame from the incoming SMILES, runs the chain, and
extracts the requested score columns.

Multiple endpoints share a **single chain execution** -- expensive steps
like docking are not repeated per endpoint.

.. note::

   Chains that expand rows (e.g. LigPrep enumeration) **must** include an
   aggregation step (``AggregateByScore`` / ``GroupPick``) to collapse the
   result back to one row per input SMILES.

Example TOML configuration::

    [[scoring.component]]
    [scoring.component.PdChemChain]

    # smiles_column defaults to "SMILES" (matching REINVENT output)
    [[scoring.component.PdChemChain.endpoint]]
    name = "Docking Score"
    weight = 1
    params.config_file = "/path/to/docking_pipeline.yaml"
    params.score_column = "docking_score"
    params.smiles_column = "SMILES"
    transform.type = "reverse_sigmoid"
    transform.high = -6
    transform.low = -13.5

    [[scoring.component.PdChemChain.endpoint]]
    name = "LogP"
    weight = 0.5
    params.config_file = "/path/to/docking_pipeline.yaml"
    params.score_column = "MolLogP"
    transform.type = "double_sigmoid"
    transform.high = 5
    transform.low = 0
"""

__all__ = ["PdChemChain"]

import logging
from typing import List

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the pdchemchain scoring component.

    All parameters are lists because REINVENT collects one value per endpoint.
    Component-level settings (``config_file``, ``smiles_column``,
    ``default_score``) are taken from the first element; ``score_column``
    determines the number of endpoints.

    Parameters
    ----------
    config_file
        Path to a pdchemchain YAML/JSON chain configuration.
    score_column
        DataFrame column(s) to extract as scores -- one per endpoint.
    smiles_column
        Name of the column used to feed SMILES into the chain.
    default_score
        Value substituted for NaN / errored rows.  Leave as NaN to let
        REINVENT transforms handle failures.
    """

    config_file: List[str]
    score_column: List[str]
    smiles_column: List[str] = Field(default_factory=lambda: ["SMILES"])
    default_score: List[float] = Field(default_factory=lambda: [float("nan")])


@add_tag("__component")
class PdChemChain:
    """REINVENT scoring component that delegates to a pdchemchain pipeline.

    The chain is loaded once from YAML in ``__init__`` and reused across
    calls.  Each ``__call__`` builds a DataFrame, runs the full chain
    **once**, then extracts the requested ``score_column`` values as
    separate endpoint arrays.
    """

    def __init__(self, params: Parameters):
        self.score_columns = params.score_column
        self.smiles_column = params.smiles_column[0]
        self.default_score = params.default_score[0]
        self.smiles_type = "rdkit_smiles"
        self.number_of_endpoints = len(params.score_column)

        config_file = params.config_file[0]

        from pdchemchain.io_utilities import load_chain

        self.chain = load_chain(config_file)

        logger.info(
            "PdChemChain: loaded chain from %s, extracting %d endpoint(s): %s",
            config_file,
            self.number_of_endpoints,
            ", ".join(self.score_columns),
        )

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> ComponentResults:
        import pandas as pd
        from pdchemchain.errormanager import rows_with_errors

        n = len(smilies)
        use_default = not np.isnan(self.default_score)

        # Guard against empty input (all SMILES filtered by @normalize_smiles)
        if n == 0:
            fill = self.default_score if use_default else np.nan
            return ComponentResults(
                [np.full(0, fill) for _ in self.score_columns]
            )

        df = pd.DataFrame({self.smiles_column: smilies})
        result_df = self.chain(df)

        error_mask = np.array(rows_with_errors(result_df, aslist=True))

        scores = []
        for col in self.score_columns:
            if col in result_df.columns:
                values = result_df[col].to_numpy(dtype=float, na_value=np.nan)
            else:
                logger.warning(
                    "PdChemChain: score column '%s' not found in chain output "
                    "(available: %s). Returning NaN.",
                    col,
                    ", ".join(result_df.columns.tolist()),
                )
                values = np.full(n, np.nan)

            if use_default:
                values[error_mask] = self.default_score
                values = np.where(np.isnan(values), self.default_score, values)

            scores.append(values)

        return ComponentResults(scores)
