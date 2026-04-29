# REINVENT Integration Example

End-to-end example of using pdchemchain as a scoring component in REINVENT
reinforcement learning. The pipeline calculates physicochemical properties
and the RL agent learns to generate molecules with zero hydrogen bond donors.

## Prerequisites

- REINVENT4 installed (`reinvent` CLI available)
- pdchemchain installed in the same environment (`pip install -e .`)
- Path to `reinvent.prior` (ships with REINVENT at `priors/reinvent.prior`)

## Workflow

Run from this directory. Adjust the prior model path in the TOML files first.

### Step 1: Sample from the prior

```bash
reinvent 01_sample_prior.toml
```

Generates `prior_sample.csv` (~1000 SMILES) for interactive pipeline development.

### Step 2: Build the scoring pipeline

Open `02_build_scoring_pipeline.py` in VS Code / Jupyter and run cell-by-cell.
This builds a `MolFromSmiles + RDKitDescriptors` chain, tests it interactively,
and saves `scoring_pipeline.yaml`.

### Step 3: Run reinforcement learning

```bash
reinvent 03_staged_learning.toml
```

Runs staged learning with three PdChemChain scoring endpoints (single chain
execution) plus a zero-weight monitor endpoint:

| Endpoint       | Weight | Target                          |
|----------------|--------|---------------------------------|
| No HBD         | 0.6    | NumHDonors = 0 (reverse sigmoid)|
| LogP           | 0.2    | MolLogP in [1, 4]               |
| TPSA           | 0.2    | TPSA in [40, 120]               |
| HBA (monitor)  | 0      | Logged only, not scored         |

### Step 4: Analyse results

Open `04_analyse_results.py` and run cell-by-cell. Shows:
- Score progression with fraction of zero-HBD molecules over steps
- Prior vs trained agent property distributions
- Top-scoring molecules
