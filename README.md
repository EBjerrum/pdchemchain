# pdchemchain

## Chainable links for chemical processing of Pandas dataframes

pdchemchain is a framework for chainable pandas DataFrame manipulations, designed for interactive notebook use and command-line deployment. Built for chemistry via RDKit, but domain-agnostic at its core. All links are self-documenting, auto-configurable, and serializable to YAML/JSON.

## Quick Start

```python
from pdchemchain.links import MolFromSmiles, HeavyAtomCount, RDKitDescriptors
import pandas as pd

df = pd.DataFrame({"Smiles": ["c1ccccc1", "CCO", "CC(=O)O"]})

# Build a pipeline interactively
chain = MolFromSmiles() + HeavyAtomCount() + RDKitDescriptors(descriptors=["MolLogP", "TPSA"])
df_out = chain(df)

# Save for reuse from the command line
chain.to_config_file("pipeline.yaml")
```

```bash
pdchemchain run pipeline.yaml --in_file molecules.csv --out_file results.csv
```

## Installation

From GitHub:
```bash
pip install git+https://github.com/EBjerrum/pdchemchain.git
```

Developer installation:
```bash
git clone git@github.com:EBjerrum/pdchemchain.git
cd pdchemchain
pip install -e .[dev]
```

## Documentation

- **[Introduction](documentation/Introduction.md)** — API concepts, code examples, and diagrams
- **[CLI Reference](documentation/CLI.md)** — full command-line usage
- **[Example Notebooks](documentation/notebooks/)** — end-to-end workflows
- **[Contribution Guide](CONTRIBUTION.md)** — how to create and contribute new links

## Command Line Usage

Saved pipelines run directly from the command line with CSV and SDF support:

```bash
# Process CSV
pdchemchain run pipeline.yaml --in_file input.csv --out_file output.csv

# Process SDF (auto-detected from extension)
pdchemchain run pipeline.yaml --in_file molecules.sdf --out_file results.sdf

# With error file for failed rows
pdchemchain run pipeline.yaml --in_file input.sdf --out_file output.sdf --error_file errors.sdf
```

See the [CLI Reference](documentation/CLI.md) for format detection, SDF handling, advanced options, and full option reference.

## Contributions

See [CONTRIBUTION.md](CONTRIBUTION.md) for how to subclass Link/RowLink and create new links. The framework is designed for minimal boilerplate — a new link is typically 10-15 lines of code.
