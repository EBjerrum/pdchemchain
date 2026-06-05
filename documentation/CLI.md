# Command Line Reference

pdchemchain includes a CLI for running saved pipelines on data files. It supports both CSV and SDF formats with automatic format detection.

**Important:** The CLI does not (yet) automatically adapt the input data to match what the pipeline expects. If your pipeline's first link requires an `ROMol` column (common for chemistry links), you must either provide SDF input (which includes `ROMol`) or include a `MolFromSmiles` link at the start of the pipeline for CSV input. Design your pipeline to match the expected input format.

## Basic Syntax

```bash
pdchemchain run <config_file> [OPTIONS]
```

Where `config_file` is a YAML or JSON file containing your saved pipeline configuration.

## Quick Examples

```bash
# Process CSV file
pdchemchain run pipeline.yaml --in_file input.csv --out_file output.csv

# Process SDF file (auto-detected from extension)
pdchemchain run pipeline.yaml --in_file molecules.sdf --out_file results.sdf

# Convert between formats
pdchemchain run pipeline.yaml --in_file molecules.sdf --out_file results.csv

# With error file for failed rows
pdchemchain run pipeline.yaml --in_file input.sdf --out_file output.sdf --error_file errors.sdf
```

## Creating Pipeline Configuration Files

Save your interactive pipeline to a config file:

```python
from pdchemchain.links import MolFromSmiles, HeavyAtomCount

chain = MolFromSmiles() + HeavyAtomCount()
chain.to_config_file('pipeline.yaml')
```

Then run from command line:

```bash
pdchemchain run pipeline.yaml --in_file molecules.csv --out_file results.csv
```

## File Format Auto-Detection

The CLI automatically detects file formats from extensions:
- **SDF format**: `.sdf`, `.sd`
- **CSV format**: `.csv`, `.tsv`, `.txt`, and any other extension

You can override auto-detection using `--in_format` and `--out_format` flags:

```bash
pdchemchain run pipeline.yaml --in_file data.txt --in_format csv --out_file results.dat --out_format sdf
```

## CSV Separator Detection

For CSV files, the separator is auto-detected by pandas when not specified:

```bash
# Auto-detect separator (comma, tab, etc.)
pdchemchain run pipeline.yaml --in_file data.csv --out_file results.csv

# Specify separator explicitly
pdchemchain run pipeline.yaml --in_file data.tsv --sep "\t" --out_file results.csv
```

## SDF File Handling

### Molecule Column Convention

SDF files use the **"ROMol"** column name by default for molecule objects. If your pipeline uses a different column name, specify it:

```bash
pdchemchain run pipeline.yaml --in_file mols.sdf --mol_column Molecule --out_file results.sdf
```

### Designing Reusable Pipelines

Pipelines saved as YAML work best when they make no assumptions about the input file — no hardcoded filenames, and no dependence on a specific file format. The recommended pattern is to **design pipelines that expect an `ROMol` column** rather than a `Smiles` column. This makes the same YAML usable with both CSV and SDF inputs via the CLI.

**Best practice — pipeline expects ROMol:**

```yaml
# pipeline.yaml — no FromFile/FromSDF inside, just processing logic
__class__: pdchemchain.base.Chain
links:
  - __class__: pdchemchain.links.chemistry.RDKitDescriptors
    in_column: ROMol
    ...
```

```bash
# Run on SDF — ROMol comes directly from the file
pdchemchain run pipeline.yaml --in_file compounds.sdf --out_file results.csv

# Run on CSV — generate ROMol from the Smiles column before running
pdchemchain run pipeline.yaml --in_file compounds.csv --mol_from_smiles_column Smiles --out_file results.csv
```

**Other valid pipeline designs:**

Pipelines can also embed file paths or expect a `Smiles` column — these are valid for specific use cases:

```yaml
# Self-contained pipeline with hardcoded input
__class__: pdchemchain.base.Chain
links:
  - __class__: pdchemchain.links.io.FromFile
    filename: /data/compounds.csv
    mol_from_smiles_column: Smiles
  - __class__: pdchemchain.links.chemistry.RDKitDescriptors
    ...
```

The `ROMol`-first design is simply the most interoperable — the same pipeline file works unchanged across file formats, and the file-loading details stay at the CLI level where they belong.

**Generating SMILES from an SDF:** if your pipeline needs a `Smiles` column derived from the loaded molecules (e.g. for canonicalization), add `MolToSmiles` as the first link in the chain rather than relying on a property stored in the SDF.

### Handling None/Failed Molecules

When processing molecules, some may fail to parse or convert (e.g., invalid SMILES). pdchemchain handles these gracefully when writing to SDF:

**Automatic placeholder substitution:**
- Failed molecules (None values) are replaced with **error placeholder molecules**
- Placeholders are wildcard atoms (`*`) with:
  - MolWt = 0.00 (clearly distinguishable)
  - Atom label = "ERROR" (visible in molecular viewers)
  - Molecule name = "ERROR_NO_MOLECULE"

**Example workflow:**

```python
from pdchemchain.links import MolFromSmiles, ToSDF
import pandas as pd

df = pd.DataFrame({
    'Smiles': ['C', 'CC', 'INVALID_SMILES', 'CCC'],
    'name': ['methane', 'ethane', 'failed', 'propane']
})

# ToSDF with handle_none=True replaces None molecules with placeholders
link = MolFromSmiles() + ToSDF('output.sdf', handle_none=True)
link(df)
```

**Round-trip support:**

```python
from pdchemchain.links import FromSDF

# Load and recognize placeholders, converting back to None
df = FromSDF('output.sdf', recognize_placeholders=True)()

# df now has None for molecules that failed originally
```

## Error File Handling

Use `--error_file` to save rows that encountered errors during processing:

```bash
pdchemchain run pipeline.yaml --in_file input.csv --out_file output.sdf --error_file errors.sdf
```

**Error file features:**
- Automatically includes error messages in the output
- For SDF format: Failed molecules use placeholder molecules (see above)
- For CSV format: Error tracebacks written directly
- `__error__` column renamed to `ERROR` in SDF (RDKit compatibility)

## Advanced Options

```bash
# Pass pandas read options
pdchemchain run pipeline.yaml --in_file data.csv --pd_read_option header=0 --pd_read_option encoding=utf-8

# Pass pandas write options
pdchemchain run pipeline.yaml --in_file input.csv --out_file output.csv --pd_write_option index=False

# SDF-specific options (prefix with sdf_)
pdchemchain run pipeline.yaml --in_file mols.sdf --pd_read_option sdf_removeHs=False

# Set debug level
pdchemchain run pipeline.yaml --in_file input.csv --debug_level DEBUG

# Use custom link definitions
pdchemchain run pipeline.yaml --in_file input.csv --custom_links my_links.py
```

## Full Option Reference

```bash
pdchemchain run --help
```

| Option | Description |
|--------|-------------|
| `--in_file PATH` | Input file path (CSV or SDF) |
| `--out_file PATH` | Output file path (CSV or SDF) |
| `--in_format [csv\|sdf]` | Override input format detection |
| `--out_format [csv\|sdf]` | Override output format detection |
| `--mol_column TEXT` | Molecule column name for SDF files (default: "ROMol") |
| `--mol_from_smiles_column TEXT` | Generate ROMol from named SMILES column after loading (CSV or SDF) |
| `--error_file PATH` | Error file path for failed rows |
| `--sep TEXT` | CSV separator (default: auto-detect) |
| `--pd_read_option TEXT` | Extra pandas read options (keyword=value, repeatable) |
| `--pd_write_option TEXT` | Extra pandas write options (keyword=value, repeatable) |
| `--debug_level TEXT` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `--custom_links PATH` | Python file with custom link definitions |
