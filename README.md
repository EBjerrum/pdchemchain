# pdchemchain

## Chainable links for chemical processing of Pandas dataframes

pdchemchain is a framework for reusable manipulations of pandas dataframes for both interactive and command line usage. The framework was made for chemistry actions via RDKit, but can easily be used for other domains as well, as the auto-configuable and self-documenting features of the framework are quite nice and versatile. The project tries to keep both interactive notebook usage as well as config-file usage in mind, and fully supports saving interactively developed pipelines for on-disk usage via config files. It has inbuild error handling for exceptions, and a custom context class that are useful to catch RDKit error messages on stderr. The framework has been written in a way that requieres minimal code to create a new custom Link for the chain by subclassing abstract baseclasses that provide the common functionality.

## API Dogma

- **API Dogma 1:** Pandas in, Pandas out

```python
df_out = link(df_in)
```

Links are subclasses of the abstract Link class, and when instantiated they will return a dataframe when called with a dataframe either directly or via the `.apply()` method.

- **API Dogma 2:** A chain is also a link

```python
chain = link1 + link2
```

Adding two links together creates a new Link of the Chain subclass. The chain can be tested interactively and is easily expanded with new links e.g. `chain = chain + link3`. Many links can be added directly to the Chain instantiation or simply by summing as list with the links `chain = sum([link1,link2,link3])`. More advanced links allow for parallel chains (`UnionLink`), as well as memory and performance optimization via partitioned processed (`SerialPartitionProcessor` and `ParallelPartitionProcessor`)

- **API Dogma 3:** All links (and thus also chains etc.) are self-documenting and auto-configurable

```python
params = link.get_params()
cloned_link = Link.from_params(params)
```

via the `.get_params()` method, a dictionary with all information to recreate the link is returned. This dictionary are easy to edit/save/load from JSON and Yaml for reuse from the command line. Parameters of nested links are also nested in the dictionary returned, and nested links will be recreated with the `.from_params()` method, so that chains and other nested links are recreated correctly.

## Toolbox

For easy referal in interactive mode, a toolbox class is available, that gives an overview of the current classes and hints on their usage.

```python
from pdchemchain import toolbox
toolbox
```

|                                  | Tooltip                                                                                               | Api                                                                                                                                                |
| :------------------------------- | :---------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| ('base', 'Chain')                | Runs links sequentially, one after the other and return the processed dataframe.                      | Chain(links: 'Union[List[Link], Tuple[Link]]')                                                                                                     |
| ('base', 'Link')                 | Base class for all Links                                                                              | Link()                                                                                                                                             |
| ('base', 'RowLink')              | Base class for all links that process dataframes row-by-row                                           | RowLink()                                                                                                                                          |
| ('base', 'UnionLink')            | Runs the dataframe through two seperate links and merges the result two dataframes into a single one. | UnionLink(link1: 'Link', link2: 'Link')                                                                                                            |
| ('chemistry', 'ElementsInList')  | Checks if a given molecule only has certain elements                                                  | ElementsInList(in_column: pdchemchain.typing.InColumnName = 'ROMol', out_column: str = 'ElementsAllowed', allowed_elements: List[int] = <factory>) |
| ('chemistry', 'HeavyAtomCount')  | Counts the number of heavy atoms                                                                      | HeavyAtomCount(in_column: pdchemchain.typing.InColumnName = 'ROMol', out_column: str = 'HeavyAtomCount')                                           |
| ('chemistry', 'HeteroAtomRatio') | Calculates the ratio of heteroatom to heavy atoms                                                     | HeteroAtomRatio(in_column: pdchemchain.typing.InColumnName = 'ROMol', out_column: str = 'HeteroAtomRatio')                                         |
|                                  |

...

(The table looks better in ipython repr, but you get the idea I hope)

## Pro's and Con's

Pros

- Simple and fast building for interactive usage in e.g. Jupyter notebooks
- Configurable command line usage via JSON or Yaml files
- Interactive and configurable are interchangable
- Easy to extend with new Link classes

Cons:

- The simplistic pipeline creation and dogmatic API gives some restrictions, as example the pipeline can't return both a dataframe and a dataframe with errors. Errors on single rows are marked
- The framework are aimed for in_memory usage, so very large dataframes can give issues as pandas usually works on copies. The `SerialPartitionProcesser` wrapper link can significantly reduce memory issues, and links like `DropColumns` or `KeepColumns` can also be used to reduce dataframe size.
- Most links work row by row, and cross-row calculations may not be fully supported (e.g. if using partitioning links). Aggregation of columns values or grouping operations will not necessarely be fully compatible with all other links.
- Certain names for columns are not allowed as they are needed for internal usage, so avoid dunder column names starting and ending with double underscores. Reserved dunder columns (automatically excluded from SDF export by `ToSDF`):

  | Column | Created By | Purpose |
  |--------|-----------|---------|
  | `__error__` | Error handlers | Exception/failure messages |
  | `__log__` | RowLogger | Row-level log messages |
  | `__id__` | ExpandableRowLink, LigPrep, GlideDock | Original compound tracking |
  | `__enum_id__` | ExpandableRowLink, LigPrep, GlideDock | Variant/pose sequence number |
  | `__MolFP__` | MolToFingerprint | Fingerprint objects (intermediate) |
  | `__ROMolLigPrep__` | LigPrep | 3D prepared molecule from LigPrep |
  | `__ROMolDocked__` | GlideDock | Docked pose from Glide |
- May not be able to build all types of fully directed acyclic graph. I'm not 100% sure if the chain creation dogma and current links allow for all types of directed acyclic graph to be defined, but it has worked for my use-cases so far.

## Installation

Installation from open-source version:
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

Currently only a couple of notebook tutorials are available, be sure to look through the code for already created Links before you implement your own

## Command Line Usage

pdchemchain includes a powerful CLI for running saved pipelines on data files. The CLI supports both CSV and SDF (Structure Data File) formats with automatic format detection.

### Basic Syntax

```bash
pdchemchain run <config_file> [OPTIONS]
```

Where `config_file` is a YAML or JSON file containing your saved pipeline configuration.

### Quick Examples

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

### File Format Auto-Detection

The CLI automatically detects file formats from extensions:
- **SDF format**: `.sdf`, `.sd`
- **CSV format**: `.csv`, `.tsv`, `.txt`, and any other extension

You can override auto-detection using `--in_format` and `--out_format` flags:

```bash
# Override for unusual extensions
pdchemchain run pipeline.yaml --in_file data.txt --in_format csv --out_file results.dat --out_format sdf
```

### CSV Separator Detection

For CSV files, the separator is auto-detected by pandas when not specified:

```bash
# Auto-detect separator (comma, tab, etc.)
pdchemchain run pipeline.yaml --in_file data.csv --out_file results.csv

# Specify separator explicitly
pdchemchain run pipeline.yaml --in_file data.tsv --sep "\t" --out_file results.csv
```

### SDF File Handling

#### Molecule Column Convention

SDF files use the **"ROMol"** column name by default for molecule objects. If your pipeline uses a different column name, specify it:

```bash
pdchemchain run pipeline.yaml --in_file mols.sdf --mol_column Molecule --out_file results.sdf
```

#### Handling None/Failed Molecules

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

# Some SMILES will fail to parse
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

### Error File Handling

Use `--error_file` to save rows that encountered errors during processing:

```bash
# Error file format auto-detected (same as output)
pdchemchain run pipeline.yaml --in_file input.csv --out_file output.sdf --error_file errors.sdf
```

**Error file features:**
- Automatically includes error messages in the output
- For SDF format: Failed molecules use placeholder molecules (see above)
- For CSV format: Error tracebacks written directly
- `__error__` column renamed to `ERROR` in SDF (RDKit compatibility)

### Advanced Options

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

### Full CLI Reference

```bash
pdchemchain run --help
```

**Key options:**
- `--in_file PATH`: Input file path (CSV or SDF)
- `--out_file PATH`: Output file path (CSV or SDF)
- `--in_format [csv|sdf]`: Override input format detection
- `--out_format [csv|sdf]`: Override output format detection
- `--mol_column TEXT`: Molecule column name for SDF files (default: "ROMol")
- `--error_file PATH`: Error file path for failed rows
- `--sep TEXT`: CSV separator (default: auto-detect)
- `--pd_read_option TEXT`: Extra pandas read options (keyword=value, repeatable)
- `--pd_write_option TEXT`: Extra pandas write options (keyword=value, repeatable)
- `--debug_level TEXT`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--custom_links PATH`: Python file with custom link definitions

### Creating Pipeline Configuration Files

Save your interactive pipeline to a config file:

```python
from pdchemchain.links import MolFromSmiles, HeavyAtomCount

# Build pipeline interactively
chain = MolFromSmiles() + HeavyAtomCount()

# Save to config file
chain.to_config_file('pipeline.yaml')
```

Then run from command line:

```bash
pdchemchain run pipeline.yaml --in_file molecules.csv --out_file results.csv
```

## Custom links

There are two link abstract classes to base a new link of, the Link class and the RowLink class. The link class are useful for links that handle the whole of the dataframe, whereas the RowLink are a nase for links that are working on a dataframe row by row. With the Link class it's needed to make manual handling of errors in rows, whereas that is included in the RowLink class (except for some corner cases, where errors doesn't raise exceptions). It is recommended to use the RowLink class as a first choice.

The new link must be a dataclass, which saves boilerplate-code in assigning keywords to attributes as well as making sure that keywords and properties are named the same which makes the autoconfigurable code work. Below is an example link that takes the content of a column X and add the predicted Y to a new column using a configurable linear equation.

```python
from dataclasses import dataclass
from pandaschain import RowLink
from pandaschain.typing import InColumnName #InColumn name is a custom type that makes the link assert that a column with that name is found.
import pandas as pd

@dataclass
class LinearModel(RowLink):
    in_column: InColumnName = 'x'
    slope: float
    bias: float
    out_column: str = 'y'

    def _row_apply(self, row: pd.Series) -> pd.Series:
        row[self.out_column] = self.slope * row[self.in_column] + self.bias
        return row
```

The private \_row_apply() method are then used by the links **call** or .apply() method. If the code in the \_row_apply method fails, or you raise a custom exception, that will be catched and handled (i.e. the exception will be added to the **error** column).

## Contributions

There are more information about how to subclass the Link class and create your own links in the [CONTRIBUTION.md](CONTRIBUTION.md) file

## Examples

Notebooks with examples of usage will be put in [documentation/notebooks](documentation/notebooks).
