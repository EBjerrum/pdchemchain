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
from pdchemchain import LinkToolBox
toolbox = LinkToolBox()
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
- Certain names for columns are not allowed as they are needed for internal usage, so avoid dunder column names starting and ending with double underscores e.g. `__error__`
- May not be able to build all types of fully directed acyclic graph. I'm not 100% sure if the chain creation dogma and current links allow for all types of directed acyclic graph to be defined, but it has worked for my use-cases so far.

## Installation

```bash
pip install git+https://github.com/EBjerrum/pdchemchain.git
```

## Documentation

Currently only a couple of notebook tutorials are available, be sure to look through the code for already created Links before you implement your own

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
