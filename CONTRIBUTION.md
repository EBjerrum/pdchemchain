# Developer notes

As a framework for links, an obvious way to contribute is to submit new links developed for your needs and workflows. Bug reports and fixes are also much welcome. For more fundamental challenges, please reach out for a discussion. If you want to get more involved, there's a lot of other improvements that are in the pipeline, reach out for a discussion.

## Docstrings

Follow Numpy style guide: [https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)
As we strive to have type-hints, it's not necessary to repeat the type in the parameters.

Example:

```lang=python
    """Example function with PEP 484 type annotations.

    The return type must be duplicated in the docstring to comply
    with the NumPy docstring style.

    Parameters
    ----------
    param1
        The first parameter.
    param2
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.

    """
```
