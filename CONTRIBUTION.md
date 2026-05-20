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

## Dataclass pitfall: `@property` does not work

All links are `@dataclass` classes. If you need to validate or intercept attribute assignment (e.g. validating a list of allowed values), **do not use `@property`** — the property descriptor replaces the dataclass field default, breaking the generated `__init__`.

Instead, use `__setattr__` to intercept specific fields:

```python
@dataclass
class MyLink(RowLink):
    calculations: List[str] = field(default_factory=lambda: ["default"])

    def __post_init__(self):
        super().__post_init__()
        self._validate_calculations(self.calculations)

    def __setattr__(self, name, value):
        if name == "calculations" and hasattr(self, "logger"):
            self._validate_calculations(value)
        super().__setattr__(name, value)

    def _validate_calculations(self, calcs):
        # your validation here
        pass
```

The `hasattr(self, "logger")` guard skips validation during `__init__` (before the logger is set up by `__post_init__`). The explicit `__post_init__` call handles the initial validation once the logger exists.
