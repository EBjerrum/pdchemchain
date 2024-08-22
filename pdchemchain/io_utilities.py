from __future__ import (
    annotations,
)

# Necessary to fix pseudo circular import (due to use of typehinting)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdchemchain.base import Link

import json
import os
from multiprocessing import current_process

import pandas as pd
import yaml

import pdchemchain


def save_as_yaml(data: dict, filename: str) -> None:
    assert filename.lower().endswith(
        (".yaml", ".yml")
    ), "Invalid file extension for YAML"
    with open(filename, "w") as file:
        yaml.dump(data, file)


def save_as_json(data: dict, filename: str) -> None:
    assert filename.lower().endswith(".json"), "Invalid file extension for JSON"
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)


def save_dict(data: dict, filename: str = None) -> None:
    if not filename:
        filename = "Output.yaml"
    if filename.lower().endswith(".json"):
        save_as_json(data, filename)
    elif filename.lower().endswith((".yaml", ".yml")):
        save_as_yaml(data, filename)
    else:
        raise ValueError("Unknown file extension")


def save_chain(
    link: Link, filename: str = None, defaults=True, version=True, log_level=True
) -> None:
    save_dict(
        link.get_params(defaults=defaults, version=version, log_level=log_level),
        filename,
    )


def load_yaml(filename: str) -> dict:
    assert filename.lower().endswith(
        (".yaml", ".yml")
    ), "Invalid file extension for YAML"
    with open(filename, "r") as file:
        return yaml.safe_load(file)


def load_json(filename: str) -> dict:
    assert filename.lower().endswith(".json"), "Invalid file extension for JSON"
    with open(filename, "r") as file:
        return json.load(file)


def load_dict(filename: str) -> dict:
    if filename.lower().endswith((".yaml", ".yml")):
        return load_yaml(filename)
    elif filename.lower().endswith(".json"):
        return load_json(filename)
    else:
        raise ValueError("Unsupported file extension")


def load_chain(filename):
    data = load_dict(filename)
    if "__class__" in data:
        return pdchemchain.base.Link.from_params(
            data
        )  # importing and using Link directly leads to circular import
    else:
        raise ValueError("Failed to find Link or Chain in loaded data.")


def df_process_to_csv(df: pd.DataFrame, filename: str, **kwargs) -> "str | None":
    """
    Write a DataFrame to a CSV file, with the option to append the process ID to the filename
    when running in a subprocess.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be written to the CSV file.

    filename : str
        The filename of the CSV file.

    **kwargs : dict
        Additional keyword arguments to be passed to the internal Pandas `to_csv` method (see help(pd.DataFrame().to_csv)).

    Returns
    -------
    str | None
        If `path_or_buf` is None, the CSV data as a string; otherwise, returns None.

    Notes
    -----
    If the function is called from a subprocess (not the main process), the process ID (pid)
    is appended to the filename to distinguish output files from different subprocesses.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df_process_to_csv(df, 'output.csv', sep=';', index=False)
    """
    cur_proc = current_process()

    if cur_proc.name != "MainProcess":
        pid = cur_proc.pid
        base_path, ext = os.path.splitext(filename)
        filename = f"{base_path}_p{pid}{ext}"

    df.to_csv(path_or_buf=filename, **kwargs)
