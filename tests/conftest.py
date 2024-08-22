import pytest
import pandas as pd

from pdchemchain.links.dataframe import NullLink
from rdkit import Chem


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "int1": [1, 2, 3],
            "int2": [2, 3, 4],
            "letters1": ["a", "b", "c"],
            "duplicates": [1, 2, 1],
            "ROMol": [
                Chem.MolFromSmiles("C"),
                Chem.MolFromSmiles("CN"),
                Chem.MolFromSmiles("OCN"),
            ],
            "Smiles": ["C", "CN", "OCN"],
        },
    )


@pytest.fixture
def testlink():
    return NullLink(name="test1")


@pytest.fixture
def testlink2():
    return NullLink(name="test2")


@pytest.fixture(scope="session")
def csv_filename(tmp_path_factory):
    sample_dataframe = pd.DataFrame(
        {
            "int1": [1, 2, 1],
            "int2": [2, 3, 4],
            "letters1": ["a", "b", "c"],
            "ROMol": [
                Chem.MolFromSmiles("C"),
                Chem.MolFromSmiles("CN"),
                Chem.MolFromSmiles("OCN"),
            ],
            "Smiles": ["C", "CN", "OCN"],
        },
    )  # For some reason, cant access a function scoped fixture from a session scoped fixture, and cant make the sample_dataframe session scoped
    fn = tmp_path_factory.mktemp("data") / "data.csv"
    sample_dataframe.to_csv(fn)
    return fn
