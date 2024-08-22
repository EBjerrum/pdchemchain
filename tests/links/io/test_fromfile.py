import tempfile
from pdchemchain.links import FromFile

from ...basetest import BaseTest

import pandas as pd
from rdkit import Chem

import os, pytest


class TestToFile(BaseTest):
    _Link = FromFile

    @pytest.fixture
    def csv_filename(self, tmpdir):
        df = pd.DataFrame(
            pd.DataFrame(
                {
                    "int1": [1, 2, 1],
                    "int2": [2, 3, 4],
                    "letters1": ["a", "b", "c"],
                    "Smiles": ["C", "CN", "OCN"],
                },
            )
        )
        filename = tmpdir + "/load_test1.csv"
        df.to_csv(filename)

        return filename

    @pytest.fixture
    def classparams(self, csv_filename):
        return {"filename": csv_filename, "pd_readcsv_options": {"sep": ","}}

    @pytest.fixture
    def alt_classparams(self, tmpdir):
        return {
            "filename": tmpdir
            + "load_test2.csv",  # This is never attempted loaded, so no need for it to exist
            "pd_readcsv_options": {"sep": "."},
        }
