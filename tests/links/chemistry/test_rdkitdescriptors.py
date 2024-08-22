from pdchemchain.links import RDKitDescriptors
from ...basetest import BaseErrorTest
import pytest
from rdkit import Chem


class TestRDKitDescriptors(BaseErrorTest):
    _Link = RDKitDescriptors
    _classparams = {
        "descriptors": ["MolWt", "MolLogP", "NumHAcceptors"],
        "in_column": "ROMol",
    }
    _alt_classparams = {
        "descriptors": ["MolWt", "MolLogP", "NumHAcceptors", "NumHDonors"],
        "in_column": "ROMol2",
    }

    def test_rdkitdescriptors(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert "MolWt" in df_o

        # type_is_correct = []
        # for number in df_o.RDKitDescriptors:
        #     #raise ValueError (f"{smiles=}")
        #     type_is_correct.append(isinstance(number, int))

        # assert all(type_is_correct)
