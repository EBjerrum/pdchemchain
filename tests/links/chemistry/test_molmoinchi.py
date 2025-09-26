from pdchemchain.links import MolToInChI
from ...basetest import BaseErrorTest


class TestMolToInChI(BaseErrorTest):
    _Link = MolToInChI
    _classparams = {
        "in_column": "ROMol",
        "out_column": "ElementsAllowed",
        "generate_keys": False,
    }

    _alt_classparams = {
        "in_column": "ROMol2",
        "out_column": "ElementsAllowed2",
        "generate_keys": True,
    }
