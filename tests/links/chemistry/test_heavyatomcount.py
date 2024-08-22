from pdchemchain.links import HeavyAtomCount
from ...basetest import BaseErrorTest


class TestHeavyAtomCount(BaseErrorTest):
    _Link = HeavyAtomCount
    _classparams = {"in_column": "ROMol", "out_column": "HeavyAtomCount"}
    _alt_classparams = {"in_column": "ROMol2", "out_column": "HeavyAtomCount2"}
