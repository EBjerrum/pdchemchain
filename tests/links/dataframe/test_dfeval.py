from pdchemchain.links import DfEval

from ...basetest import BaseTest as BaseTest


class TestDfEval(BaseTest):
    _Link = DfEval
    _classparams = {"eval_str": "int1 + int2", "out_column": "int3"}
    _alt_classparams = {"eval_str": "int1 - int2", "out_column": "int4"}


class TestDfEval_nooutcolumn(BaseTest):
    _Link = DfEval
    _classparams = {
        "eval_str": "int3 = int1 + int2",
    }
    _alt_classparams = {
        "eval_str": "int4 = int1 - int2",
    }
