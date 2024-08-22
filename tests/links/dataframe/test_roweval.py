from pdchemchain.errormanager import rows_with_errors
from pdchemchain.links import RowEval

from ...basetest import BaseErrorTest
import pandas as pd


class TestRowEval(BaseErrorTest):
    _Link = RowEval
    _classparams = {
        "eval_str": "row.letters1 + row.mixed_types",
        "out_column": "eval_result",
    }
    _alt_classparams = {
        "eval_str": "letters2 + mixed_types",
        "out_column": "eval_result2",
    }
