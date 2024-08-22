from pdchemchain.links import StripErrors

from ...basetest import BaseTest


class TestToFileNoFile(BaseTest):
    _Link = StripErrors
    _classparams = {"filename": None}
    _alt_classparams = {"filename": "Something"}
