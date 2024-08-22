from pdchemchain.links import ElementsInList
from ...basetest import BaseErrorTest


class TestElementsInList(BaseErrorTest):
    _Link = ElementsInList
    _classparams = {
        "allowed_elements": {6, 7, 8, 9, 16, 17, 35},
        "in_column": "ROMol",
        "out_column": "ElementsAllowed",
    }

    _alt_classparams = {
        "allowed_elements": {6, 4, 8, 9, 16, 17, 35},
        "in_column": "ROMol2",
        "out_column": "ElementsAllowed2",
    }
