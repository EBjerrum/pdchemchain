from pdchemchain.links.dataframe import NullLink
from ...basetest import BaseTest as BaseTest


class TestNullLink(BaseTest):
    _Link = NullLink
    _classparams = {"name": "link1"}
    _alt_classparams = {"name": "link2"}
