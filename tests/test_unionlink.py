from pdchemchain.base import UnionLink, Link
from pdchemchain.links.dataframe import NullLink
import pytest
from .basetest import BaseTest
from dataclasses import fields


class TestUnionLink(BaseTest):
    _Link = UnionLink
    _classparams = {
        "link1": NullLink(name="test1_1"),
        "link2": NullLink(name="test1_2"),
    }
    _alt_classparams = {
        "link1": NullLink(name="test2_1"),
        "link2": NullLink(name="test2_2"),
    }
