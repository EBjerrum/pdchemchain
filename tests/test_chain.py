from pdchemchain.base import Chain, Link
from pdchemchain.links.dataframe import NullLink
import pytest
from .basetest import BaseTest
from dataclasses import fields


class TestChain(BaseTest):
    _Link = Chain
    _classparams = {"links": [NullLink(name="test1")]}
    _alt_classparams = {"links": [NullLink(name="test2")]}

    def test_add_operator_order(self, LinkClass, classparams, alt_classparams):
        link1 = LinkClass(**classparams)
        link2 = LinkClass(**alt_classparams)

        chain = (
            link1 + link2
        )  # chain + chain, doesnt give nested chain but updates .links, thus need to compare to .links[0]
        assert chain.links[0] is link1.links[0]
        assert chain.links[0] is not link2.links[0]

        assert chain.links[1] is not link1.links[0]
        assert chain.links[1] is link2.links[0]

    def test_sum_operator(self, LinkClass, classparams, alt_classparams):
        link1 = LinkClass(**classparams)
        link2 = LinkClass(**alt_classparams)
        classparams["links"] = [NullLink(name="test3")]
        link3 = LinkClass(**classparams)

        chain = sum([link1, link2, link3])
        assert isinstance(chain, Chain)
        assert chain.links[0] is link1.links[0]
        assert chain.links[1] is link2.links[0]
        assert chain.links[2] is link3.links[0]

    def test_cloning_negative(self, link, LinkClass, alt_classparams):
        pass
        # TODO, this is getting a bit weird to test, illustrates trouble between using dicts as kwargs (with Link Objects)
        # and using dicts as params (with child param dicts)
        # params = link.get_params()

        # #Create the alt_classparams in a way that can be used for .from_params()
        # link2 = LinkClass.from_params(alt_classparams)
        # alt_classparams_dict = link2.get_params()

        # #update params
        # for key in params.keys():
        #     params[key] = alt_classparams_dict[key]

        # #Aren't we doing a bit of circular reasoning??
        # link2 = LinkClass.from_params(params)

        # assertations = []
        # for name, value in alt_classparams.items():
        #     assertations.append(getattr(link, name) != getattr(link2, name))

        # assert all(assertations)
