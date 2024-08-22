from pdchemchain.links import Query

from ...basetest import BaseTest


class TestQuery(BaseTest):
    _Link = Query
    _classparams = {"query": "int1 > 1"}
    _alt_classparams = {"query": "int2 < 2"}

    def test_query(self, link, sample_dataframe):
        df_dropped = link(sample_dataframe)
        assert min(df_dropped.int1) > 1
        assert min(sample_dataframe.int1) < 2


class TestQueryNoConf(BaseTest):
    _Link = Query
    _classparams = {"query": ""}
    _alt_classparams = {"query": "int2 < 2"}
