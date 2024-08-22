from pdchemchain.links import KeepColumns

from ...basetest import BaseTest


class TestKeepColumns(BaseTest):
    _Link = KeepColumns
    _classparams = {"columns": ["int1"]}
    _alt_classparams = {"columns": ["int2"]}

    def test_keep_column(self, link, sample_dataframe):
        df_dropped = link(sample_dataframe)
        assert "int2" in sample_dataframe
        assert "int2" not in df_dropped


class TestKeepColumnsEmptyConf(BaseTest):
    _Link = KeepColumns
    _classparams = {"columns": []}
    _alt_classparams = {"columns": ["int2"]}
