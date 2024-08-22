from pdchemchain.links import DropColumns

from ...basetest import BaseTest


class TestDropColumns(BaseTest):
    _Link = DropColumns
    _classparams = {"columns": ["int1"]}
    _alt_classparams = {"columns": ["int2"]}

    def test_dropping_column(self, link, sample_dataframe):
        df_dropped = link(sample_dataframe)
        assert "int1" in sample_dataframe
        assert "int1" not in df_dropped
