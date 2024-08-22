from pdchemchain.links import DropDuplicates

from ...basetest import BaseTest


class TestDropDuplicates(BaseTest):
    _Link = DropDuplicates
    _classparams = {"columns": ["duplicates"]}
    _alt_classparams = {"columns": ["int2"]}

    def test_duplicate_removal(self, link, sample_dataframe):
        df_o = link(sample_dataframe)
        assert max(sample_dataframe.duplicates.value_counts()) > 1
        assert max(df_o.duplicates.value_counts()) == 1


class TestDropDuplicates2(BaseTest):
    _Link = DropDuplicates
    _classparams = {"columns": []}
    _alt_classparams = {"columns": ["int1"]}
