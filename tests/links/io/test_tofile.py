import tempfile
from pdchemchain.links import ToFile, Link

from ...basetest import BaseTest
import pytest

import os, copy


class TestToFile(BaseTest):
    _Link = ToFile

    @pytest.fixture
    def save_filename(self, tmpdir):
        return tmpdir + "/save_test1.csv"

    @pytest.fixture
    def classparams(self, save_filename):
        return {"filename": save_filename, "pd_tocsv_options": {"sep": ","}}

    @pytest.fixture
    def alt_classparams(self, tmpdir):
        return {"filename": tmpdir + "save_test2.csv", "pd_tocsv_options": {"sep": "."}}

    def test_saving_file(self, link, sample_dataframe, save_filename):
        # if os.path.exists(save_filename):
        #     os.remove(save_filename)
        df = link(sample_dataframe)
        assert os.path.exists(save_filename)
