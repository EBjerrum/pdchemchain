from ...basetest import BaseTest
import pytest

from pdchemchain.links.hpc import SerialPartitionProcessor


class TestSerialPartitionProcessor(BaseTest):
    _Link = SerialPartitionProcessor

    @pytest.fixture
    def classparams(self, testlink):
        return {"link": testlink, "num_partitions": 3}

    @pytest.fixture
    def alt_classparams(self, testlink2):
        return {"link": testlink2, "num_partitions": 2}
