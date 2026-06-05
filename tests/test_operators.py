"""Tests for Link operator overloads and fluent methods: |, .chunked(), .parallel()"""
import psutil
import pytest

from pdchemchain.base import Chain, UnionLink
from pdchemchain.links.dataframe import NullLink
from pdchemchain.links.hpc import ParallelPartitionProcessor, SerialPartitionProcessor


@pytest.fixture
def links():
    return [NullLink(name=f"link{i}") for i in range(5)]


class TestOrOperator:
    def test_returns_union_link(self, links):
        result = links[0] | links[1]
        assert isinstance(result, UnionLink)

    def test_operand_order(self, links):
        result = links[0] | links[1]
        assert result.link1 is links[0]
        assert result.link2 is links[1]

    def test_ror_operand_order(self, links):
        # __ror__ triggers when left side doesn't handle |; simulate by calling directly
        result = links[0].__ror__(links[1])
        assert result.link1 is links[1]
        assert result.link2 is links[0]

    def test_or_with_chain(self, links):
        chain = links[0] + links[1]
        result = chain | links[2]
        assert isinstance(result, UnionLink)
        assert result.link1 is chain
        assert result.link2 is links[2]

    def test_pipeline_expression(self, links):
        # link1 + link2 + (link3 | link4) + link5
        pipeline = links[0] + links[1] + (links[2] | links[3]) + links[4]
        assert isinstance(pipeline, Chain)
        assert len(pipeline.links) == 4
        assert isinstance(pipeline.links[2], UnionLink)

    def test_unsupported_type_returns_not_implemented(self, links):
        result = links[0].__or__("not_a_link")
        assert result is NotImplemented


class TestChunked:
    def test_returns_serial_partition_processor(self, links):
        result = links[0].chunked(100)
        assert isinstance(result, SerialPartitionProcessor)

    def test_partition_size(self, links):
        result = links[0].chunked(250)
        assert result.partition_size == 250
        assert result.num_partitions is None

    def test_wraps_link(self, links):
        result = links[0].chunked(100)
        assert result.link is links[0]


class TestParallel:
    def test_returns_parallel_partition_processor(self, links):
        result = links[0].parallel(4)
        assert isinstance(result, ParallelPartitionProcessor)

    def test_num_partitions_is_twice_workers(self, links):
        result = links[0].parallel(4)
        assert result.num_processes == 4
        assert result.num_partitions == 8

    def test_wraps_link(self, links):
        result = links[0].parallel(4)
        assert result.link is links[0]

    def test_default_workers_uses_physical_cores(self, links):
        result = links[0].parallel()
        expected = psutil.cpu_count(logical=False)
        assert result.num_processes == expected
        assert result.num_partitions == 2 * expected
