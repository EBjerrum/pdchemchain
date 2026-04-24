import pytest
import pandas as pd
from rdkit import Chem

from pdchemchain.base import Link, Chain, UnionLink
from pdchemchain.typing import Partitionable
from pdchemchain.links.dataframe import NullLink, DropDuplicates, DfEval
from pdchemchain.links.chemistry import MolToFingerprint
from pdchemchain.links.clustering import ButinaClustering
from pdchemchain.links.contrib.embeddings import UMAPEmbedding, tSNEEmbedding
from pdchemchain.links.hpc import SerialPartitionProcessor, ParallelPartitionProcessor


class TestPartitionableFlag:
    """Test _partitionable class attribute on individual links."""

    def test_default_is_yes(self):
        assert NullLink._partitionable == Partitionable.YES

    def test_rowlink_is_yes(self):
        assert MolToFingerprint._partitionable == Partitionable.YES

    def test_butina_is_no(self):
        assert ButinaClustering._partitionable == Partitionable.NO

    def test_drop_duplicates_is_no(self):
        assert DropDuplicates._partitionable == Partitionable.NO

    def test_umap_is_no(self):
        assert UMAPEmbedding._partitionable == Partitionable.NO

    def test_tsne_is_no(self):
        assert tSNEEmbedding._partitionable == Partitionable.NO

    def test_dfeval_is_maybe(self):
        assert DfEval._partitionable == Partitionable.MAYBE

    def test_instance_flag(self):
        link = ButinaClustering()
        assert link._partitionable == Partitionable.NO


class TestTernaryLogic:
    """Test that min() gives correct ternary AND semantics."""

    def test_yes_and_yes(self):
        assert min(Partitionable.YES, Partitionable.YES) == Partitionable.YES

    def test_yes_and_maybe(self):
        assert min(Partitionable.YES, Partitionable.MAYBE) == Partitionable.MAYBE

    def test_yes_and_no(self):
        assert min(Partitionable.YES, Partitionable.NO) == Partitionable.NO

    def test_maybe_and_maybe(self):
        assert min(Partitionable.MAYBE, Partitionable.MAYBE) == Partitionable.MAYBE

    def test_maybe_and_no(self):
        assert min(Partitionable.MAYBE, Partitionable.NO) == Partitionable.NO

    def test_no_and_no(self):
        assert min(Partitionable.NO, Partitionable.NO) == Partitionable.NO


class TestChainPartitionable:
    """Test that Chain._partitionable propagates correctly."""

    def test_all_partitionable(self):
        chain = NullLink() + NullLink()
        assert chain._partitionable == Partitionable.YES

    def test_one_non_partitionable(self):
        chain = NullLink() + ButinaClustering()
        assert chain._partitionable == Partitionable.NO

    def test_maybe_propagates(self):
        chain = NullLink() + DfEval(eval_str="a + b")
        assert chain._partitionable == Partitionable.MAYBE

    def test_no_dominates_maybe(self):
        chain = DfEval(eval_str="a + b") + ButinaClustering()
        assert chain._partitionable == Partitionable.NO

    def test_non_partitionable_link_names(self):
        chain = NullLink() + ButinaClustering() + NullLink()
        assert chain._non_partitionable_links() == ["ButinaClustering"]

    def test_maybe_appears_in_link_names(self):
        chain = NullLink() + DfEval(eval_str="a + b")
        assert "DfEval" in chain._non_partitionable_links()

    def test_multiple_non_partitionable(self):
        chain = DropDuplicates(columns=["a"]) + ButinaClustering()
        names = chain._non_partitionable_links()
        assert "DropDuplicates" in names
        assert "ButinaClustering" in names

    def test_nested_chain_propagates(self):
        inner = NullLink() + ButinaClustering()
        outer = NullLink() + inner
        assert outer._partitionable == Partitionable.NO
        assert "ButinaClustering" in outer._non_partitionable_links()


class TestUnionLinkPartitionable:
    """Test that UnionLink._partitionable propagates correctly."""

    def test_both_partitionable(self):
        union = UnionLink(link1=NullLink(), link2=NullLink())
        assert union._partitionable == Partitionable.YES

    def test_one_non_partitionable(self):
        union = UnionLink(link1=NullLink(), link2=ButinaClustering())
        assert union._partitionable == Partitionable.NO

    def test_maybe_propagates(self):
        union = UnionLink(link1=NullLink(), link2=DfEval(eval_str="a + b"))
        assert union._partitionable == Partitionable.MAYBE

    def test_non_partitionable_link_names(self):
        union = UnionLink(link1=ButinaClustering(), link2=NullLink())
        assert union._non_partitionable_links() == ["ButinaClustering"]


class TestPartitionProcessorCheck:
    """Test that partition processors handle ternary partitionability."""

    def test_yes_accepted_silently(self):
        SerialPartitionProcessor(link=NullLink(), partition_size=100)

    # --- NO level: raises by default, warns with override ---

    def test_no_raises(self):
        with pytest.raises(ValueError, match="not partitionable"):
            SerialPartitionProcessor(
                link=ButinaClustering(), partition_size=100
            )

    def test_no_override_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            SerialPartitionProcessor(
                link=ButinaClustering(), partition_size=100,
                allow_non_partitionable=True,
            )
        assert "not partitionable" in caplog.text
        assert "ButinaClustering" in caplog.text

    def test_no_chain_raises(self):
        chain = NullLink() + ButinaClustering()
        with pytest.raises(ValueError, match="ButinaClustering"):
            SerialPartitionProcessor(link=chain, partition_size=100)

    def test_no_dominates_maybe_in_chain(self):
        """Chain with both NO and MAYBE should raise (NO dominates)."""
        chain = DfEval(eval_str="a + b") + ButinaClustering()
        with pytest.raises(ValueError):
            SerialPartitionProcessor(link=chain, partition_size=100)

    def test_parallel_also_checks(self):
        with pytest.raises(ValueError, match="not partitionable"):
            ParallelPartitionProcessor(
                link=ButinaClustering(), partition_size=100
            )

    def test_error_lists_offending_links(self):
        chain = NullLink() + DropDuplicates(columns=["a"]) + ButinaClustering()
        with pytest.raises(ValueError, match="DropDuplicates") as exc_info:
            SerialPartitionProcessor(link=chain, partition_size=100)
        assert "ButinaClustering" in str(exc_info.value)

    def test_error_suggests_override(self):
        with pytest.raises(ValueError, match="allow_non_partitionable=True"):
            SerialPartitionProcessor(
                link=ButinaClustering(), partition_size=100
            )

    def test_error_suggests_move(self):
        with pytest.raises(ValueError, match="Move them outside"):
            SerialPartitionProcessor(
                link=ButinaClustering(), partition_size=100
            )

    # --- MAYBE level: always warns, no flag needed ---

    def test_maybe_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            SerialPartitionProcessor(
                link=DfEval(eval_str="a + b"), partition_size=100
            )
        assert "may not be partitionable" in caplog.text
        assert "DfEval" in caplog.text

    def test_maybe_chain_warns(self, caplog):
        import logging
        chain = NullLink() + DfEval(eval_str="a + b")
        with caplog.at_level(logging.WARNING):
            SerialPartitionProcessor(link=chain, partition_size=100)
        assert "DfEval" in caplog.text
