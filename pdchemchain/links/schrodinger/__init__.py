"""
Schrodinger integration links (LigPrep, Glide docking).

These links require a Schrodinger installation accessible via $SCHRODINGER
or configured in ~/.pdchemchain/config.yaml.
"""

from dataclasses import dataclass

from pdchemchain.links.dataframe import GroupPick
from pdchemchain.typing import InColumnName

from .glide import GlideDock
from .ligprep import LigPrep


@dataclass
class AggregateByScore(GroupPick):
    """Select the best-scoring row per group from an expanded DataFrame.

    Convenience subclass of :class:`~pdchemchain.links.dataframe.GroupPick`
    with docking-specific defaults. Equivalent to
    ``GroupPick(group_by="__id__", action="collapse")``.
    """

    group_by: InColumnName = "__id__"
    score_column: InColumnName = "docking_score"
    mode: str = "min"
    action: str = "collapse"


__all__ = ["LigPrep", "GlideDock", "AggregateByScore"]
