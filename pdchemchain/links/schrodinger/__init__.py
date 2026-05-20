"""
Schrodinger integration links (LigPrep, Glide docking).

These links require a Schrodinger installation accessible via $SCHRODINGER
or configured in ~/.pdchemchain/config.yaml.
"""

from .ligprep import LigPrep
from .glide import GlideDock
from .aggregate import AggregateByScore

__all__ = ["LigPrep", "GlideDock", "AggregateByScore"]
