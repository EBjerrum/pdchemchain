from enum import IntEnum


class InColumnName(str):
    """Special string type, that should be used to typehint properties that are in_columns.
    This information will be used to assert that the input columns exists in processed dataframes"""

    pass


class OutColumnName(str):
    """Special string type, that should be used to typehint properties that are out_columns.
    This information may be used to warn users that the columns already exists and will be overwritten"""

    pass


class Partitionable(IntEnum):
    """Ternary partitionability flag for links.

    Controls whether a link can safely be used inside a partition processor.
    Uses IntEnum so min() gives correct AND semantics: NO < MAYBE < YES.

    NO: Link requires the full dataset (e.g. clustering, deduplication).
        Partition processors will raise ValueError.
    MAYBE: Link may or may not be partitionable depending on configuration
        (e.g. DfEval with user-supplied expressions). Partition processors
        will emit a warning.
    YES: Link is safe to partition (e.g. row-wise operations).
        Partition processors allow silently.
    """

    NO = 0
    MAYBE = 1
    YES = 2
