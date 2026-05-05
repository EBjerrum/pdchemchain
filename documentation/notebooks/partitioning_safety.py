# %%
"""
Partitioning Safety in pdchemchain
===================================

pdchemchain supports processing large datasets in partitions using
SerialPartitionProcessor and ParallelPartitionProcessor. However, not all
links produce correct results when the data is split — for example,
deduplication needs to see all rows to find duplicates.

This notebook demonstrates the ternary partitionability system that
catches these mistakes at chain-construction time, before any data is processed.
"""

#%% Imports
import pandas as pd
from pdchemchain.links.dataframe import NullLink, DropDuplicates, DfEval
from pdchemchain.links.hpc import SerialPartitionProcessor

#%% Create a sample dataset with duplicates
df = pd.DataFrame({
    "Smiles": ["CCO", "CCN", "CCO", "CCC", "CCN", "CCCO"],
    "score": [1.0, 2.0, 1.0, 3.0, 2.0, 4.0],
})
print(f"Dataset: {len(df)} rows, {df['Smiles'].nunique()} unique SMILES")
df

# %% DropDuplicates works fine on the full dataset
dedup = DropDuplicates(columns=["Smiles"])
result = dedup(df)
print(f"After dedup: {len(result)} rows — correct!")
result

# %% But what if we partition first? Let's see what happens when we try.
# pdchemchain catches this at construction time:
try:
    proc = SerialPartitionProcessor(
        link=dedup,
        partition_size=3,
    )
except ValueError as e:
    print(f"Caught: {e}")

# %% The error tells us exactly what's wrong and what to do:
#   - Which links are non-partitionable: ['DropDuplicates']
#   - Why: they require the full dataset
#   - Fix option 1: move them outside the partition processor
#   - Fix option 2: set allow_non_partitionable=True to override

# %% The correct pattern: dedup BEFORE the partitioned processing
# Dedup on the full dataset first, then partition the expensive computation.
heavy_computation = NullLink(name="expensive_link")  # placeholder

chain = dedup + SerialPartitionProcessor(
    link=heavy_computation,
    partition_size=3,
)

result = chain(df)
print(f"After dedup + partitioned processing: {len(result)} rows — correct, and fewer rows to process!")
result

# %% This also works through chains and nesting.
# If DropDuplicates is anywhere inside a chain passed to a partition processor,
# it will be detected:
chain_with_dedup = NullLink() + DropDuplicates(columns=["Smiles"]) + NullLink()
print(f"Chain partitionable? {chain_with_dedup._partitionable}")
print(f"Offending links: {chain_with_dedup._non_partitionable_links()}")

# %% The system uses ternary logic (YES / MAYBE / NO)
# Some links like DfEval *might* be non-partitionable depending on
# the expression. These get a MAYBE — the processor warns but doesn't block.
from pdchemchain.typing import Partitionable

print(f"NullLink:       {NullLink._partitionable.name}")
print(f"DropDuplicates: {DropDuplicates._partitionable.name}")
print(f"DfEval:         {DfEval._partitionable.name}")

# %% DfEval in a partition processor triggers a warning (check your logs):
import logging
logging.basicConfig(level=logging.WARNING)

proc = SerialPartitionProcessor(
    link=DfEval(eval_str="score * 2", out_column="doubled"),
    partition_size=3,
)
# This works fine — "score * 2" is row-independent.
# But if someone wrote "score.rank()" it would give wrong results per-partition.
# The warning reminds you to verify.
result = proc(df)
result

# %% For expert users who understand the risk, allow_non_partitionable=True
# overrides the block for NO-level links:
proc = SerialPartitionProcessor(
    link=DropDuplicates(columns=["Smiles"]),
    partition_size=3,
    allow_non_partitionable=True,  # "I know what I'm doing"
)
result = proc(df)
print(f"With override: {len(result)} rows — may have partition-boundary duplicates!")
result
# %%
