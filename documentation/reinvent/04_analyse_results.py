# %%
"""Analyse REINVENT staged learning results.

Quick visual analysis comparing the prior distribution (from step 1) with
the RL-trained agent output.  Shows score progression and property shifts.

Prerequisites:
    - prior_sample.csv from 01_sample_prior.toml
    - staged_learning_0.csv from 03_staged_learning.toml (REINVENT output)
    - scoring_pipeline.yaml from 02_build_scoring_pipeline.py
"""

# %% Imports
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdchemchain.io_utilities import load_chain

# %% Load data
# Prior sample (baseline distribution)
prior_df = pd.read_csv("prior_sample.csv")

# RL output -- REINVENT writes staged_learning_0.csv, staged_learning_1.csv, etc.
rl_files = sorted(glob.glob("staged_learning_*.csv"))
if not rl_files:
    raise FileNotFoundError(
        "No staged_learning_*.csv files found. "
        "Run: reinvent 03_staged_learning.toml"
    )
rl_df = pd.concat([pd.read_csv(f) for f in rl_files], ignore_index=True)
print(f"Prior: {len(prior_df)} SMILES, RL output: {len(rl_df)} rows from {len(rl_files)} file(s)")

# %% Score the prior and RL output with the same pipeline
# Wrap in ParallelPartitionProcessor for speed on large RL output
from pdchemchain.links.hpc import ParallelPartitionProcessor

chain = load_chain("scoring_pipeline.yaml")
parallel_chain = ParallelPartitionProcessor(link=chain, num_processes=16, partition_size=500)

prior_scored = parallel_chain(prior_df)
rl_scored = parallel_chain(rl_df)

# %% Score progression and fraction of 0 HBD over RL steps
fig, ax1 = plt.subplots(figsize=(10, 4))

if "step" in rl_scored.columns and "Score" in rl_scored.columns:
    grouped = rl_scored.groupby("step")

    # Mean score (left y-axis)
    step_scores = grouped["Score"].mean()
    ax1.plot(step_scores.index, step_scores.values, linewidth=1.5, color="tab:blue", label="Mean score")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Mean total score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Fraction of 0 HBD (right y-axis) -- the smoking gun
    ax2 = ax1.twinx()
    frac_zero_hbd = grouped["NumHDonors"].apply(lambda x: (x == 0).mean())
    ax2.plot(frac_zero_hbd.index, frac_zero_hbd.values, linewidth=1.5, color="tab:red", label="Frac. 0 HBD")
    ax2.set_ylabel("Fraction with 0 H-bond donors", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Score progression and HBD elimination during RL")
    ax1.grid(True, alpha=0.3)
else:
    ax1.text(0.5, 0.5, "step/Score columns not found in RL output",
             transform=ax1.transAxes, ha="center")
plt.tight_layout()
plt.show()

# %% Property distributions: prior vs late-stage agent
# Take the last 25% of RL steps as "trained agent" sample
if "step" in rl_scored.columns:
    max_step = rl_scored["step"].max()
    agent_scored = rl_scored[rl_scored["step"] > max_step * 0.75]
else:
    agent_scored = rl_scored.tail(len(rl_scored) // 4)

properties = ["NumHDonors", "MolLogP", "TPSA", "MolWt"]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, prop in zip(axes.flat, properties):
    prior_vals = prior_scored[prop].dropna()
    agent_vals = agent_scored[prop].dropna()

    bins = np.linspace(
        min(prior_vals.min(), agent_vals.min()),
        max(prior_vals.max(), agent_vals.max()),
        50,
    )

    ax.hist(prior_vals, bins=bins, alpha=0.5, density=True, label="Prior")
    ax.hist(agent_vals, bins=bins, alpha=0.5, density=True, label="Agent")
    ax.set_title(prop)
    ax.set_ylabel("Density")
    ax.legend()

fig.suptitle("Property distributions: Prior vs Trained Agent", fontsize=14)
fig.tight_layout()
plt.show()

# %% Top-scoring molecules from the run
if "Score" in agent_scored.columns:
    score_col = "Score"
    properties = properties + ["Score"]
else:
    score_col = "MolLogP"

top = agent_scored.nlargest(10, score_col)
print(f"\nTop 10 molecules by {score_col}:")
print(top[["SMILES", *properties]].to_string())
#%%