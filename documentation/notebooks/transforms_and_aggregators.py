# %%
"""Transforms and Aggregators — score normalization and combination.

Transforms map raw numeric scores to [0, 1].
Direction is inferred from good_value vs bad_value — no objective/reverse parameters.

Aggregators combine multiple [0, 1] score columns into a single compound score.
NaN values (molecules that didn't reach an expensive stage) contribute 0.0.

Evaluating a transform object in a cell shows the curve automatically via _repr_png_.
"""

# %% Imports
import math
import pandas as pd

from pdchemchain.links.transforms import (
    StepTransform,
    LinearTransform,
    SigmoidTransform,
    GaussianTransform,
    PlateauTransform,
)
from pdchemchain.links.aggregators import WeightedSum, WeightedGeometricMean, ArithmeticMean

# %% Sample data — raw descriptor scores we want to normalize
df = pd.DataFrame({
    "MolWt":       [280, 380, 450, 520, 650],
    "logP":        [1.2, 2.8, 3.5, 4.9, 6.1],
    "qed":         [0.82, 0.71, 0.55, 0.38, 0.21],
    "docking":     [-11.2, -8.4, -6.5, -4.1, -2.3],
    "sa_score":    [1.8, 2.5, 3.1, 4.4, 6.2],
})
df

# %% -------------------------------------------------------------------
# StepTransform — hard threshold
# Higher QED is better; bad_value sets the cutoff.
# Evaluating the object shows the curve.
# -------------------------------------------------------------------
StepTransform(in_column="qed", out_column="qed_pass",
              good_value=0.7, bad_value=0.4)

# %% Descending step: lower SA score is better
StepTransform(in_column="sa_score", out_column="sa_pass",
              good_value=2.0, bad_value=4.0)

# %% Apply to data
step = StepTransform(in_column="qed", out_column="qed_pass",
                     good_value=0.7, bad_value=0.4)
step(df)[["qed", "qed_pass"]]

# %% -------------------------------------------------------------------
# LinearTransform — smooth ramp, clipped at both ends
# -------------------------------------------------------------------
LinearTransform(in_column="MolWt", out_column="mw_score",
                good_value=400, bad_value=600)

# %% Ascending: higher docking binding energy (less negative) mapped
LinearTransform(in_column="logP", out_column="logP_score",
                good_value=2.5, bad_value=5.0)

# %% Apply to data
linear = LinearTransform(in_column="MolWt", out_column="mw_score",
                         good_value=400, bad_value=600)
linear(df)[["MolWt", "mw_score"]]

# %% -------------------------------------------------------------------
# SigmoidTransform — smooth S-curve, auto-tuned steepness
# Default k is chosen so score ≈ 0.98 at good_value, ≈ 0.02 at bad_value.
# -------------------------------------------------------------------
SigmoidTransform(in_column="docking", out_column="dock_score",
                 good_value=-10.0, bad_value=-4.0)

# %% Explicit k: sharper transition
SigmoidTransform(in_column="docking", out_column="dock_score_sharp",
                 good_value=-10.0, bad_value=-4.0, k=2.0)

# %% Apply and compare auto vs explicit k
sig_auto  = SigmoidTransform(in_column="docking", out_column="dock_auto",  good_value=-10, bad_value=-4)
sig_sharp = SigmoidTransform(in_column="docking", out_column="dock_sharp", good_value=-10, bad_value=-4, k=2.0)
chain = sig_auto + sig_sharp
chain(df)[["docking", "dock_auto", "dock_sharp"]]

# %% -------------------------------------------------------------------
# GaussianTransform — bell curve, peak at good_value
# Useful when a target range is ideal and both extremes are bad.
# sigma = |good - bad| / 3  (3-sigma ≈ 0.01 at bad_value)
# -------------------------------------------------------------------
GaussianTransform(in_column="logP", out_column="logP_score",
                  good_value=2.5, bad_value=5.5)

# %% The curve falls symmetrically from the peak — both directions penalised
gauss = GaussianTransform(in_column="logP", out_column="logP_score",
                          good_value=2.5, bad_value=5.5)
gauss(df)[["logP", "logP_score"]]

# %% -------------------------------------------------------------------
# PlateauTransform — flat top within good range, sigmoid shoulders
# good_value = plateau where score is 1.0
# bad_value = outer bounds where score reaches ~0.02
# Asymmetric slopes come naturally when left/right distances differ.
# -------------------------------------------------------------------
PlateauTransform(in_column="logP", out_column="logP_score",
                 good_value=(1.0, 3.5), bad_value=(-1.0, 6.0))

# %% Asymmetric example: tighter on the high side
PlateauTransform(in_column="MolWt", out_column="mw_score",
                 good_value=(300, 450), bad_value=(150, 550))

# %% Apply plateau
plateau = PlateauTransform(in_column="logP", out_column="logP_score",
                           good_value=(1.0, 3.5), bad_value=(-1.0, 6.0))
plateau(df)[["logP", "logP_score"]]

# %% Saving a plot to file
fig = PlateauTransform(in_column="logP", out_column="logP_score",
                       good_value=(1.0, 3.5), bad_value=(-1.0, 6.0)).plot()
fig.savefig("logP_plateau.png", dpi=150)
print("Saved to logP_plateau.png")

# %% -------------------------------------------------------------------
# Aggregators — combining [0, 1] scores into a single compound score
# Build a fully scored DataFrame first
# -------------------------------------------------------------------
from pdchemchain.base import Chain

score_chain = (
    SigmoidTransform(in_column="qed",      out_column="qed_score",  good_value=0.8, bad_value=0.3)
    + SigmoidTransform(in_column="sa_score", out_column="sa_score_t", good_value=2.0, bad_value=5.0)
    + LinearTransform( in_column="MolWt",    out_column="mw_score",   good_value=400, bad_value=600)
)

df_scored = score_chain(df)
df_scored[["qed", "qed_score", "sa_score", "sa_score_t", "MolWt", "mw_score"]]

# %% WeightedSum — normalised to [0, 1], NaN treated as 0.0
agg = WeightedSum(
    columns=["qed_score", "sa_score_t", "mw_score"],
    weights=[2.0, 1.0, 1.0],   # qed weighted double; normalised internally
    out_column="cheap_score",
)
agg(df_scored)[["qed_score", "sa_score_t", "mw_score", "cheap_score"]]

# %% WeightedGeometricMean — all terms must be good; a zero in any term drags the result to near zero
geom = WeightedGeometricMean(
    columns=["qed_score", "sa_score_t", "mw_score"],
    weights=[2.0, 1.0, 1.0],
    out_column="cheap_score_geom",
)
geom(df_scored)[["qed_score", "sa_score_t", "mw_score", "cheap_score_geom"]]

# %% ArithmeticMean — unweighted
mean = ArithmeticMean(columns=["qed_score", "sa_score_t", "mw_score"],
                      out_column="cheap_score_mean")
mean(df_scored)[["qed_score", "sa_score_t", "mw_score", "cheap_score_mean"]]

# %% -------------------------------------------------------------------
# Funnel scoring: NaN handling
# Molecules that don't pass the cheap gate never get docking scores.
# Their dock_score is NaN → treated as 0.0 by the aggregator.
# Max achievable final_score for non-docked molecules: 0.5 (cheap weight)
# -------------------------------------------------------------------
from pdchemchain.links.dataframe import Query

# Simulate: only top-2 molecules get docked
df_funnel = df_scored.copy()
df_funnel["dock_score"] = [0.92, 0.74, float("nan"), float("nan"), float("nan")]

final = WeightedSum(
    columns=["cheap_score", "dock_score"],
    weights=[1.0, 1.0],
    out_column="final_score",
)
# Build cheap_score first, then aggregate
full_chain = agg + final
df_funnel = full_chain(df_funnel)
df_funnel[["cheap_score", "dock_score", "final_score"]]

# %%
