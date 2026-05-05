# %%
"""Schrodinger Integration Example: LigPrep + Glide Docking + MM-GBSA

Demonstrates the full Schrodinger pipeline in pdchemchain:
1. LigPrep — 3D preparation, tautomers, ionization states
2. GlideDock — SP docking into a receptor grid
3. AggregateByScore — pick best pose per compound
4. PrimeMMGBSA — binding free energy estimation

Target: HSP90α N-terminal domain (PDB: 1UYD)
HSP90 (Heat Shock Protein 90) is a molecular chaperone essential for the stability
and function of numerous client proteins involved in cell growth and survival.
The N-terminal ATP-binding domain is the primary drug target — inhibitors compete
with ATP binding, destabilizing client proteins and triggering their degradation.
HSP90 inhibitors are pursued as anticancer agents since many oncoproteins (HER2,
BRAF, ALK, etc.) are HSP90 clients.

Grid file from ReinventCommunity (DockStream examples).
"""

# %% Download grid file if not present
from pathlib import Path
import urllib.request

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
grid_file = data_dir / "1UYD_grid.zip"

if not grid_file.exists():
    url = "https://github.com/MolecularAI/ReinventCommunity/raw/master/notebooks/data/DockStream/1UYD_grid.zip"
    print(f"Downloading grid file from {url}...")
    urllib.request.urlretrieve(url, grid_file)
    print(f"Saved to {grid_file}")
else:
    print(f"Grid file already present: {grid_file}")

# Glide needs an absolute path (the .in file is written to a temp dir)
grid_file = grid_file.resolve()

# %% Define ligands with experimental affinity
# HSP90AA1 inhibitors from ExCAPE-DB (https://zenodo.org/records/2543724)
# 10 actives binned across pXC50 range (5.0–9.7) + 5 screening inactives
import pandas as pd

df = pd.DataFrame({
    "Name": [
        # Actives (ChEMBL/PubChem confirmed binders)
        "CHEMBL366637", "PubChem_2920939", "PubChem_2290403",
        "CHEMBL1834387", "CHEMBL1807793", "CHEMBL3234765",
        "CHEMBL1939377", "PubChem_25064779", "CHEMBL252164", "CHEMBL1230584",
        # Inactives (PubChem screening)
        "Inactive_1", "Inactive_2", "Inactive_3", "Inactive_4", "Inactive_5",
    ],
    "Smiles": [
        "C12=C(C(=O)C=C(C1=O)NC(C(=CC=C[C@H](OC)[C@H](C(=C[C@@H]([C@H]([C@H](C[C@@H](C2)C)OC)O)C)C)OC(=O)N)C)=O)NCCCCOC=3C=C4OC(N5CCOCC5)=CC(C4=CC3)=O",
        "ClC1=CC2=C(N(CC(O)COC3=CC=C(N(C)C(=O)C)C=C3)C4=C2C=C(Cl)C=C4)C=C1",
        "O=C1NC(=O)N=C1CC=2C=3C(N(C2)CC#C)=CC=CC3",
        "N=1C(=NC(=C(C1)C(=O)OCC)CCC)N",
        "C1=C(C2=CC(=C1)N3C4=C(C(=C3CCN(C[C@@H](C)[C@H](C)N2)C)C)C(CC(C4)(C)C)=O)C(=O)N",
        "C1(=CC(=C(C=C1N2C(=C(N=N2)C(NCC)=O)C3=CC=C(C=C3)CN4CCCC(C4)O)C(C)C)O)O",
        "C=1C(C)=C2C=C(C1OC)C(=O)NCCCOC3=CC=CC(=C3)SC4=CC2=NC(N)=N4",
        "ClC1=C(C=2C=3C(COCC3C=CC2)=C1)C=4N=C(SCCN(C)C)N=C(N4)N",
        "OC1=CC(=C(C=C1C(C)C)C2=C(C(=NO2)C(=O)NCC)C=3C=CC(CN4CCOCC4)=CC3)O",
        "ClC=1N=C(N)N=C2N(CC=3N=CC(=C(C3C)OC)C)C=C(C12)C#CCC(C)(O)C",
        # Inactives
        "OC(C(C)C)(C1=CC=CC=C1)C#CCN2CCCC2",
        "O(C(=O)N1CCN(CC1)C(=O)COCC2=NOC(C3=CC=4OCOC4C=C3)=C2)CC",
        "S(=O)(=O)(N1CCN(CC1)C=2C=CC(=CC2)C(=O)C)C3=CC=4CC(N(C4C=C3)C(=O)C)C",
        "S=C(N1CCC(NC(=O)C=2OC=CC2)CC1)NC=3C=CC(F)=CC3",
        "FC(F)(F)C1(O)N(N=C(C1)CC)C(=O)CCC2=CC=CC=C2",
    ],
    "pXC50": [5.00, 5.29, 5.88, 6.47, 7.06, 7.64, 8.24, 8.82, 9.40, 9.70,
              None, None, None, None, None],
})
df

# %% ===== STEP 1: LigPrep — ligand preparation =====
# LigPrep generates 3D conformers, tautomers, and ionization states.
# One input molecule can produce multiple "variants" (one-to-many expansion).
# Key output columns:
#   __id__         — original compound index (groups variants)
#   __enum_id__    — unique variant ID
#   __ROMolLigPrep__ — 3D RDKit Mol object
#   LigPrepSmiles  — SMILES of the prepared variant
from pdchemchain.links.schrodinger import LigPrep, GlideDock, AggregateByScore, PrimeMMGBSA

ligprep = LigPrep(
    in_column="Smiles",
    njobs=16,           # Parallel subjobs (auto-detects available CPU slots)
    max_stereo=4,      # Limit stereoisomers
)

df_prepped = ligprep(df)
print(f"LigPrep: {len(df)} compounds → {len(df_prepped)} variants")
df_prepped[["Name", "__id__", "__enum_id__", "LigPrepSmiles"]].head(10)

# %% ===== STEP 2: Glide Docking — SP mode =====
# GlideDock reads from:  __ROMolLigPrep__ (3D Mol objects from LigPrep)
# GlideDock produces:
#   docking_score   — primary Glide SP score (kcal/mol, more negative = better)
#   glide_gscore    — GlideScore (empirical scoring function)
#   glide_emodel    — Emodel (used for pose ranking within a ligand)
#   __ROMolDocked__ — docked 3D pose as RDKit Mol (consumed by MM-GBSA downstream)
glide = GlideDock(
    grid_file=str(grid_file),
    njobs=16,
    keep_tempdir=False #Wheter to keep temporary files, useful for debugging.
)

df_docked = glide(df_prepped)
print(f"Docked: {len(df_docked)} poses")
df_docked[["Name", "__id__", "docking_score"]].head(10)

# %% ===== STEP 3: Aggregate — best pose per compound =====
# AggregateByScore picks the best-scoring variant per __id__ (collapses expansion).
# After this step, we're back to one row per input compound.
agg = AggregateByScore()

df_best = agg(df_docked)
print(f"After aggregation: {len(df_best)} compounds")
df_best[["Name", "docking_score"]].head()

# %% ===== STEP 4: MM-GBSA — binding free energy =====
# PrimeMMGBSA reads from: __ROMolDocked__ (docked 3D poses from GlideDock)
# PrimeMMGBSA produces:
#   mmgbsa_dg_bind_bind     — MM-GBSA binding free energy (kcal/mol, more negative = better)
#   mmgbsa_lig_strain  — ligand strain energy upon binding
# Requires a receptor structure — accepts the same Glide grid .zip file
# (extracts the *_recep.mae automatically) or a direct .mae path.
mmgbsa = PrimeMMGBSA(receptor_file=str(grid_file), njobs=16)

df_mmgbsa = mmgbsa(df_best)
print(f"MM-GBSA complete: {len(df_mmgbsa)} compounds")
df_mmgbsa[["Name", "docking_score", "mmgbsa_dg_bind"]].head()

# %% ===== Correlation with experimental affinity =====
import matplotlib.pyplot as plt

# Assign pXC50=11 to inactives for plotting (beyond active range, visually distinct)
df_plot = df_mmgbsa.copy()
df_plot["pXC50_plot"] = df_plot["pXC50"].fillna(11.0)
df_actives = df_plot[df_plot["pXC50"].notna()]
df_inactives = df_plot[df_plot["pXC50"].isna()]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Glide SP vs pXC50 (score on X, affinity on Y)
ax = axes[0]
ax.scatter(df_actives["docking_score"], df_actives["pXC50_plot"], label="Actives", color="C0")
if len(df_inactives):
    ax.scatter(df_inactives["docking_score"], df_inactives["pXC50_plot"],
               label="Inactives (pXC50=11)", color="red", marker="x")
    ax.axvspan(df_inactives["docking_score"].min(), df_inactives["docking_score"].max(),
               alpha=0.08, color="red")
ax.set_xlabel("Glide SP docking score (kcal/mol)")
ax.set_ylabel("pXC50 (experimental)")
ax.set_title("Docking vs. Affinity")
ax.legend(fontsize=8)

# MM-GBSA vs pXC50 (score on X, affinity on Y)
ax = axes[1]
ax.scatter(df_actives["mmgbsa_dg_bind"], df_actives["pXC50_plot"], label="Actives", color="C0")
if len(df_inactives):
    ax.scatter(df_inactives["mmgbsa_dg_bind"], df_inactives["pXC50_plot"],
               label="Inactives (pXC50=11)", color="red", marker="x")
    ax.axvspan(df_inactives["mmgbsa_dg_bind"].min(), df_inactives["mmgbsa_dg_bind"].max(),
               alpha=0.08, color="red")
ax.set_xlabel("MM-GBSA dG_bind (kcal/mol)")
ax.set_ylabel("pXC50 (experimental)")
ax.set_title("MM-GBSA vs. Affinity")
ax.legend(fontsize=8)

fig.tight_layout()

# %% ===== Observations =====
# Glide SP docking scores show poor correlation with experimental pXC50 — inactives
# score in the same range as actives. 
#
# MM-GBSA shows better separation: Some actives are predicted better than the in-
# actives range. A subset of actives falls on a seemingly linear trend, while
# others are predicted worse than expected if they should be on that trend, maybe 
# due to incorrect binding poses from SP docking

# %% ===== Full pipeline as a single chain =====
# All steps can be composed into one chain and serialized to YAML
full_pipeline = ligprep + glide + agg + mmgbsa
print(f"Pipeline has {len(full_pipeline.links)} links")

# Save for CLI reuse:
full_pipeline.to_config_file("schrodinger_pipeline.yaml")
# Then run: pdchemchain run schrodinger_pipeline.yaml --in_file ligands.csv --out_file results.csv
# %%
