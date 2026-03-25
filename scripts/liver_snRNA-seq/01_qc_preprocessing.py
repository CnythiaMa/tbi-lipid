"""
Step 1: QC, filtering, normalization, and dimensionality reduction
Liver snRNA-seq: Control vs 7dpi TBI
"""

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 2
sc.settings.n_jobs = 4

DATA_DIR   = "/Users/maxue/Documents/vscode/tbi/data/liver-snRNA-seq"
RESULT_DIR = "/Users/maxue/Documents/vscode/tbi/results/liver_snRNA-seq"
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
TAB_DIR    = os.path.join(RESULT_DIR, "tables")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading control...")
ctl = sc.read_10x_mtx(os.path.join(DATA_DIR, "control"), var_names="gene_symbols", cache=True)
ctl.obs["condition"] = "control"

print("Loading 7dpi...")
dpi = sc.read_10x_mtx(os.path.join(DATA_DIR, "7dpi"), var_names="gene_symbols", cache=True)
dpi.obs["condition"] = "7dpi"

adata = ad.concat([ctl, dpi], label="sample", keys=["control", "7dpi"])
adata.obs_names_make_unique()
print(f"Raw cells: {adata.n_obs}, genes: {adata.n_vars}")

# ── 2. Basic QC metrics ───────────────────────────────────────────────────────
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

# Save QC overview
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, cond in zip(axes, ["control", "7dpi"]):
    sub = adata[adata.obs["condition"] == cond]
    ax[0].hist(sub.obs["n_genes_by_counts"], bins=80, color="steelblue", edgecolor="none")
    ax[0].set_xlabel("Genes detected"); ax[0].set_title(f"{cond}: genes/nucleus")
    ax[1].hist(sub.obs["total_counts"], bins=80, color="coral", edgecolor="none")
    ax[1].set_xlabel("UMI counts"); ax[1].set_title(f"{cond}: UMI/nucleus")
    ax[2].hist(sub.obs["pct_counts_mt"], bins=80, color="green", edgecolor="none")
    ax[2].set_xlabel("% mitochondrial"); ax[2].set_title(f"{cond}: % MT")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "qc_histograms.png"), dpi=150)
plt.close()

# Print QC statistics per condition
for cond in ["control", "7dpi"]:
    sub = adata[adata.obs["condition"] == cond]
    print(f"\n--- {cond} QC stats ---")
    print(f"  Nuclei: {sub.n_obs}")
    print(f"  Median genes/nucleus: {sub.obs['n_genes_by_counts'].median():.0f}")
    print(f"  Median UMI/nucleus:   {sub.obs['total_counts'].median():.0f}")
    print(f"  Median %MT:           {sub.obs['pct_counts_mt'].median():.2f}")

# ── 3. Filtering ──────────────────────────────────────────────────────────────
# snRNA-seq: lower MT threshold; use adaptive thresholds
MIN_GENES = 200
MAX_GENES = 6000
MAX_MT    = 5.0   # snRNA-seq: nuclei have very low MT

sc.pp.filter_cells(adata, min_genes=MIN_GENES)
sc.pp.filter_genes(adata, min_cells=10)

adata = adata[adata.obs["n_genes_by_counts"] < MAX_GENES].copy()
adata = adata[adata.obs["pct_counts_mt"] < MAX_MT].copy()

print(f"\nAfter QC filtering: {adata.n_obs} nuclei, {adata.n_vars} genes")
for cond in ["control", "7dpi"]:
    n = (adata.obs["condition"] == cond).sum()
    print(f"  {cond}: {n} nuclei")

# ── 4. Normalize & log1p ─────────────────────────────────────────────────────
adata.layers["counts"] = adata.X.copy()   # keep raw counts for DE
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata  # store normalized for visualization

# ── 5. Highly variable genes ──────────────────────────────────────────────────
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key="condition")
print(f"Highly variable genes: {adata.var['highly_variable'].sum()}")

# ── 6. Scale → PCA ────────────────────────────────────────────────────────────
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack", n_comps=50)

# Elbow plot
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, show=False)
plt.savefig(os.path.join(FIG_DIR, "pca_elbow.png"), dpi=150)
plt.close()

# ── 7. Neighbors → UMAP → Leiden clustering ──────────────────────────────────
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
sc.tl.umap(adata, min_dist=0.3)
sc.tl.leiden(adata, resolution=0.5, key_added="leiden_r05")
sc.tl.leiden(adata, resolution=0.8, key_added="leiden_r08")

# UMAP plots
for color in ["condition", "leiden_r05", "leiden_r08", "n_genes_by_counts", "pct_counts_mt"]:
    sc.pl.umap(adata, color=color, show=False, title=color)
    plt.savefig(os.path.join(FIG_DIR, f"umap_{color}.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ── 8. Save processed object ──────────────────────────────────────────────────
out_h5 = os.path.join(RESULT_DIR, "liver_snRNA_processed.h5ad")
adata.write_h5ad(out_h5)
print(f"\nSaved processed AnnData → {out_h5}")

# Summary table
summary = adata.obs.groupby(["condition", "leiden_r05"]).size().reset_index(name="n_nuclei")
summary.to_csv(os.path.join(TAB_DIR, "cluster_cell_counts.csv"), index=False)

print("\n=== Step 1 complete ===")
