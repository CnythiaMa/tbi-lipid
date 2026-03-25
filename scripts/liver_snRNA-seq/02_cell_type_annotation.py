"""
Step 2: Cell type annotation using liver marker genes
Marker reference: mouse liver atlas (Halpern et al., MacParland et al.)
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

RESULT_DIR = "/Users/maxue/Documents/vscode/tbi/results/liver_snRNA-seq"
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
TAB_DIR    = os.path.join(RESULT_DIR, "tables")

# Load processed data
adata = sc.read_h5ad(os.path.join(RESULT_DIR, "liver_snRNA_processed.h5ad"))
print(f"Loaded: {adata.n_obs} nuclei, {adata.n_vars} genes")

# ── Liver cell type markers (mouse) ──────────────────────────────────────────
MARKERS = {
    "Hepatocyte_pericentral":  ["Cyp2e1", "Cyp1a2", "Glul", "Axin2"],
    "Hepatocyte_periportal":   ["Cyp2f2", "Hal", "Sds", "Ass1", "Pck1", "G6pc"],
    "Hepatocyte_mid":          ["Alb", "Ttr", "Apob", "Apoe", "Fabp1", "Hnf4a"],
    "Kupffer_cell":            ["Clec4f", "Csf1r", "Cd68", "Adgre1", "Timd4", "Marco"],
    "Monocyte_macrophage":     ["Ccr2", "Ly6c2", "S100a8", "S100a9", "Cd14"],
    "Hepatic_stellate":        ["Acta2", "Col1a1", "Col1a2", "Dcn", "Des", "Pdgfrb"],
    "Portal_fibroblast":       ["Mfap4", "Thy1", "Fibin", "Wt1"],
    "Endothelial_LSEC":        ["Stab1", "Stab2", "Clec1b", "Lyve1", "Fcgr2b"],
    "Endothelial_portal":      ["Dll4", "Efnb2", "Notch4"],
    "Cholangiocyte":           ["Krt7", "Krt19", "Sox9", "Epcam", "Cftr"],
    "NK_NKT":                  ["Nkg7", "Gzma", "Gzmb", "Klrb1c"],
    "T_cell":                  ["Cd3e", "Cd3d", "Cd4", "Cd8a", "Trac"],
    "B_cell":                  ["Cd79a", "Ms4a1", "Pax5"],
    "Plasmacytoid_DC":         ["Siglech", "Bst2", "Irf7"],
    "Conventional_DC":         ["Itgax", "H2-Aa", "H2-Ab1", "Cd74", "Flt3"],
}

# ── Score each cell type ──────────────────────────────────────────────────────
for ct, genes in MARKERS.items():
    valid = [g for g in genes if g in adata.raw.var_names]
    if valid:
        sc.tl.score_genes(adata, gene_list=valid, score_name=f"score_{ct}", use_raw=True)
        print(f"  {ct}: {len(valid)}/{len(genes)} markers found")
    else:
        print(f"  {ct}: NO markers found")

# ── Auto-annotate clusters by relative score ──────────────────────────────────
score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
score_df = adata.obs[score_cols + ["leiden_r05"]]
cluster_mean = score_df.groupby("leiden_r05").mean()
cluster_mean.columns = [c.replace("score_", "") for c in cluster_mean.columns]

# Hepatocyte score dominates in liver; use relative scoring:
# For each cluster, subtract per-cell-type global mean, then take max
ct_global_mean = cluster_mean.mean(axis=0)
cluster_relative = cluster_mean - ct_global_mean  # center each cell type

# Tiered annotation:
# 1) If non-hepatocyte relative score is highest AND absolute > 0.3 → assign non-hepatocyte
# 2) Among hepatocyte subtypes: pericentral/periportal/mid based on relative scores
HEPATOCYTE_TYPES = {"Hepatocyte_pericentral", "Hepatocyte_periportal", "Hepatocyte_mid"}
NON_HEP_TYPES = [c for c in cluster_relative.columns if c not in HEPATOCYTE_TYPES]

best_ct = {}
print("\n--- Cluster annotation (relative scoring) ---")
for clust in cluster_relative.index:
    row = cluster_relative.loc[clust]
    non_hep_row = row[NON_HEP_TYPES]
    hep_row = row[list(HEPATOCYTE_TYPES)]

    best_non_hep = non_hep_row.idxmax()
    best_non_hep_score = non_hep_row.max()
    best_hep = hep_row.idxmax()
    best_hep_score = hep_row.max()

    # Assign non-hepatocyte if its relative score is clearly higher (threshold 0.1)
    if best_non_hep_score > best_hep_score + 0.1:
        best_ct[clust] = best_non_hep
        label = f"→ {best_non_hep} (non-hep wins: {best_non_hep_score:.3f} vs hep {best_hep_score:.3f})"
    else:
        best_ct[clust] = best_hep
        label = f"→ {best_hep} (hep: {best_hep_score:.3f}, best non-hep: {best_non_hep}={best_non_hep_score:.3f})"

    top3 = cluster_mean.loc[clust].nlargest(3)
    top3_str = ", ".join([f"{k}={v:.3f}" for k, v in top3.items()])
    print(f"  Cluster {clust}: {label}")
    print(f"    Raw top3: {top3_str}")

best_ct_series = pd.Series(best_ct)

# Apply annotation
adata.obs["cell_type_auto"] = adata.obs["leiden_r05"].map(best_ct_series)

# ── Broader cell type grouping ────────────────────────────────────────────────
BROAD = {
    "Hepatocyte_pericentral": "Hepatocyte",
    "Hepatocyte_periportal":  "Hepatocyte",
    "Hepatocyte_mid":         "Hepatocyte",
    "Kupffer_cell":           "Kupffer_cell",
    "Monocyte_macrophage":    "Monocyte_macrophage",
    "Hepatic_stellate":       "Hepatic_stellate",
    "Portal_fibroblast":      "Fibroblast",
    "Endothelial_LSEC":       "Endothelial",
    "Endothelial_portal":     "Endothelial",
    "Cholangiocyte":          "Cholangiocyte",
    "NK_NKT":                 "NK_NKT",
    "T_cell":                 "T_cell",
    "B_cell":                 "B_cell",
    "Plasmacytoid_DC":        "Dendritic_cell",
    "Conventional_DC":        "Dendritic_cell",
}
adata.obs["cell_type_broad"] = adata.obs["cell_type_auto"].map(BROAD).fillna("Unknown")

# ── Dot plot of marker genes ──────────────────────────────────────────────────
key_markers = {
    "Hepatocyte":       ["Alb", "Apob", "Hnf4a", "Cyp2e1", "Cyp2f2", "Fabp1"],
    "Kupffer":          ["Clec4f", "Csf1r", "Adgre1", "Timd4"],
    "Stellate":         ["Acta2", "Col1a1", "Dcn", "Des"],
    "Endothelial":      ["Stab1", "Stab2", "Lyve1", "Clec1b"],
    "Cholangiocyte":    ["Krt7", "Krt19", "Sox9"],
    "T/NK":             ["Cd3e", "Nkg7", "Gzma"],
}
flat_markers = [g for genes in key_markers.values() for g in genes if g in adata.raw.var_names]

sc.pl.dotplot(adata, flat_markers, groupby="leiden_r05", use_raw=True,
              standard_scale="var", show=False, title="Marker genes by cluster")
plt.savefig(os.path.join(FIG_DIR, "dotplot_markers_by_cluster.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── UMAP colored by cell type ─────────────────────────────────────────────────
sc.pl.umap(adata, color="cell_type_broad", show=False, title="Cell types (broad)")
plt.savefig(os.path.join(FIG_DIR, "umap_cell_type_broad.png"), dpi=150, bbox_inches="tight")
plt.close()

sc.pl.umap(adata, color="cell_type_auto", show=False, title="Cell types (detailed)")
plt.savefig(os.path.join(FIG_DIR, "umap_cell_type_auto.png"), dpi=150, bbox_inches="tight")
plt.close()

# Score heatmap: clusters × cell types
fig, ax = plt.subplots(figsize=(12, 6))
score_plot = cluster_mean.T
sns.heatmap(score_plot, cmap="RdBu_r", center=0, ax=ax,
            linewidths=0.3, annot=True, fmt=".2f", annot_kws={"fontsize": 6})
ax.set_title("Cell type scores per cluster (mean)")
ax.set_xlabel("Leiden cluster")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "celltype_score_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Cell proportion table ─────────────────────────────────────────────────────
prop = adata.obs.groupby(["condition", "cell_type_broad"]).size().reset_index(name="n")
total = adata.obs.groupby("condition").size().reset_index(name="total")
prop = prop.merge(total, on="condition")
prop["proportion"] = prop["n"] / prop["total"]
prop.to_csv(os.path.join(TAB_DIR, "cell_type_proportions.csv"), index=False)

# Bar plot of proportions
pivot = prop.pivot(index="cell_type_broad", columns="condition", values="proportion").fillna(0)
pivot.plot(kind="bar", figsize=(12, 5), width=0.7)
plt.title("Cell type proportions: control vs 7dpi")
plt.ylabel("Proportion")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cell_type_proportions.png"), dpi=150)
plt.close()

print("\nCell type proportions:")
print(prop.to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
adata.write_h5ad(os.path.join(RESULT_DIR, "liver_snRNA_annotated.h5ad"))
cluster_mean.to_csv(os.path.join(TAB_DIR, "cluster_celltype_scores.csv"))
best_ct_series.to_frame(name="best_cell_type").to_csv(os.path.join(TAB_DIR, "cluster_annotation.csv"))

print("\n=== Step 2 complete ===")
