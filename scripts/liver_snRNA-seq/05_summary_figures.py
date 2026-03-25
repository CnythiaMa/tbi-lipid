"""
Step 5: Summary figures integrating liver snRNA-seq findings
with lipidomics and brain data context
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

RESULT_DIR = "/Users/maxue/Documents/vscode/tbi/results/liver_snRNA-seq"
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
TAB_DIR    = os.path.join(RESULT_DIR, "tables")

adata = sc.read_h5ad(os.path.join(RESULT_DIR, "liver_snRNA_final.h5ad"))

# ── Fig A: Cell composition overview ─────────────────────────────────────────
prop = adata.obs.groupby(["condition", "cell_type_broad"]).size().reset_index(name="n")
total = adata.obs.groupby("condition").size().reset_index(name="total")
prop = prop.merge(total, on="condition")
prop["pct"] = prop["n"] / prop["total"] * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
palette = sns.color_palette("tab20", n_colors=prop["cell_type_broad"].nunique())
ct_colors = dict(zip(sorted(prop["cell_type_broad"].unique()), palette))

for ax, cond in zip([ax1, ax2], ["control", "7dpi"]):
    sub = prop[prop["condition"] == cond].sort_values("pct", ascending=False)
    bars = ax.bar(range(len(sub)), sub["pct"],
                  color=[ct_colors[ct] for ct in sub["cell_type_broad"]])
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["cell_type_broad"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("% of nuclei")
    ax.set_title(f"{cond} (n={sub['n'].sum()})")
    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{row['pct']:.1f}%", ha="center", fontsize=7)

plt.suptitle("Liver snRNA-seq: Cell type composition", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "summary_A_cell_composition.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Fig B: Core pathway summary bar chart ─────────────────────────────────────
all_path = pd.read_csv(os.path.join(TAB_DIR, "hepatocyte_all_pathways_log2FC.csv"))

PATHWAY_LABELS = {
    "IL6_STAT3_acute_phase": "IL-6/STAT3\nAcute Phase",
    "Lipid_synthesis_VLDL":  "Lipid Synthesis\n& VLDL",
    "DHA_PUFA_synthesis":    "DHA/PUFA\nSynthesis",
    "FA_oxidation":          "FA\nβ-Oxidation",
    "Lipid_uptake_transport":"Lipid\nUptake",
    "Lipid_droplet_LDAM":   "LD / LDAM\nBiology",
    "Cholesterol_bile_acid": "Cholesterol\n& Bile Acid",
    "Inflammation_TBI":      "Inflammatory\nSignaling",
    "Glucose_energy":        "Glucose &\nEnergy",
}

path_summary = (all_path.groupby("pathway")["log2FC"]
                .agg(["mean", "std", "count"])
                .reset_index())
path_summary["label"] = path_summary["pathway"].map(PATHWAY_LABELS).fillna(path_summary["pathway"])
path_summary["sem"] = path_summary["std"] / np.sqrt(path_summary["count"])
path_summary = path_summary.sort_values("mean", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e74c3c" if v > 0 else "#2980b9" for v in path_summary["mean"]]
ax.bar(range(len(path_summary)), path_summary["mean"],
       yerr=path_summary["sem"], color=colors, capsize=4, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(path_summary)))
ax.set_xticklabels(path_summary["label"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Mean log2FC (7dpi / control)")
ax.set_title("Hepatocyte pathway changes at 7dpi (mean ± SEM)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "summary_B_pathway_barchart.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Fig C: Key genes grouped barplot ──────────────────────────────────────────
KEY_GENE_GROUPS = {
    "IL-6/STAT3": ["Saa1", "Saa2", "Socs3", "Crp", "Stat3", "Hamp", "Lcn2"],
    "VLDL/ApoB":  ["Apob", "Mttp", "Apoc3", "Apoe", "Apoa1"],
    "DHA/PUFA":   ["Fads2", "Fads1", "Elovl2", "Elovl5", "Acsl6"],
    "FA synth":   ["Fasn", "Scd1", "Acaca"],
    "LD/LDAM":    ["Plin2", "Lpl", "Cd36", "Mgll", "Plin3"],
}

gene_fc_map = {}
for _, row in all_path.iterrows():
    gene_fc_map[row["gene"]] = row["log2FC"]

fig, axes = plt.subplots(1, len(KEY_GENE_GROUPS), figsize=(16, 5), sharey=False)
for ax, (group, genes) in zip(axes, KEY_GENE_GROUPS.items()):
    genes_found = [(g, gene_fc_map.get(g, 0)) for g in genes if g in gene_fc_map]
    if not genes_found:
        ax.set_visible(False)
        continue
    names, fcs = zip(*genes_found)
    colors = ["#e74c3c" if v > 0 else "#2980b9" for v in fcs]
    ax.bar(range(len(names)), fcs, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_title(group, fontsize=10)
    ax.set_ylabel("log2FC" if ax == axes[0] else "")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.axhline(-0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.6)

plt.suptitle("Hepatocyte key gene changes: 7dpi vs control", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "summary_C_key_genes_grouped.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Fig D: Kupffer LDAM comparison ───────────────────────────────────────────
kup_df_path = os.path.join(TAB_DIR, "kupffer_LDAM_expression.csv")
if os.path.exists(kup_df_path):
    kup_df = pd.read_csv(kup_df_path)
    if "log2FC" in kup_df.columns and len(kup_df) > 0:
        kup_df = kup_df.sort_values("log2FC", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#e74c3c" if v > 0 else "#2980b9" for v in kup_df["log2FC"]]
        ax.bar(range(len(kup_df)), kup_df["log2FC"], color=colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(kup_df)))
        ax.set_xticklabels(kup_df.index if "gene" not in kup_df.columns else kup_df["gene"],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title("Kupffer cells: LDAM markers (7dpi vs control)")
        ax.set_ylabel("log2FC")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "summary_D_kupffer_LDAM.png"), dpi=150, bbox_inches="tight")
        plt.close()

# ── Fig E: UMAP split by condition with key marker overlay ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, cond in zip(axes, ["control", "7dpi"]):
    sub = adata[adata.obs["condition"] == cond]
    sc.pl.umap(sub, color="cell_type_broad", ax=ax, show=False,
               title=f"{cond}\n(n={sub.n_obs})", legend_loc="right margin",
               size=6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "summary_E_umap_split_condition.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Fig F: Key finding summary - combined read-out table ─────────────────────
# Read DE results for key genes
key_summary = []
for ct in ["Hepatocyte", "Kupffer_cell", "Hepatic_stellate", "Endothelial"]:
    fpath = os.path.join(TAB_DIR, f"DE_{ct}_7dpi_vs_control.csv")
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    KEY_GENES = ["Il6ra", "Stat3", "Socs3", "Saa1", "Saa2", "Crp", "Hamp",
                 "Apob", "Mttp", "Fasn", "Scd1", "Fads2", "Elovl2",
                 "Plin2", "Lpl", "Cd36", "Mgll", "Apoe"]
    for gene in KEY_GENES:
        row = df[df["gene"] == gene]
        if not row.empty:
            r = row.iloc[0]
            key_summary.append({
                "cell_type": ct,
                "gene": gene,
                "log2FC": r["log2FC"],
                "FDR": r["FDR"],
                "significant": r.get("significant", r["FDR"] < 0.05),
            })

if key_summary:
    key_df = pd.DataFrame(key_summary)
    key_df.to_csv(os.path.join(TAB_DIR, "key_genes_summary.csv"), index=False)

    # Heatmap
    key_pivot = key_df.pivot_table(index="gene", columns="cell_type", values="log2FC")
    # Keep only genes detected in at least one cell type
    key_pivot = key_pivot.dropna(how="all")

    fig, ax = plt.subplots(figsize=(10, max(6, len(key_pivot) * 0.45)))
    sns.heatmap(key_pivot, cmap="RdBu_r", center=0, ax=ax,
                linewidths=0.5, annot=True, fmt=".2f",
                annot_kws={"fontsize": 8},
                cbar_kws={"label": "log2FC (7dpi/ctrl)"},
                vmin=-2, vmax=2)
    ax.set_title("Key gene expression changes across liver cell types\n(7dpi vs Control)")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "summary_F_key_genes_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\nKey gene summary:")
    print(key_pivot.to_string())

print("\n=== Step 5 complete ===")
print(f"\nAll figures saved to: {FIG_DIR}")
print(f"All tables saved to:  {TAB_DIR}")
