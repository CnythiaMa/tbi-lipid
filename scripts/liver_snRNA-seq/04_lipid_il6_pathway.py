"""
Step 4: Focused analysis on lipid metabolism, IL-6/STAT3 signaling,
LDAM markers, and brain-liver axis genes
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

adata = sc.read_h5ad(os.path.join(RESULT_DIR, "liver_snRNA_annotated.h5ad"))
print(f"Loaded: {adata.n_obs} nuclei")

# ── Gene sets of interest ─────────────────────────────────────────────────────
PATHWAYS = {
    # IL-6/STAT3 acute phase response in hepatocytes
    "IL6_STAT3_acute_phase": [
        "Il6ra", "Il6st", "Stat3", "Socs1", "Socs3",
        "Crp", "Saa1", "Saa2", "Saa3", "Hamp", "Hp",
        "Fga", "Fgb", "Fgg", "Orm1", "Orm2", "Apcs",
        "Lcn2", "Serpina3n", "A2m",
    ],
    # Hepatic lipid synthesis & VLDL secretion
    "Lipid_synthesis_VLDL": [
        "Fasn", "Acaca", "Acacb", "Scd1", "Scd2",
        "Dgat1", "Dgat2", "Gpam", "Agpat2",
        "Apob", "Mttp", "Apoc3", "Apoc2", "Apoa1", "Apoa2",
        "Apoe", "Apoa4",
    ],
    # DHA / PUFA synthesis
    "DHA_PUFA_synthesis": [
        "Fads1", "Fads2", "Elovl2", "Elovl4", "Elovl5", "Elovl6",
        "Pla2g4a", "Pla2g4b", "Lpla2", "Mfsd2a",
        "Acsl6", "Acsl1", "Acsl4",
    ],
    # Fatty acid beta-oxidation
    "FA_oxidation": [
        "Cpt1a", "Cpt2", "Hadha", "Hadhb", "Acadm", "Acadl",
        "Acadvl", "Acads", "Echs1", "Ehhadh", "Acox1", "Acox3",
        "Ppara", "Ppard",
    ],
    # Lipid uptake & transport
    "Lipid_uptake_transport": [
        "Cd36", "Fabp1", "Fabp4", "Fabp5", "Ldlr", "Vldlr",
        "Lpl", "Lipc", "Lipa", "Scarb1", "Abca1", "Abcg8",
        "Npc1", "Npc2", "Soat1", "Soat2",
    ],
    # Lipid droplet biology
    "Lipid_droplet_LDAM": [
        "Plin1", "Plin2", "Plin3", "Plin4", "Plin5",
        "Mgll", "Atgl", "Lipe", "Abdh5",
        "Rab18", "Rab7", "Sqstm1", "Lc3b",
    ],
    # Cholesterol metabolism
    "Cholesterol_bile_acid": [
        "Hmgcr", "Mvk", "Fdft1", "Cyp51", "Dhcr7",
        "Cyp7a1", "Cyp7b1", "Cyp8b1", "Cyp27a1",
        "Abcb11", "Slc10a1", "Abcc2",
        "Nr1h4", "Nr1h3",
    ],
    # Inflammatory signaling
    "Inflammation_TBI": [
        "Tnf", "Il1b", "Il6", "Il10", "Il18",
        "Tgfb1", "Tgfb2", "Nlrp3", "Pycard",
        "Nfkb1", "Rela", "Ccl2", "Cxcl10",
        "Hmgb1", "S100a8", "S100a9",
    ],
    # Hepatic glucose / energy metabolism (context)
    "Glucose_energy": [
        "G6pc", "Pck1", "Pck2", "Gck", "Pfkl",
        "Aldob", "Ldha", "Ldhb", "Ppargc1a", "Ppargc1b",
        "Foxo1", "Akt1", "Insr", "Irs1", "Irs2",
    ],
    # Kupffer-specific LDAM-like
    "Kupffer_LDAM": [
        "Clec4f", "Timd4", "Csf1r", "Cx3cr1",
        "Plin2", "Lpl", "Cd36", "Fabp5",
        "Mgll", "Trem2", "Apoe", "Lgals3",
    ],
}

# ── Helper: compute pathway statistics per condition × cell type ───────────────
def pathway_stats(adata, genes, cell_type, condition_key="condition"):
    """Return mean expression of each gene in ctrl and 7dpi for a cell type."""
    sub = adata[adata.obs["cell_type_broad"] == cell_type]
    valid = [g for g in genes if g in sub.raw.var_names]
    rows = []
    for g in valid:
        idx = list(sub.raw.var_names).index(g)
        from scipy.sparse import issparse
        X = sub.raw.X
        col = X[:, idx].toarray().flatten() if issparse(X) else X[:, idx]
        for cond in ["control", "7dpi"]:
            mask = sub.obs[condition_key] == cond
            rows.append({
                "gene": g,
                "condition": cond,
                "mean_expr": col[mask].mean(),
                "pct_expr": (col[mask] > 0).mean() * 100,
                "n_cells": mask.sum(),
            })
    return pd.DataFrame(rows)

# ── IL-6/STAT3 pathway in hepatocytes (KEY QUESTION) ─────────────────────────
print("\n=== IL-6/STAT3 Acute Phase Response in Hepatocytes ===")
il6_df = pathway_stats(adata, PATHWAYS["IL6_STAT3_acute_phase"], "Hepatocyte")
if not il6_df.empty:
    il6_pivot = il6_df.pivot_table(index="gene", columns="condition", values="mean_expr")
    il6_pivot["log2FC"] = np.log2(
        (il6_pivot.get("7dpi", 0) + 1e-4) / (il6_pivot.get("control", 0) + 1e-4)
    )
    il6_pivot = il6_pivot.sort_values("log2FC", ascending=False)
    il6_pivot.to_csv(os.path.join(TAB_DIR, "hepatocyte_IL6_STAT3_expression.csv"))
    print(il6_pivot.to_string())

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    il6_plot = il6_pivot["log2FC"].sort_values(ascending=True)
    colors = ["red" if x > 0 else "blue" for x in il6_plot]
    il6_plot.plot(kind="barh", color=colors, ax=ax)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("log2FC (7dpi / control)")
    ax.set_title("Hepatocyte: IL-6/STAT3 Pathway (7dpi vs Control)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "hepatocyte_IL6_STAT3_barplot.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ── Lipid synthesis pathway ───────────────────────────────────────────────────
print("\n=== Lipid Synthesis & VLDL Secretion in Hepatocytes ===")
lipid_df = pathway_stats(adata, PATHWAYS["Lipid_synthesis_VLDL"], "Hepatocyte")
if not lipid_df.empty:
    lipid_pivot = lipid_df.pivot_table(index="gene", columns="condition", values="mean_expr")
    lipid_pivot["log2FC"] = np.log2(
        (lipid_pivot.get("7dpi", 0) + 1e-4) / (lipid_pivot.get("control", 0) + 1e-4)
    )
    lipid_pivot = lipid_pivot.sort_values("log2FC", ascending=False)
    lipid_pivot.to_csv(os.path.join(TAB_DIR, "hepatocyte_lipid_synthesis.csv"))
    print(lipid_pivot.to_string())

# ── DHA/PUFA synthesis ────────────────────────────────────────────────────────
print("\n=== DHA/PUFA Synthesis Pathway in Hepatocytes ===")
dha_df = pathway_stats(adata, PATHWAYS["DHA_PUFA_synthesis"], "Hepatocyte")
if not dha_df.empty:
    dha_pivot = dha_df.pivot_table(index="gene", columns="condition", values="mean_expr")
    dha_pivot["log2FC"] = np.log2(
        (dha_pivot.get("7dpi", 0) + 1e-4) / (dha_pivot.get("control", 0) + 1e-4)
    )
    dha_pivot = dha_pivot.sort_values("log2FC", ascending=False)
    dha_pivot.to_csv(os.path.join(TAB_DIR, "hepatocyte_DHA_PUFA.csv"))
    print(dha_pivot.to_string())

# ── LDAM markers in Kupffer cells ─────────────────────────────────────────────
print("\n=== LDAM Markers in Kupffer Cells ===")
kupffer_df = pathway_stats(adata, PATHWAYS["Kupffer_LDAM"], "Kupffer_cell")
if not kupffer_df.empty:
    kupffer_pivot = kupffer_df.pivot_table(index="gene", columns="condition", values="mean_expr")
    kupffer_pivot["log2FC"] = np.log2(
        (kupffer_pivot.get("7dpi", 0) + 1e-4) / (kupffer_pivot.get("control", 0) + 1e-4)
    )
    kupffer_pivot = kupffer_pivot.sort_values("log2FC", ascending=False)
    kupffer_pivot.to_csv(os.path.join(TAB_DIR, "kupffer_LDAM_expression.csv"))
    print(kupffer_pivot.to_string())

# ── Multi-pathway heatmap for hepatocytes ─────────────────────────────────────
print("\nGenerating multi-pathway heatmap...")

# Collect all pathway log2FCs for hepatocytes
pathway_rows = []
for pathway_name, genes in PATHWAYS.items():
    if pathway_name == "Kupffer_LDAM":
        continue
    df = pathway_stats(adata, genes, "Hepatocyte")
    if df.empty:
        continue
    pivot = df.pivot_table(index="gene", columns="condition", values="mean_expr")
    pivot["log2FC"] = np.log2(
        (pivot.get("7dpi", 0) + 1e-4) / (pivot.get("control", 0) + 1e-4)
    )
    for gene in pivot.index:
        pathway_rows.append({
            "pathway": pathway_name,
            "gene": gene,
            "log2FC": pivot.loc[gene, "log2FC"],
            "mean_control": pivot.loc[gene, "control"] if "control" in pivot.columns else 0,
            "mean_7dpi": pivot.loc[gene, "7dpi"] if "7dpi" in pivot.columns else 0,
        })

all_pathway_df = pd.DataFrame(pathway_rows)
all_pathway_df.to_csv(os.path.join(TAB_DIR, "hepatocyte_all_pathways_log2FC.csv"), index=False)

# Pivot for heatmap
if not all_pathway_df.empty:
    heatmap_data = all_pathway_df.pivot_table(index="gene", columns="pathway", values="log2FC")
    # Only show genes with at least one pathway |log2FC| > 0.3
    row_max = heatmap_data.abs().max(axis=1)
    heatmap_data = heatmap_data[row_max > 0.3]
    heatmap_data = heatmap_data.sort_values(list(heatmap_data.columns)[0], ascending=False)

    n_genes = len(heatmap_data)
    fig, ax = plt.subplots(figsize=(12, max(8, n_genes * 0.28)))
    sns.heatmap(heatmap_data, cmap="RdBu_r", center=0, ax=ax,
                linewidths=0.3, vmin=-2, vmax=2,
                cbar_kws={"label": "log2FC (7dpi/ctrl)"})
    ax.set_title("Hepatocyte pathway gene expression changes (7dpi vs control)")
    ax.set_xlabel("Pathway")
    ax.set_ylabel("Gene")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "hepatocyte_pathway_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ── Violin plots for key genes ─────────────────────────────────────────────────
KEY_GENES = ["Saa1", "Saa2", "Socs3", "Crp", "Stat3", "Il6ra",
             "Fasn", "Scd1", "Apob", "Mttp", "Fads2", "Elovl2",
             "Plin2", "Lpl", "Cd36", "Mgll"]

hep_adata = adata[adata.obs["cell_type_broad"] == "Hepatocyte"].copy()
valid_key_genes = [g for g in KEY_GENES if g in hep_adata.raw.var_names]

if len(valid_key_genes) > 0:
    sc.pl.violin(hep_adata, valid_key_genes, groupby="condition",
                 use_raw=True, show=False, rotation=45)
    plt.suptitle("Key genes: Hepatocyte, control vs 7dpi")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "violin_hepatocyte_key_genes.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ── Score LDAM and acute-phase signatures on all cells ──────────────────────
print("\nScoring LDAM and acute-phase gene signatures...")
ldam_genes = ["Plin2", "Lpl", "Cd36", "Fabp5", "Apoe", "Trem2"]
acute_genes = ["Saa1", "Saa2", "Crp", "Orm1", "Hamp", "Serpina3n"]
dha_genes   = ["Fads2", "Elovl2", "Elovl5", "Acsl6"]

for name, genes in [("LDAM_score", ldam_genes), ("AcutePhase_score", acute_genes), ("DHA_synth_score", dha_genes)]:
    valid = [g for g in genes if g in adata.raw.var_names]
    if len(valid) > 1:
        sc.tl.score_genes(adata, gene_list=valid, score_name=name, use_raw=True)
        sc.pl.umap(adata, color=name, show=False, title=name, cmap="RdBu_r", vcenter=0)
        plt.savefig(os.path.join(FIG_DIR, f"umap_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {name}: scored on {len(valid)} genes")

# Score comparison violin
score_names = [s for s in ["LDAM_score", "AcutePhase_score", "DHA_synth_score"]
               if s in adata.obs.columns]
if score_names:
    fig, axes = plt.subplots(1, len(score_names), figsize=(6 * len(score_names), 5))
    if len(score_names) == 1:
        axes = [axes]
    for ax, score in zip(axes, score_names):
        data = [(adata.obs[score][adata.obs["condition"] == cond].values,
                 adata.obs[score][adata.obs["cell_type_broad"] == ct].values)
                for cond in ["control", "7dpi"]
                for ct in ["Hepatocyte", "Kupffer_cell"]]
        # Simple condition comparison per cell type
        for ct, ct_color in [("Hepatocyte", "steelblue"), ("Kupffer_cell", "coral")]:
            for cond, fill in [("control", False), ("7dpi", True)]:
                mask = (adata.obs["condition"] == cond) & (adata.obs["cell_type_broad"] == ct)
                vals = adata.obs[score][mask].values
                label = f"{ct} {cond}"
                ax.violinplot([vals], positions=[list(["control", "7dpi"]).index(cond) +
                              (0 if ct == "Hepatocyte" else 0.4)],
                              showmedians=True)
        ax.set_title(score)
        ax.set_xticks([0, 0.2, 1, 1.2])
        ax.set_xticklabels(["Hep\nctl", "Kup\nctl", "Hep\n7dpi", "Kup\n7dpi"], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "signature_scores_violin.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ── Save updated adata ────────────────────────────────────────────────────────
adata.write_h5ad(os.path.join(RESULT_DIR, "liver_snRNA_final.h5ad"))

print("\n=== Step 4 complete ===")
