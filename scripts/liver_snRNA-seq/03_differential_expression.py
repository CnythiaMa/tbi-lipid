"""
Step 3: Pseudo-bulk differential expression analysis
Control vs 7dpi per major cell type
Uses DESeq2-style approach via pydeseq2 if available, otherwise Wilcoxon rank-sum
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.sparse import issparse
import os, warnings
warnings.filterwarnings('ignore')

RESULT_DIR = "/Users/maxue/Documents/vscode/tbi/results/liver_snRNA-seq"
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
TAB_DIR    = os.path.join(RESULT_DIR, "tables")

adata = sc.read_h5ad(os.path.join(RESULT_DIR, "liver_snRNA_annotated.h5ad"))
print(f"Loaded: {adata.n_obs} nuclei")
print("Cell types:", adata.obs["cell_type_broad"].value_counts().to_dict())

# ── Wilcoxon rank-sum DE per cell type ───────────────────────────────────────
CELL_TYPES_TO_TEST = ["Hepatocyte", "Kupffer_cell", "Hepatic_stellate",
                       "Endothelial", "Cholangiocyte", "Monocyte_macrophage"]

all_de_results = {}

for ct in CELL_TYPES_TO_TEST:
    mask = adata.obs["cell_type_broad"] == ct
    sub = adata[mask].copy()
    n_ctl = (sub.obs["condition"] == "control").sum()
    n_dpi = (sub.obs["condition"] == "7dpi").sum()
    print(f"\n{ct}: {n_ctl} control, {n_dpi} 7dpi nuclei")

    if n_ctl < 10 or n_dpi < 10:
        print(f"  Skipping {ct}: too few cells")
        continue

    # Subset to expressed genes (>0 in >5% of cells in this type)
    X = sub.raw.X if sub.raw is not None else sub.X
    if issparse(X):
        X = X.toarray()
    frac_expr = (X > 0).mean(axis=0)
    gene_mask = frac_expr > 0.05
    genes_use = np.array(sub.raw.var_names if sub.raw is not None else sub.var_names)[gene_mask]
    X_filt = X[:, gene_mask]

    # Wilcoxon test: 7dpi vs control
    ctl_idx = sub.obs["condition"].values == "control"
    dpi_idx = sub.obs["condition"].values == "7dpi"
    X_ctl = X_filt[ctl_idx, :]
    X_dpi = X_filt[dpi_idx, :]

    results = []
    for i, gene in enumerate(genes_use):
        vals_ctl = X_ctl[:, i]
        vals_dpi = X_dpi[:, i]
        stat, pval = stats.mannwhitneyu(vals_dpi, vals_ctl, alternative="two-sided")

        mean_ctl = vals_ctl.mean()
        mean_dpi = vals_dpi.mean()
        # Log2FC based on normalized mean expression
        log2fc = np.log2((mean_dpi + 1e-4) / (mean_ctl + 1e-4))

        results.append({
            "gene": gene,
            "mean_control": mean_ctl,
            "mean_7dpi": mean_dpi,
            "log2FC": log2fc,
            "pval": pval,
            "n_control": ctl_idx.sum(),
            "n_7dpi": dpi_idx.sum(),
        })

    df = pd.DataFrame(results)
    # BH FDR correction
    from scipy.stats import rankdata
    n = len(df)
    sorted_idx = np.argsort(df["pval"].values)
    sorted_pvals = df["pval"].values[sorted_idx]
    ranks = np.arange(1, n + 1)
    fdr = np.minimum(1, sorted_pvals * n / ranks)
    # Accumulate from right
    for j in range(n - 2, -1, -1):
        fdr[j] = min(fdr[j], fdr[j + 1])
    fdr_full = np.empty(n)
    fdr_full[sorted_idx] = fdr
    df["FDR"] = fdr_full

    df = df.sort_values("pval")
    df["significant"] = (df["FDR"] < 0.05) & (df["log2FC"].abs() > 0.5)

    out_path = os.path.join(TAB_DIR, f"DE_{ct}_7dpi_vs_control.csv")
    df.to_csv(out_path, index=False)
    all_de_results[ct] = df

    sig = df[df["significant"]]
    up = sig[sig["log2FC"] > 0]
    dn = sig[sig["log2FC"] < 0]
    print(f"  Significant DE genes: {len(sig)} ({len(up)} up, {len(dn)} down)")
    if len(up) > 0:
        print(f"  Top 10 UP: {', '.join(up.head(10)['gene'].tolist())}")
    if len(dn) > 0:
        print(f"  Top 10 DOWN: {', '.join(dn.head(10)['gene'].tolist())}")

# ── Volcano plots ─────────────────────────────────────────────────────────────
n_types = len(all_de_results)
if n_types > 0:
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5))
    if n_types == 1:
        axes = [axes]

    for ax, (ct, df) in zip(axes, all_de_results.items()):
        colors = np.where(
            (df["FDR"] < 0.05) & (df["log2FC"] > 0.5), "red",
            np.where((df["FDR"] < 0.05) & (df["log2FC"] < -0.5), "blue", "grey")
        )
        ax.scatter(df["log2FC"], -np.log10(df["pval"] + 1e-300),
                   c=colors, alpha=0.4, s=8, linewidths=0)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(-0.5, color="blue", linestyle="--", linewidth=0.8)
        ax.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("log2FC (7dpi / control)")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(ct)

        # Label top genes
        top_label = df[(df["FDR"] < 0.05) & (df["log2FC"].abs() > 1)].head(12)
        for _, row in top_label.iterrows():
            ax.text(row["log2FC"], -np.log10(row["pval"] + 1e-300),
                    row["gene"], fontsize=5, ha="center")

    plt.suptitle("Volcano: 7dpi vs Control (per cell type)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "volcano_all_celltypes.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ── Heatmap: top DE genes in Hepatocytes ─────────────────────────────────────
if "Hepatocyte" in all_de_results:
    df_hep = all_de_results["Hepatocyte"]
    top_up = df_hep[df_hep["significant"] & (df_hep["log2FC"] > 0)].head(25)["gene"].tolist()
    top_dn = df_hep[df_hep["significant"] & (df_hep["log2FC"] < 0)].head(25)["gene"].tolist()
    top_genes = top_up + top_dn

    sub_hep = adata[adata.obs["cell_type_broad"] == "Hepatocyte"].copy()
    valid_genes = [g for g in top_genes if g in sub_hep.raw.var_names]

    if len(valid_genes) > 4:
        sc.pl.heatmap(sub_hep, valid_genes, groupby="condition",
                      use_raw=True, standard_scale="var", show=False,
                      cmap="RdBu_r", figsize=(14, max(6, len(valid_genes) * 0.3)))
        plt.title("Hepatocyte: top DE genes 7dpi vs control")
        plt.savefig(os.path.join(FIG_DIR, "heatmap_hepatocyte_DE.png"), dpi=150, bbox_inches="tight")
        plt.close()

# ── Summary table: DE gene counts per cell type ───────────────────────────────
summary_rows = []
for ct, df in all_de_results.items():
    sig = df[df["significant"]]
    summary_rows.append({
        "cell_type": ct,
        "total_tested": len(df),
        "n_up": (sig["log2FC"] > 0).sum(),
        "n_down": (sig["log2FC"] < 0).sum(),
        "n_sig_total": len(sig),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TAB_DIR, "DE_summary.csv"), index=False)
print("\n=== DE Summary ===")
print(summary_df.to_string(index=False))

print("\n=== Step 3 complete ===")
