#!/usr/bin/env python3
"""
GSE180862 Analysis: Brain vs Blood comparison for lipid metabolism and IL-6 signaling
- FPI model, 24h and 7d, Drop-seq scRNA-seq
- Tissues: Cortex, Hippocampus, Blood
- Focus: microglia LDAM signature, brain-peripheral IL-6, lipid metabolism
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import mmread
from scipy.sparse import csr_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/volume/data/xyma/jsonl/training/tbi/data/GSE180862")
RESULTS_DIR = Path("/volume/data/xyma/jsonl/training/tbi/results/GSE180862")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans'
})

# =============================================================================
# 1. Load all three tissues
# =============================================================================
def load_tissue(tissue_name):
    """Load Drop-seq data for a tissue."""
    print(f"  Loading {tissue_name}...")
    barcodes = pd.read_csv(DATA_DIR / f"{tissue_name}.barcodes.tsv.gz",
                           header=None, sep='\t')[0].values
    features = pd.read_csv(DATA_DIR / f"{tissue_name}.features.tsv.gz",
                           header=None, sep='\t')
    meta = pd.read_csv(DATA_DIR / f"{tissue_name}.metaData.tsv.gz", sep='\t')

    # Load sparse matrix (genes x cells format, need to transpose to cells x genes)
    raw_mtx = mmread(DATA_DIR / f"{tissue_name}.digital_expression.mtx.gz")
    print(f"    Raw MTX shape: {raw_mtx.shape}, features: {len(features)}, barcodes: {len(barcodes)}")
    # Determine orientation
    if raw_mtx.shape[0] == len(features) and raw_mtx.shape[1] == len(barcodes):
        mtx = raw_mtx.T.tocsr()  # genes x cells -> cells x genes
    elif raw_mtx.shape[0] == len(barcodes) and raw_mtx.shape[1] == len(features):
        mtx = raw_mtx.tocsr()  # already cells x genes
    else:
        print(f"    WARNING: shape mismatch, trying transpose")
        mtx = raw_mtx.T.tocsr()

    # Create AnnData
    gene_names = features[1].values if features.shape[1] >= 2 else features[0].values
    # Handle duplicate gene names
    from collections import Counter
    gene_counts = Counter(gene_names)
    seen = Counter()
    unique_names = []
    for g in gene_names:
        if gene_counts[g] > 1:
            seen[g] += 1
            unique_names.append(f"{g}_{seen[g]}")
        else:
            unique_names.append(g)
    gene_names = unique_names

    adata = sc.AnnData(X=mtx, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=gene_names))

    # Add metadata - row names in TSV are the barcodes (pandas index)
    # meta already has barcodes as index when read with sep='\t'
    common = adata.obs.index.intersection(meta.index)
    print(f"    Barcode match: {len(common)} / {adata.shape[0]}")
    adata = adata[common].copy()
    for col in meta.columns:
        adata.obs[col] = meta.loc[common, col].values

    print(f"    {adata.shape[0]} cells × {adata.shape[1]} genes")
    print(f"    Cell types: {adata.obs['CellType'].value_counts().to_dict()}")
    print(f"    Conditions: {adata.obs['Condition'].value_counts().to_dict()}")
    print(f"    Timepoints: {adata.obs['Timepoint'].value_counts().to_dict()}")

    return adata

print("Loading tissues...")
cortex = load_tissue("Cortex")
blood = load_tissue("Blood")
hip = load_tissue("Hippocampus")

# =============================================================================
# 2. Define gene sets
# =============================================================================
LDAM_GENES = ['Lpl', 'Apoe', 'Cd36', 'Fabp5', 'Fabp4', 'Plin2', 'Plin3',
              'Lipa', 'Soat1', 'Acsl1', 'Mgll']
DAM_GENES = ['Trem2', 'Tyrobp', 'Cst7', 'Spp1', 'Itgax', 'Axl', 'Lgals3',
             'Clec7a', 'Gpnmb', 'Igf1']
IL6_GENES = ['Il6', 'Il6ra', 'Il6st', 'Jak1', 'Jak2', 'Stat3', 'Socs3']
LIPID_METABOLISM = ['Fasn', 'Scd1', 'Scd2', 'Cpt1a', 'Hmgcr', 'Abca1',
                    'Nr1h3', 'Pparg', 'Mfsd2a', 'Ldlr', 'Vldlr']
ACUTE_PHASE = ['Lcn2', 'Serpina3n', 'Saa3', 'Hp', 'A2m']
INFLAMMATION = ['Il1b', 'Tnf', 'Ccl2', 'Cxcl10', 'Ptgs2']
DHA_GENES = ['Mfsd2a', 'Elovl2', 'Fads1', 'Fads2', 'Fabp7']

# =============================================================================
# 3. Normalize and compute scores
# =============================================================================
print("\nNormalizing...")
for adata in [cortex, blood, hip]:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# =============================================================================
# 4. Figure 1: LDAM/DAM scores in Microglia vs aMG, TBI vs Sham
# =============================================================================
print("\nAnalyzing microglia subsets in cortex...")

# Extract microglia
mg_mask = cortex.obs['CellType'].isin(['MG', 'aMG'])
mg_cortex = cortex[mg_mask].copy()
print(f"Cortex microglia: {mg_cortex.shape[0]} cells (MG: {(mg_cortex.obs['CellType']=='MG').sum()}, aMG: {(mg_cortex.obs['CellType']=='aMG').sum()})")

def score_genes(adata, gene_list, score_name):
    """Score gene set, handling missing genes."""
    available = [g for g in gene_list if g in adata.var_names]
    if len(available) >= 2:
        sc.tl.score_genes(adata, available, score_name=score_name)
    elif len(available) == 1:
        adata.obs[score_name] = adata[:, available[0]].X.toarray().flatten() if hasattr(adata[:, available[0]].X, 'toarray') else adata[:, available[0]].X.flatten()
    else:
        adata.obs[score_name] = 0
    return available

# Score microglia
for name, genes in [('LDAM_score', LDAM_GENES), ('DAM_score', DAM_GENES),
                     ('IL6_score', IL6_GENES), ('Lipid_score', LIPID_METABOLISM),
                     ('Inflam_score', INFLAMMATION)]:
    avail = score_genes(mg_cortex, genes, name)
    print(f"  {name}: {len(avail)}/{len(genes)} genes available")

# Figure: Box plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

score_names = ['LDAM_score', 'DAM_score', 'IL6_score', 'Lipid_score', 'Inflam_score']
titles = ['LDAM Signature', 'DAM Signature', 'IL-6 Pathway', 'Lipid Metabolism', 'Inflammation']

for idx, (score, title) in enumerate(zip(score_names, titles)):
    ax = axes.flatten()[idx]
    plot_data = mg_cortex.obs[['CellType', 'Condition', 'Timepoint', score]].copy()
    plot_data['Group'] = plot_data['CellType'] + '\n' + plot_data['Condition']

    # Order groups
    order = ['MG\nSham', 'MG\nTBI', 'aMG\nSham', 'aMG\nTBI']
    available_order = [o for o in order if o in plot_data['Group'].values]

    sns.boxplot(data=plot_data, x='Group', y=score, order=available_order,
                palette=['#90CAF9', '#E53935', '#90CAF9', '#E53935'],
                ax=ax, showfliers=False)
    sns.stripplot(data=plot_data, x='Group', y=score, order=available_order,
                  color='black', alpha=0.1, size=1, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('')

axes.flatten()[5].set_visible(False)

plt.suptitle('GSE180862: Gene Set Scores in Cortical Microglia\n(MG = homeostatic, aMG = activated, FPI model)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig1_mg_scores_boxplot.png', bbox_inches='tight')
plt.close()
print("  Saved: fig1_mg_scores_boxplot.png")

# =============================================================================
# 5. Figure 2: Key LDAM genes — MG vs aMG, TBI vs Sham, dot plot style
# =============================================================================
print("\nGenerating Figure 2: Key gene expression dot plot...")

key_genes_all = LDAM_GENES + DAM_GENES[:6] + IL6_GENES + DHA_GENES
key_genes_available = [g for g in key_genes_all if g in mg_cortex.var_names]

# Compute mean expression and fraction expressing per group
groups = mg_cortex.obs.groupby(['CellType', 'Condition'])
dotplot_data = []

for (ct, cond), idx_group in groups:
    subset = mg_cortex[idx_group.index]
    for gene in key_genes_available:
        expr = subset[:, gene].X.toarray().flatten() if hasattr(subset[:, gene].X, 'toarray') else subset[:, gene].X.flatten()
        dotplot_data.append({
            'Gene': gene, 'CellType': ct, 'Condition': cond,
            'Mean_Expr': expr.mean(),
            'Frac_Expressing': (expr > 0).mean(),
            'Group': f"{ct}_{cond}"
        })

dot_df = pd.DataFrame(dotplot_data)

# Create manual dot plot
fig, ax = plt.subplots(figsize=(18, 8))

group_order = ['MG_Sham', 'MG_TBI', 'aMG_Sham', 'aMG_TBI']
gene_order = key_genes_available

for gi, gene in enumerate(gene_order):
    for gri, group in enumerate(group_order):
        row = dot_df[(dot_df['Gene'] == gene) & (dot_df['Group'] == group)]
        if len(row) == 0:
            continue
        frac = row['Frac_Expressing'].values[0]
        mean_expr = row['Mean_Expr'].values[0]
        size = frac * 200
        color = mean_expr
        ax.scatter(gi, gri, s=size, c=color, cmap='Reds', vmin=0, vmax=dot_df['Mean_Expr'].quantile(0.95),
                   edgecolors='black', linewidths=0.5)

ax.set_xticks(range(len(gene_order)))
ax.set_xticklabels(gene_order, rotation=90, fontsize=8)
ax.set_yticks(range(len(group_order)))
ax.set_yticklabels(['MG Sham', 'MG TBI', 'aMG Sham', 'aMG TBI'])

# Add colorbar and size legend
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(0, dot_df['Mean_Expr'].quantile(0.95)))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Mean Expression', shrink=0.5)

ax.set_title('GSE180862: LDAM/DAM/IL-6/DHA Gene Expression\nin Cortical Microglia (FPI model)', fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig2_dotplot_mg.png', bbox_inches='tight')
plt.close()
print("  Saved: fig2_dotplot_mg.png")

# =============================================================================
# 6. Figure 3: Brain vs Blood comparison for IL-6 and lipid genes
# =============================================================================
print("\nAnalyzing Brain vs Blood...")

# Score all tissues
for adata, tissue_name in [(cortex, 'Cortex'), (blood, 'Blood'), (hip, 'Hippocampus')]:
    for name, genes in [('LDAM_score', LDAM_GENES), ('IL6_score', IL6_GENES),
                         ('Lipid_score', LIPID_METABOLISM)]:
        score_genes(adata, genes, name)

# Compare gene expression across tissues for specific cell types
# For brain: microglia (MG + aMG)
# For blood: all cells (or monocytes)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

compare_genes = ['Il6', 'Il6ra', 'Il6st', 'Stat3', 'Lpl', 'Cd36', 'Apoe', 'Socs3']

for idx, gene in enumerate(compare_genes):
    ax = axes.flatten()[idx]

    tissue_data = []

    # Cortex microglia
    if gene in cortex.var_names:
        mg_mask_c = cortex.obs['CellType'].isin(['MG', 'aMG'])
        for cond in ['Sham', 'TBI']:
            mask = mg_mask_c & (cortex.obs['Condition'] == cond)
            if mask.sum() > 0:
                expr = cortex[mask, gene].X.toarray().flatten() if hasattr(cortex[mask, gene].X, 'toarray') else cortex[mask, gene].X.flatten()
                for v in expr:
                    tissue_data.append({'Tissue': f'Cortex MG\n{cond}', 'Expr': v})

    # Hippocampus microglia
    if gene in hip.var_names and 'CellType' in hip.obs.columns:
        mg_mask_h = hip.obs['CellType'].isin(['MG', 'aMG'])
        for cond in ['Sham', 'TBI']:
            mask = mg_mask_h & (hip.obs['Condition'] == cond)
            if mask.sum() > 0:
                expr = hip[mask, gene].X.toarray().flatten() if hasattr(hip[mask, gene].X, 'toarray') else hip[mask, gene].X.flatten()
                for v in expr:
                    tissue_data.append({'Tissue': f'Hip MG\n{cond}', 'Expr': v})

    # Blood all cells
    if gene in blood.var_names:
        for cond in ['Sham', 'TBI']:
            mask = blood.obs['Condition'] == cond
            if mask.sum() > 0:
                expr = blood[mask, gene].X.toarray().flatten() if hasattr(blood[mask, gene].X, 'toarray') else blood[mask, gene].X.flatten()
                for v in expr:
                    tissue_data.append({'Tissue': f'Blood\n{cond}', 'Expr': v})

    if tissue_data:
        tdf = pd.DataFrame(tissue_data)
        order = [o for o in ['Cortex MG\nSham', 'Cortex MG\nTBI',
                              'Hip MG\nSham', 'Hip MG\nTBI',
                              'Blood\nSham', 'Blood\nTBI'] if o in tdf['Tissue'].values]
        colors = []
        for o in order:
            if 'TBI' in o:
                colors.append('#E53935')
            else:
                colors.append('#90CAF9')
        sns.violinplot(data=tdf, x='Tissue', y='Expr', order=order,
                       palette=colors, ax=ax, cut=0, inner='box', linewidth=0.5)
        ax.set_title(gene, fontweight='bold', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('log1p(expr)')
        ax.tick_params(axis='x', labelsize=7)

plt.suptitle('GSE180862: Brain Microglia vs Blood — IL-6 & Lipid Genes\n(FPI model, Sham vs TBI)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig3_brain_blood_comparison.png', bbox_inches='tight')
plt.close()
print("  Saved: fig3_brain_blood_comparison.png")

# =============================================================================
# 7. Figure 4: Timepoint comparison (24h vs 7d) in microglia
# =============================================================================
print("\nAnalyzing timepoint differences in microglia...")

fig, axes = plt.subplots(2, 5, figsize=(22, 9))

time_genes = ['Plin2', 'Lpl', 'Cd36', 'Apoe', 'Fabp5',
              'Il6', 'Il6ra', 'Il6st', 'Mfsd2a', 'Grn']

for idx, gene in enumerate(time_genes):
    ax = axes.flatten()[idx]
    if gene not in cortex.var_names:
        ax.set_title(f'{gene}\n(not found)', fontsize=10)
        continue

    mg_data = cortex[cortex.obs['CellType'].isin(['MG', 'aMG'])].copy()

    plot_data = []
    for tp in ['24hrs', '7days']:
        for cond in ['Sham', 'TBI']:
            mask = (mg_data.obs['Timepoint'] == tp) & (mg_data.obs['Condition'] == cond)
            if mask.sum() > 0:
                expr = mg_data[mask, gene].X.toarray().flatten() if hasattr(mg_data[mask, gene].X, 'toarray') else mg_data[mask, gene].X.flatten()
                mean_e = expr.mean()
                frac_e = (expr > 0).mean()
                plot_data.append({
                    'Timepoint': tp.replace('hrs', 'h').replace('days', 'd'),
                    'Condition': cond,
                    'Mean': mean_e, 'Frac': frac_e,
                    'Group': f"{tp}\n{cond}"
                })

    if plot_data:
        pdf = pd.DataFrame(plot_data)
        colors = ['#90CAF9', '#E53935', '#64B5F6', '#C62828']
        bars = ax.bar(range(len(pdf)), pdf['Mean'],
                      color=colors[:len(pdf)], edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(pdf)))
        ax.set_xticklabels(pdf['Group'], fontsize=7)
        # Add fraction labels
        for i, (_, row) in enumerate(pdf.iterrows()):
            ax.text(i, row['Mean'] + 0.01, f"{row['Frac']:.0%}", ha='center', fontsize=7)

    ax.set_title(gene, fontweight='bold', fontsize=11)
    ax.set_ylabel('Mean log1p(expr)')

plt.suptitle('GSE180862: Key Gene Expression in Cortical Microglia\n24h vs 7d Post-FPI (bar = mean, label = % expressing)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig4_timepoint_comparison.png', bbox_inches='tight')
plt.close()
print("  Saved: fig4_timepoint_comparison.png")

# =============================================================================
# 8. Figure 5: Differential expression summary — TBI vs Sham per cell type
# =============================================================================
print("\nComputing pseudo-bulk DE for microglia in cortex...")

# Pseudo-bulk: average expression per condition in microglia
mg_data = cortex[cortex.obs['CellType'].isin(['MG', 'aMG'])].copy()

tbi_mask = mg_data.obs['Condition'] == 'TBI'
sham_mask = mg_data.obs['Condition'] == 'Sham'

tbi_mean = np.array(mg_data[tbi_mask].X.mean(axis=0)).flatten()
sham_mean = np.array(mg_data[sham_mask].X.mean(axis=0)).flatten()

# log2FC with pseudocount
pseudo = 0.01
log2fc = np.log2((tbi_mean + pseudo) / (sham_mean + pseudo))

de_df = pd.DataFrame({
    'gene': mg_data.var_names,
    'TBI_mean': tbi_mean,
    'Sham_mean': sham_mean,
    'log2FC': log2fc
}).set_index('gene')

# Report key genes
print("\nPseudo-bulk log2FC (TBI vs Sham) in cortical microglia:")
all_key = set(LDAM_GENES + DAM_GENES + IL6_GENES + DHA_GENES + ACUTE_PHASE + INFLAMMATION)
for gene in sorted(all_key):
    if gene in de_df.index:
        fc = de_df.loc[gene, 'log2FC']
        tbi_m = de_df.loc[gene, 'TBI_mean']
        sham_m = de_df.loc[gene, 'Sham_mean']
        if abs(fc) > 0.3:
            direction = '↑' if fc > 0 else '↓'
            print(f"  {gene:12s}: log2FC={fc:+.2f} {direction}  (TBI={tbi_m:.3f}, Sham={sham_m:.3f})")

# Save DE results
de_df.to_csv(RESULTS_DIR / 'cortex_mg_pseudobulk_DE.csv')

# =============================================================================
# 9. Figure 5: Heatmap of key genes across cell types
# =============================================================================
print("\nGenerating Figure 5: Cross-cell-type heatmap...")

all_cell_types = cortex.obs['CellType'].unique()
heatmap_genes = ['Plin2', 'Lpl', 'Cd36', 'Apoe', 'Fabp5', 'Trem2', 'Spp1',
                 'Il6', 'Il6ra', 'Il6st', 'Stat3', 'Mfsd2a', 'Lcn2',
                 'Il1b', 'Tnf', 'Ccl2', 'Grn', 'Mgll']
heatmap_genes = [g for g in heatmap_genes if g in cortex.var_names]

# Mean expression per cell type, TBI only
tbi_cortex = cortex[cortex.obs['Condition'] == 'TBI']
ct_means = pd.DataFrame(index=heatmap_genes, columns=sorted(all_cell_types), dtype=float)

for ct in all_cell_types:
    ct_mask = tbi_cortex.obs['CellType'] == ct
    if ct_mask.sum() < 10:
        continue
    subset = tbi_cortex[ct_mask]
    for gene in heatmap_genes:
        if gene in subset.var_names:
            ct_means.loc[gene, ct] = np.array(subset[:, gene].X.mean(axis=0)).flatten()[0]

ct_means = ct_means.dropna(axis=1, how='all').astype(float)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(ct_means, cmap='YlOrRd', annot=True, fmt='.2f', linewidths=0.5,
            ax=ax, cbar_kws={'label': 'Mean log1p(expr)'})
ax.set_title('GSE180862: Key Gene Expression Across Cell Types\n(Cortex, TBI condition only)',
             fontweight='bold', fontsize=13)
ax.set_ylabel('Gene')
ax.set_xlabel('Cell Type')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig5_celltype_heatmap.png', bbox_inches='tight')
plt.close()
print("  Saved: fig5_celltype_heatmap.png")

# =============================================================================
# 10. Summary
# =============================================================================
print("\n" + "=" * 70)
print("GSE180862 ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
Dataset: GSE180862 (Arneson et al., 2022)
Model: Mouse FPI, 24h and 7d
Tissues analyzed: Cortex ({cortex.shape[0]} cells), Blood ({blood.shape[0]} cells), Hippocampus ({hip.shape[0]} cells)

Cortex microglia: MG={int((cortex.obs['CellType']=='MG').sum())}, aMG={int((cortex.obs['CellType']=='aMG').sum())}

Key pseudo-bulk DE results (TBI vs Sham, cortical MG+aMG):
""")

for gene_set_name, gene_set in [('LDAM', LDAM_GENES), ('IL-6', IL6_GENES), ('DHA', DHA_GENES)]:
    print(f"  {gene_set_name}:")
    for gene in gene_set:
        if gene in de_df.index and abs(de_df.loc[gene, 'log2FC']) > 0.1:
            fc = de_df.loc[gene, 'log2FC']
            print(f"    {gene:10s}: {fc:+.2f}")

print(f"\nAll results saved to: {RESULTS_DIR}")
