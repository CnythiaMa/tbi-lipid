#!/usr/bin/env python3
"""
GSE226211: Endothelial cell Mfsd2a expression analysis
+ cross-cell-type comparison across Intact / 3dpi / 5dpi
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import mmread
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/volume/data/xyma/jsonl/training/tbi/data/GSE226211")
RESULTS_DIR = Path("/volume/data/xyma/jsonl/training/tbi/results/GSE226211")

plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans'
})

# Sample metadata
SAMPLE_META = {
    'GSM7068147_MUC13721': {'condition': 'Intact', 'timepoint': 'Intact'},
    'GSM7068148_MUC13722': {'condition': '3dpi_CTRL', 'timepoint': '3dpi'},
    'GSM7068149_MUC13723': {'condition': '3dpi_CTRL', 'timepoint': '3dpi'},
    'GSM7068150_MUC13724': {'condition': '3dpi_INH', 'timepoint': '3dpi'},
    'GSM7068151_MUC13725': {'condition': '3dpi_INH', 'timepoint': '3dpi'},
    'GSM7068152_MUC13726': {'condition': 'Intact', 'timepoint': 'Intact'},
    'GSM7068153_MUC13727': {'condition': '5dpi_CTRL', 'timepoint': '5dpi'},
    'GSM7068154_MUC13729': {'condition': '5dpi_INH', 'timepoint': '5dpi'},
    'GSM7068155_MUC13730': {'condition': '5dpi_INH', 'timepoint': '5dpi'},
    'GSM7068156_MUC13731': {'condition': 'Intact', 'timepoint': 'Intact'},
    'GSM7068157_MUC13732': {'condition': 'Intact', 'timepoint': 'Intact'},
    'GSM7068158_MUC18415': {'condition': '5dpi_CTRL', 'timepoint': '5dpi'},
    'GSM7068159_MUC29190': {'condition': 'Intact', 'timepoint': 'Intact'},
    'GSM7068160_21L008532': {'condition': '5dpi_CTRL', 'timepoint': '5dpi'},
    'GSM7068161_21L008533': {'condition': '5dpi_INH', 'timepoint': '5dpi'},
}

# Only CTRL (no inhibitor)
CTRL_SAMPLES = {k: v for k, v in SAMPLE_META.items()
                if v['condition'] in ['Intact', '3dpi_CTRL', '5dpi_CTRL']}

# =============================================================================
# 1. Load all CTRL samples
# =============================================================================
print("Loading scRNA-seq data...")
adatas = []
for sample_id, meta in CTRL_SAMPLES.items():
    prefix = DATA_DIR / sample_id
    barcodes = pd.read_csv(f"{prefix}_barcodes.tsv.gz", header=None, sep='\t')[0].values
    features = pd.read_csv(f"{prefix}_features.tsv.gz", header=None, sep='\t')
    mtx = mmread(f"{prefix}_matrix.mtx.gz")
    gene_names = features[1].values
    adata = sc.AnnData(X=mtx.T.tocsr(),
                       obs=pd.DataFrame(index=[f"{sample_id}_{b}" for b in barcodes]),
                       var=pd.DataFrame(index=gene_names))
    adata.var_names_make_unique()
    adata.obs['sample'] = sample_id
    adata.obs['condition'] = meta['condition']
    adata.obs['timepoint'] = meta['timepoint']
    adatas.append(adata)

adata = sc.concat(adatas, join='inner')
adata.var_names_make_unique()
print(f"Total: {adata.shape[0]} cells × {adata.shape[1]} genes")

# Preprocess
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()

# HVG + PCA + clustering
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata_hvg = adata[:, adata.var['highly_variable']].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=50)
sc.pp.neighbors(adata_hvg, n_pcs=30)
sc.tl.umap(adata_hvg)
sc.tl.leiden(adata_hvg, resolution=0.8)
adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
adata.obs['leiden'] = adata_hvg.obs['leiden']

# Cell type assignment using markers
markers = {
    'Microglia': ['Tmem119', 'P2ry12', 'Cx3cr1', 'Hexb', 'Csf1r', 'Aif1'],
    'Astrocyte': ['Gfap', 'Aqp4', 'Aldh1l1', 'S100b', 'Slc1a3'],
    'Oligodendrocyte': ['Mbp', 'Plp1', 'Mog', 'Cnp'],
    'OPC': ['Pdgfra', 'Cspg4', 'Sox10'],
    'Neuron': ['Rbfox3', 'Snap25', 'Syt1', 'Stmn2'],
    'Endothelial': ['Pecam1', 'Cldn5', 'Flt1', 'Tie1', 'Kdr'],
    'Macrophage': ['Ccr2', 'Lyz2', 'Ms4a7'],
    'Pericyte': ['Pdgfrb', 'Rgs5', 'Kcnj8'],
}

for ct, genes in markers.items():
    avail = [g for g in genes if g in adata.var_names]
    if avail:
        sc.tl.score_genes(adata, avail, score_name=f'{ct}_score')

marker_scores = [f'{ct}_score' for ct in markers if f'{ct}_score' in adata.obs.columns]
cluster_ct = {}
for cl in adata.obs['leiden'].unique():
    mask = adata.obs['leiden'] == cl
    scores = {s.replace('_score', ''): adata.obs.loc[mask, s].mean() for s in marker_scores}
    cluster_ct[cl] = max(scores, key=scores.get)

adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_ct)
print("\nCell type counts:")
print(adata.obs['cell_type'].value_counts())

# =============================================================================
# 2. Extract endothelial cells
# =============================================================================
endo = adata[adata.obs['cell_type'] == 'Endothelial'].copy()
print(f"\nEndothelial cells: {endo.shape[0]}")
print(f"  Per condition: {endo.obs['condition'].value_counts().to_dict()}")

# =============================================================================
# 3. Mfsd2a and related genes in endothelial cells
# =============================================================================
# BBB transport and integrity genes
BBB_GENES = [
    'Mfsd2a',       # DHA transporter (LPC-DHA)
    'Slc2a1',       # Glut1, glucose transporter (BBB marker)
    'Abcb1a',       # P-glycoprotein (efflux)
    'Cldn5',        # Claudin-5 (tight junction, BBB integrity)
    'Ocln',         # Occludin (tight junction)
    'Tjp1',         # ZO-1 (tight junction)
    'Pecam1',       # CD31 (endothelial marker)
    'Flt1',         # VEGFR1
    'Kdr',          # VEGFR2
    'Icam1',        # ICAM-1 (inflammation, leukocyte adhesion)
    'Vcam1',        # VCAM-1 (inflammation)
    'Sele',         # E-selectin (inflammation)
    'Mmp9',         # Matrix metalloproteinase 9 (BBB breakdown)
]

# Also check in all cell types for comparison
ALL_CELL_TYPES = sorted(adata.obs['cell_type'].unique())

print("\n=== Mfsd2a expression across cell types and conditions ===\n")

# Per cell type, per condition
results = []
for ct in ALL_CELL_TYPES:
    for cond in ['Intact', '3dpi_CTRL', '5dpi_CTRL']:
        mask = (adata.obs['cell_type'] == ct) & (adata.obs['condition'] == cond)
        n_cells = mask.sum()
        if n_cells < 10:
            continue
        for gene in ['Mfsd2a']:
            if gene not in adata.var_names:
                continue
            expr = adata[mask, gene].X.toarray().flatten()
            results.append({
                'cell_type': ct, 'condition': cond, 'gene': gene,
                'n_cells': int(n_cells),
                'mean_expr': expr.mean(),
                'frac_expressing': (expr > 0).mean(),
                'median_expr': np.median(expr[expr > 0]) if (expr > 0).any() else 0,
            })

mfsd2a_df = pd.DataFrame(results)
print(mfsd2a_df.to_string(index=False))

# =============================================================================
# 4. Endothelial-specific detailed analysis
# =============================================================================
print("\n\n=== Endothelial BBB gene expression by condition ===\n")

endo_results = []
for gene in BBB_GENES:
    if gene not in adata.var_names:
        continue
    for cond in ['Intact', '3dpi_CTRL', '5dpi_CTRL']:
        mask = (endo.obs['condition'] == cond)
        if mask.sum() < 5:
            continue
        expr = endo[mask, gene].X.toarray().flatten()
        intact_mask = endo.obs['condition'] == 'Intact'
        intact_expr = endo[intact_mask, gene].X.toarray().flatten()

        # Calculate pseudo log2FC
        pseudo = 0.001
        fc = np.log2((expr.mean() + pseudo) / (intact_expr.mean() + pseudo)) if cond != 'Intact' else 0

        # Wilcoxon test vs Intact
        if cond != 'Intact' and intact_mask.sum() > 10:
            try:
                stat, pval = stats.mannwhitneyu(expr, intact_expr, alternative='two-sided')
            except:
                pval = 1.0
        else:
            pval = np.nan

        endo_results.append({
            'gene': gene, 'condition': cond,
            'mean_expr': expr.mean(),
            'frac_expressing': (expr > 0).mean(),
            'log2FC_vs_Intact': fc,
            'pval_vs_Intact': pval,
            'n_cells': int(mask.sum()),
        })

endo_df = pd.DataFrame(endo_results)
print(endo_df.to_string(index=False))
endo_df.to_csv(RESULTS_DIR / 'endothelial_bbb_genes.csv', index=False)

# =============================================================================
# 5. Figure 5: Mfsd2a across cell types and conditions
# =============================================================================
print("\nGenerating Figure 5: Mfsd2a cross-cell-type analysis...")

gene = 'Mfsd2a'
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Mfsd2a expression by cell type (all conditions pooled)
ax = axes[0]
ct_means = []
for ct in ALL_CELL_TYPES:
    mask = adata.obs['cell_type'] == ct
    if mask.sum() < 10 and gene in adata.var_names:
        continue
    expr = adata[mask, gene].X.toarray().flatten()
    ct_means.append({'cell_type': ct, 'mean': expr.mean(), 'frac': (expr > 0).mean()})
ct_df = pd.DataFrame(ct_means).sort_values('mean', ascending=True)

colors = ['#E53935' if ct == 'Endothelial' else '#1E88E5' if ct == 'Microglia' else '#90CAF9'
          for ct in ct_df['cell_type']]
bars = ax.barh(ct_df['cell_type'], ct_df['mean'], color=colors, edgecolor='black', linewidth=0.5)
for i, (_, row) in enumerate(ct_df.iterrows()):
    ax.text(row['mean'] + 0.01, i, f"{row['frac']:.0%}", va='center', fontsize=9)
ax.set_xlabel('Mean log1p(expr)')
ax.set_title('Mfsd2a Expression\nby Cell Type (all conditions)', fontweight='bold')

# Panel B: Mfsd2a in Endothelial by condition
ax = axes[1]
endo_conds = ['Intact', '3dpi_CTRL', '5dpi_CTRL']
endo_expr_data = []
for cond in endo_conds:
    mask = endo.obs['condition'] == cond
    if mask.sum() > 0:
        expr = endo[mask, gene].X.toarray().flatten()
        for v in expr:
            endo_expr_data.append({'Condition': cond, 'Expression': v})

edf = pd.DataFrame(endo_expr_data)
sns.violinplot(data=edf, x='Condition', y='Expression', order=endo_conds,
               palette=['#4CAF50', '#FF9800', '#E53935'], ax=ax, cut=0, inner='box')

# Add stats
for i, cond in enumerate(endo_conds):
    vals = edf[edf['Condition'] == cond]['Expression']
    frac = (vals > 0).mean()
    mean_v = vals.mean()
    ax.text(i, ax.get_ylim()[1] * 0.95, f'mean={mean_v:.3f}\n{frac:.0%} expr',
            ha='center', fontsize=8, color='blue')

# Add significance
for cond_idx, cond in enumerate(['3dpi_CTRL', '5dpi_CTRL']):
    row = endo_df[(endo_df['gene'] == 'Mfsd2a') & (endo_df['condition'] == cond)]
    if len(row) > 0:
        pval = row['pval_vs_Intact'].values[0]
        fc = row['log2FC_vs_Intact'].values[0]
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
        ax.text(cond_idx + 1, ax.get_ylim()[1] * 0.8,
                f'FC={fc:+.2f}\np={pval:.1e}\n{sig}',
                ha='center', fontsize=8, color='red', fontweight='bold')

ax.set_title('Mfsd2a in Endothelial Cells\n(Intact vs 3dpi vs 5dpi)', fontweight='bold')
ax.set_ylabel('log1p(expr)')

# Panel C: Mfsd2a in Microglia vs Endothelial by condition
ax = axes[2]
compare_data = []
for ct in ['Endothelial', 'Microglia']:
    for cond in endo_conds:
        mask = (adata.obs['cell_type'] == ct) & (adata.obs['condition'] == cond)
        if mask.sum() < 5:
            continue
        expr = adata[mask, gene].X.toarray().flatten()
        for v in expr:
            compare_data.append({'Cell_Condition': f'{ct[:4]}\n{cond.replace("_CTRL","")}',
                                 'Expression': v, 'CellType': ct})

cdf = pd.DataFrame(compare_data)
order = [f'Endo\n{c}' for c in ['Intact', '3dpi', '5dpi']] + \
        [f'Micr\n{c}' for c in ['Intact', '3dpi', '5dpi']]
available_order = [o for o in order if o in cdf['Cell_Condition'].values]
palette = {o: '#4CAF50' if 'Intact' in o else '#FF9800' if '3dpi' in o else '#E53935'
           for o in available_order}
sns.boxplot(data=cdf, x='Cell_Condition', y='Expression', order=available_order,
            palette=palette, ax=ax, showfliers=False)

ax.axvline(x=2.5, color='black', linestyle='--', alpha=0.5)
ax.text(1, ax.get_ylim()[1] * 0.95, 'Endothelial', ha='center', fontsize=10, fontweight='bold')
ax.text(4, ax.get_ylim()[1] * 0.95, 'Microglia', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Mfsd2a: Endothelial vs Microglia\nby Condition', fontweight='bold')
ax.set_ylabel('log1p(expr)')
ax.tick_params(axis='x', labelsize=8)

plt.suptitle('GSE226211: Mfsd2a (DHA Transporter) Expression — Cortical Stab Wound',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig5_mfsd2a_endothelial.png', bbox_inches='tight')
plt.close()
print("  Saved: fig5_mfsd2a_endothelial.png")

# =============================================================================
# 6. Figure 6: BBB integrity genes in endothelial cells
# =============================================================================
print("\nGenerating Figure 6: BBB gene heatmap...")

bbb_available = [g for g in BBB_GENES if g in adata.var_names]

# Build FC matrix: 3dpi and 5dpi vs Intact
fc_mat = pd.DataFrame(index=bbb_available, columns=['3dpi_CTRL', '5dpi_CTRL'], dtype=float)
frac_mat = pd.DataFrame(index=bbb_available, columns=['Intact', '3dpi_CTRL', '5dpi_CTRL'], dtype=float)
pval_mat = pd.DataFrame(index=bbb_available, columns=['3dpi_CTRL', '5dpi_CTRL'], dtype=float)

intact_mask = endo.obs['condition'] == 'Intact'
for gene in bbb_available:
    intact_expr = endo[intact_mask, gene].X.toarray().flatten()
    frac_mat.loc[gene, 'Intact'] = (intact_expr > 0).mean()

    for cond in ['3dpi_CTRL', '5dpi_CTRL']:
        cond_mask = endo.obs['condition'] == cond
        cond_expr = endo[cond_mask, gene].X.toarray().flatten()
        pseudo = 0.001
        fc_mat.loc[gene, cond] = np.log2((cond_expr.mean() + pseudo) / (intact_expr.mean() + pseudo))
        frac_mat.loc[gene, cond] = (cond_expr > 0).mean()
        try:
            _, pval = stats.mannwhitneyu(cond_expr, intact_expr, alternative='two-sided')
        except:
            pval = 1.0
        pval_mat.loc[gene, cond] = pval

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.2]})

# Panel A: log2FC heatmap
vmax = max(abs(fc_mat.values.astype(float).min()), abs(fc_mat.values.astype(float).max()), 1)
sns.heatmap(fc_mat.astype(float), cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
            annot=True, fmt='.2f', linewidths=0.5, ax=ax1,
            cbar_kws={'label': 'log₂FC vs Intact'})

# Add significance stars
for i, gene in enumerate(fc_mat.index):
    for j, cond in enumerate(fc_mat.columns):
        pval = pval_mat.loc[gene, cond]
        if pval < 0.05:
            marker = '***' if pval < 0.001 else ('**' if pval < 0.01 else '*')
            ax1.text(j + 0.5, i + 0.85, marker, ha='center', va='center',
                     fontsize=8, color='black', fontweight='bold')

ax1.set_title('log₂FC vs Intact\n(Endothelial cells)', fontweight='bold')
ax1.set_ylabel('')

# Panel B: Fraction expressing
sns.heatmap(frac_mat.astype(float), cmap='YlOrRd', vmin=0, vmax=1,
            annot=True, fmt='.2f', linewidths=0.5, ax=ax2,
            cbar_kws={'label': 'Fraction expressing'})
ax2.set_title('Fraction of Cells Expressing\n(Endothelial cells)', fontweight='bold')
ax2.set_ylabel('')

plt.suptitle('GSE226211: BBB Transport & Integrity Genes in Endothelial Cells\n(Cortical Stab Wound)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig6_bbb_endothelial_heatmap.png', bbox_inches='tight')
plt.close()
print("  Saved: fig6_bbb_endothelial_heatmap.png")

# =============================================================================
# 7. Print summary
# =============================================================================
print("\n" + "=" * 70)
print("ENDOTHELIAL Mfsd2a SUMMARY")
print("=" * 70)

for cond in ['Intact', '3dpi_CTRL', '5dpi_CTRL']:
    mask = endo.obs['condition'] == cond
    if mask.sum() == 0:
        continue
    expr = endo[mask, 'Mfsd2a'].X.toarray().flatten()
    print(f"\n  {cond}:")
    print(f"    N cells: {mask.sum()}")
    print(f"    Mean expr: {expr.mean():.4f}")
    print(f"    Fraction expressing: {(expr > 0).mean():.1%}")
    if cond != 'Intact':
        intact_expr = endo[endo.obs['condition'] == 'Intact', 'Mfsd2a'].X.toarray().flatten()
        pseudo = 0.001
        fc = np.log2((expr.mean() + pseudo) / (intact_expr.mean() + pseudo))
        _, pval = stats.mannwhitneyu(expr, intact_expr, alternative='two-sided')
        print(f"    log2FC vs Intact: {fc:+.3f}")
        print(f"    P-value: {pval:.2e}")
        print(f"    Significance: {'***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))}")

# Also check Microglia for comparison
print("\n  --- Microglia comparison ---")
mg_mask = adata.obs['cell_type'] == 'Microglia'
for cond in ['Intact', '3dpi_CTRL', '5dpi_CTRL']:
    mask = mg_mask & (adata.obs['condition'] == cond)
    if mask.sum() == 0:
        continue
    expr = adata[mask, 'Mfsd2a'].X.toarray().flatten()
    print(f"  MG {cond}: mean={expr.mean():.4f}, frac={((expr>0).mean()):.1%}")

print(f"\n{'='*70}")
print("Key finding: Compare Endothelial vs Microglia Mfsd2a changes after injury")
print(f"{'='*70}")
