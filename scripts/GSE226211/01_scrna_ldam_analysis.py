#!/usr/bin/env python3
"""
GSE226211/GSE226207 Analysis: scRNA-seq LDAM scoring in microglia subtypes
- Cortical stab wound model (identical to user's model)
- Intact / 3dpi / 5dpi (CTRL and INH conditions)
- Focus: LDAM gene signature in reactive microglia clusters
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import mmread
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/volume/data/xyma/jsonl/training/tbi/data/GSE226211")
RESULTS_DIR = Path("/volume/data/xyma/jsonl/training/tbi/results/GSE226211")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans'
})

# Sample metadata (from GEO GSE226207)
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

# Focus on Intact and CTRL conditions (no inhibitor)
CTRL_SAMPLES = {k: v for k, v in SAMPLE_META.items()
                if v['condition'] in ['Intact', '3dpi_CTRL', '5dpi_CTRL']}

# =============================================================================
# 1. Load and merge CTRL samples
# =============================================================================
print("Loading scRNA-seq samples (Intact + CTRL only)...")
adatas = []
for sample_id, meta in CTRL_SAMPLES.items():
    prefix = DATA_DIR / sample_id
    barcodes = pd.read_csv(f"{prefix}_barcodes.tsv.gz", header=None, sep='\t')[0].values
    features = pd.read_csv(f"{prefix}_features.tsv.gz", header=None, sep='\t')
    mtx = mmread(f"{prefix}_matrix.mtx.gz")

    gene_names = features[1].values
    # genes x cells -> cells x genes
    adata = sc.AnnData(X=mtx.T.tocsr(),
                       obs=pd.DataFrame(index=[f"{sample_id}_{b}" for b in barcodes]),
                       var=pd.DataFrame(index=gene_names))
    adata.var_names_make_unique()
    adata.obs['sample'] = sample_id
    adata.obs['condition'] = meta['condition']
    adata.obs['timepoint'] = meta['timepoint']
    adatas.append(adata)
    print(f"  {sample_id}: {adata.shape[0]} cells, {meta['condition']}")

# Merge
print("Merging...")
adata = sc.concat(adatas, join='inner')
adata.var_names_make_unique()
print(f"Total: {adata.shape[0]} cells × {adata.shape[1]} genes")

# =============================================================================
# 2. Basic preprocessing and clustering
# =============================================================================
print("\nPreprocessing...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)

# Store raw counts
adata.layers['counts'] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Store normalized for later
adata.raw = adata.copy()

# HVG selection and scaling
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata_hvg = adata[:, adata.var['highly_variable']].copy()
sc.pp.scale(adata_hvg, max_value=10)

# PCA + neighbors + clustering
sc.tl.pca(adata_hvg, n_comps=50)
sc.pp.neighbors(adata_hvg, n_pcs=30)
sc.tl.umap(adata_hvg)
sc.tl.leiden(adata_hvg, resolution=0.8)

# Transfer embeddings back
adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
adata.obs['leiden'] = adata_hvg.obs['leiden']

print(f"Clusters: {adata.obs['leiden'].nunique()}")

# =============================================================================
# 3. Identify microglia clusters using markers
# =============================================================================
print("\nIdentifying cell types...")

# Key markers
markers = {
    'Microglia': ['Tmem119', 'P2ry12', 'Cx3cr1', 'Hexb', 'Csf1r', 'Aif1'],
    'Astrocyte': ['Gfap', 'Aqp4', 'Aldh1l1', 'S100b', 'Slc1a3'],
    'Oligodendrocyte': ['Mbp', 'Plp1', 'Mog', 'Cnp'],
    'OPC': ['Pdgfra', 'Cspg4', 'Sox10'],
    'Neuron': ['Rbfox3', 'Snap25', 'Syt1', 'Stmn2'],
    'Endothelial': ['Pecam1', 'Cldn5', 'Flt1'],
    'Macrophage': ['Ccr2', 'Lyz2', 'Ms4a7'],
}

# Score each cluster for marker expression
cluster_scores = {}
for ct, genes in markers.items():
    avail = [g for g in genes if g in adata.var_names]
    if avail:
        sc.tl.score_genes(adata, avail, score_name=f'{ct}_score')

# Assign cell types based on highest marker score
marker_scores = ['Microglia_score', 'Astrocyte_score', 'Oligodendrocyte_score',
                 'OPC_score', 'Neuron_score', 'Endothelial_score', 'Macrophage_score']
available_scores = [s for s in marker_scores if s in adata.obs.columns]

# Per-cluster mean scores
cluster_ct = {}
for cl in adata.obs['leiden'].unique():
    mask = adata.obs['leiden'] == cl
    scores = {s.replace('_score', ''): adata.obs.loc[mask, s].mean() for s in available_scores}
    best_ct = max(scores, key=scores.get)
    cluster_ct[cl] = best_ct

adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_ct)
print("\nCell type assignment:")
print(adata.obs['cell_type'].value_counts())

# =============================================================================
# 4. Focus on Microglia — LDAM scoring
# =============================================================================
print("\nFocusing on Microglia...")
mg = adata[adata.obs['cell_type'] == 'Microglia'].copy()
print(f"Microglia cells: {mg.shape[0]}")

if mg.shape[0] < 50:
    # Also include Macrophage as they may contain reactive microglia
    mg = adata[adata.obs['cell_type'].isin(['Microglia', 'Macrophage'])].copy()
    print(f"Including Macrophages: {mg.shape[0]} total myeloid cells")

# Sub-cluster microglia
sc.pp.neighbors(mg, n_pcs=20, use_rep='X_pca' if 'X_pca' in mg.obsm else None)
sc.tl.leiden(mg, resolution=0.6, key_added='mg_subcluster')
sc.tl.umap(mg)
print(f"Microglia subclusters: {mg.obs['mg_subcluster'].nunique()}")

# Gene sets
LDAM_GENES = ['Lpl', 'Apoe', 'Cd36', 'Fabp5', 'Fabp4', 'Plin2', 'Plin3',
              'Lipa', 'Soat1', 'Mgll']
DAM_GENES = ['Trem2', 'Tyrobp', 'Cst7', 'Spp1', 'Itgax', 'Axl', 'Lgals3',
             'Clec7a', 'Gpnmb', 'Igf1']
HOMEOSTATIC = ['Tmem119', 'P2ry12', 'Cx3cr1', 'Hexb', 'Csf1r', 'Siglech']
IL6_GENES = ['Il6', 'Il6ra', 'Il6st', 'Stat3', 'Socs3']
DHA_GENES = ['Mfsd2a', 'Elovl2', 'Fads1', 'Fads2']
PHAGO_GENES = ['Cd68', 'Mertk', 'Axl', 'Tyrobp', 'Trem2']
INFLAMMATION = ['Il1b', 'Tnf', 'Ccl2', 'Cxcl10', 'Ptgs2', 'Nos2']

for name, genes in [('LDAM', LDAM_GENES), ('DAM', DAM_GENES),
                     ('Homeostatic', HOMEOSTATIC), ('IL6', IL6_GENES),
                     ('Inflammation', INFLAMMATION), ('Phagocytosis', PHAGO_GENES)]:
    avail = [g for g in genes if g in mg.var_names]
    if len(avail) >= 2:
        sc.tl.score_genes(mg, avail, score_name=f'{name}_score')
    print(f"  {name}: {len(avail)}/{len(genes)} genes")

# =============================================================================
# 5. Figure 1: UMAP of microglia colored by condition + scores
# =============================================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 4, figsize=(24, 11))

# UMAP by condition
sc.pl.umap(mg, color='condition', ax=axes[0, 0], show=False, title='Condition',
           palette={'Intact': '#4CAF50', '3dpi_CTRL': '#FF9800', '5dpi_CTRL': '#E53935'})

sc.pl.umap(mg, color='mg_subcluster', ax=axes[0, 1], show=False, title='Subcluster', legend_loc='on data')

# Score UMAPs
for idx, (score, title) in enumerate([
    ('LDAM_score', 'LDAM Score'), ('DAM_score', 'DAM Score'),
    ('Homeostatic_score', 'Homeostatic Score'), ('IL6_score', 'IL-6 Score'),
    ('Inflammation_score', 'Inflammation Score'), ('Phagocytosis_score', 'Phagocytosis Score'),
]):
    if score in mg.obs.columns:
        row, col = divmod(idx + 2, 4)
        sc.pl.umap(mg, color=score, ax=axes[row, col], show=False, title=title,
                   cmap='RdYlBu_r', vmin=-0.3, vmax=0.5)

plt.suptitle('GSE226211: Microglia Subclusters — Cortical Stab Wound\n(Intact / 3dpi / 5dpi CTRL)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig1_mg_umap_scores.png', bbox_inches='tight')
plt.close()
print("  Saved: fig1_mg_umap_scores.png")

# =============================================================================
# 6. Figure 2: Score comparison across conditions
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

score_list = ['LDAM_score', 'DAM_score', 'Homeostatic_score',
              'IL6_score', 'Inflammation_score', 'Phagocytosis_score']
titles = ['LDAM', 'DAM', 'Homeostatic', 'IL-6 Pathway', 'Inflammation', 'Phagocytosis']

for idx, (score, title) in enumerate(zip(score_list, titles)):
    ax = axes.flatten()[idx]
    if score not in mg.obs.columns:
        ax.set_visible(False)
        continue
    order = ['Intact', '3dpi_CTRL', '5dpi_CTRL']
    available_order = [o for o in order if o in mg.obs['condition'].values]
    sns.violinplot(data=mg.obs, x='condition', y=score, order=available_order,
                   palette=['#4CAF50', '#FF9800', '#E53935'], ax=ax, cut=0, inner='box')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('')

plt.suptitle('GSE226211: Gene Set Scores in Microglia by Condition\n(Cortical Stab Wound, scRNA-seq)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig2_scores_by_condition.png', bbox_inches='tight')
plt.close()
print("  Saved: fig2_scores_by_condition.png")

# =============================================================================
# 7. Figure 3: Key individual gene expression
# =============================================================================
key_genes_plot = ['Plin2', 'Lpl', 'Cd36', 'Apoe', 'Fabp5', 'Mgll',
                  'Trem2', 'Spp1', 'Cst7', 'Grn',
                  'Il6', 'Il6ra', 'Il6st', 'Mfsd2a',
                  'Il1b', 'Tnf']
key_genes_plot = [g for g in key_genes_plot if g in mg.var_names]

ncols = 4
nrows = (len(key_genes_plot) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3))
axes = axes.flatten()

for idx, gene in enumerate(key_genes_plot):
    ax = axes[idx]
    expr = mg[:, gene].X.toarray().flatten() if hasattr(mg[:, gene].X, 'toarray') else mg[:, gene].X.flatten()
    plot_df = pd.DataFrame({'Condition': mg.obs['condition'].values, 'Expression': expr})

    order = ['Intact', '3dpi_CTRL', '5dpi_CTRL']
    available_order = [o for o in order if o in plot_df['Condition'].values]

    sns.violinplot(data=plot_df, x='Condition', y='Expression', order=available_order,
                   palette=['#4CAF50', '#FF9800', '#E53935'], ax=ax, cut=0, inner='box')

    # Add fraction expressing
    for oi, o in enumerate(available_order):
        vals = plot_df[plot_df['Condition'] == o]['Expression']
        frac = (vals > 0).mean()
        ax.text(oi, ax.get_ylim()[1] * 0.9, f'{frac:.0%}', ha='center', fontsize=8, color='blue')

    ax.set_title(gene, fontweight='bold', fontsize=11)
    ax.set_xlabel('')

for idx in range(len(key_genes_plot), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('GSE226211: Key Gene Expression in Microglia\n(Cortical Stab Wound, Intact vs 3dpi vs 5dpi)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig3_key_genes_violin.png', bbox_inches='tight')
plt.close()
print("  Saved: fig3_key_genes_violin.png")

# =============================================================================
# 8. Figure 4: Subcluster characterization
# =============================================================================
print("\nCharacterizing microglia subclusters...")

# Per-subcluster composition and scores
subcluster_stats = []
for cl in sorted(mg.obs['mg_subcluster'].unique(), key=int):
    mask = mg.obs['mg_subcluster'] == cl
    n_cells = mask.sum()
    comp = mg.obs.loc[mask, 'condition'].value_counts(normalize=True)
    stats = {'cluster': cl, 'n_cells': int(n_cells)}
    for cond in ['Intact', '3dpi_CTRL', '5dpi_CTRL']:
        stats[f'frac_{cond}'] = comp.get(cond, 0)
    for score in score_list:
        if score in mg.obs.columns:
            stats[score] = mg.obs.loc[mask, score].mean()
    subcluster_stats.append(stats)

sc_df = pd.DataFrame(subcluster_stats)
sc_df.to_csv(RESULTS_DIR / 'mg_subcluster_stats.csv', index=False)
print(sc_df.to_string(index=False))

# Heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2]})

# Composition
comp_cols = [c for c in sc_df.columns if c.startswith('frac_')]
comp_data = sc_df.set_index('cluster')[comp_cols]
comp_data.columns = [c.replace('frac_', '') for c in comp_cols]
comp_data.plot(kind='barh', stacked=True, ax=ax1,
               color=['#4CAF50', '#FF9800', '#E53935'])
ax1.set_xlabel('Fraction')
ax1.set_ylabel('Subcluster')
ax1.set_title('Condition Composition')
ax1.legend(fontsize=8)

# Score heatmap
score_cols = [c for c in sc_df.columns if c.endswith('_score')]
score_data = sc_df.set_index('cluster')[score_cols]
score_data.columns = [c.replace('_score', '') for c in score_cols]
sns.heatmap(score_data, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
            ax=ax2, linewidths=0.5, cbar_kws={'label': 'Mean Score'})
ax2.set_title('Gene Set Scores per Subcluster')

plt.suptitle('GSE226211: Microglia Subcluster Characterization', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig4_subcluster_characterization.png', bbox_inches='tight')
plt.close()
print("  Saved: fig4_subcluster_characterization.png")

# =============================================================================
# 9. Pseudo-bulk DE: 3dpi vs Intact, 5dpi vs Intact
# =============================================================================
print("\nPseudo-bulk DE in microglia...")

all_key_genes = list(set(LDAM_GENES + DAM_GENES + IL6_GENES + DHA_GENES +
                         INFLAMMATION + ['Grn', 'Hdac3', 'Mfsd2a', 'Fasn', 'Scd1', 'Scd2',
                                          'Cpt1a', 'Hmgcr', 'Abca1', 'Nr1h3', 'Lcn2', 'Serpina3n']))

intact = mg[mg.obs['condition'] == 'Intact']
dpi3 = mg[mg.obs['condition'] == '3dpi_CTRL']
dpi5 = mg[mg.obs['condition'] == '5dpi_CTRL']

print(f"\n{'Gene':12s} {'3dpi log2FC':>12s} {'5dpi log2FC':>12s}")
print("-" * 40)
de_rows = []
for gene in sorted(all_key_genes):
    if gene not in mg.var_names:
        continue
    intact_expr = np.array(intact[:, gene].X.mean(axis=0)).flatten()[0]
    dpi3_expr = np.array(dpi3[:, gene].X.mean(axis=0)).flatten()[0] if dpi3.shape[0] > 0 else 0
    dpi5_expr = np.array(dpi5[:, gene].X.mean(axis=0)).flatten()[0] if dpi5.shape[0] > 0 else 0

    pseudo = 0.01
    fc3 = np.log2((dpi3_expr + pseudo) / (intact_expr + pseudo))
    fc5 = np.log2((dpi5_expr + pseudo) / (intact_expr + pseudo))

    if abs(fc3) > 0.3 or abs(fc5) > 0.3:
        print(f"  {gene:12s} {fc3:+.2f}         {fc5:+.2f}")

    de_rows.append({'gene': gene, '3dpi_log2FC': fc3, '5dpi_log2FC': fc5,
                    'intact_mean': intact_expr, '3dpi_mean': dpi3_expr, '5dpi_mean': dpi5_expr})

de_df = pd.DataFrame(de_rows)
de_df.to_csv(RESULTS_DIR / 'mg_pseudobulk_DE.csv', index=False)

print(f"\nAll results saved to: {RESULTS_DIR}")
