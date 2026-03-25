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

DATA_DIR = Path("/Users/maxue/Documents/vscode/tbi/data/GSE226211")
RESULTS_DIR = Path("/Users/maxue/Documents/vscode/tbi/results/GSE226211")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans'
})

# Sample metadata — INH samples excluded (Intact + CTRL only)
SAMPLE_META = {
    'GSM7068147_MUC13721': {'condition': 'Intact',    'timepoint': 'Intact'},
    'GSM7068148_MUC13722': {'condition': '3dpi_CTRL', 'timepoint': '3dpi'},
    'GSM7068149_MUC13723': {'condition': '3dpi_CTRL', 'timepoint': '3dpi'},
    'GSM7068152_MUC13726': {'condition': 'Intact',    'timepoint': 'Intact'},
    'GSM7068153_MUC13727': {'condition': '5dpi_CTRL', 'timepoint': '5dpi'},
    'GSM7068156_MUC13731': {'condition': 'Intact',    'timepoint': 'Intact'},
    'GSM7068157_MUC13732': {'condition': 'Intact',    'timepoint': 'Intact'},
    'GSM7068158_MUC18415': {'condition': '5dpi_CTRL', 'timepoint': '5dpi'},
    'GSM7068159_MUC29190': {'condition': 'Intact',    'timepoint': 'Intact'},
    'GSM7068160_21L008532': {'condition': '5dpi_CTRL', 'timepoint': '5dpi'},
}

CTRL_SAMPLES = SAMPLE_META  # all entries are Intact/CTRL, no INH

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
sc.tl.pca(adata_hvg, n_comps=50, random_state=0)
sc.pp.neighbors(adata_hvg, n_pcs=30, random_state=0)
sc.tl.umap(adata_hvg, random_state=0)
sc.tl.leiden(adata_hvg, resolution=0.8, random_state=0)

# Transfer embeddings back
adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
adata.obs['leiden'] = adata_hvg.obs['leiden']

print(f"Clusters: {adata.obs['leiden'].nunique()}")

# =============================================================================
# 3. Cell type annotation — relative scoring (ref: liver snRNA-seq pipeline)
# =============================================================================
print("\nIdentifying cell types...")

MARKERS = {
    'Microglia':      ['Tmem119', 'P2ry12', 'Cx3cr1', 'Hexb', 'Csf1r', 'Aif1'],
    'Astrocyte':      ['Gfap', 'Aqp4', 'Aldh1l1', 'S100b', 'Slc1a3'],
    'Oligodendrocyte':['Mbp', 'Plp1', 'Mog', 'Cnp'],
    'OPC':            ['Pdgfra', 'Cspg4', 'Sox10'],
    'Neuron':         ['Rbfox3', 'Snap25', 'Syt1', 'Stmn2'],
    'Endothelial':    ['Pecam1', 'Cldn5', 'Flt1', 'Tie1', 'Kdr'],
    'Macrophage':     ['Ccr2', 'Lyz2', 'Ms4a7'],
    'Pericyte':       ['Pdgfrb', 'Rgs5', 'Kcnj8'],
    'T_cell':         ['Cd3e', 'Cd3d', 'Cd4', 'Cd8a', 'Trac'],
    'B_cell':         ['Cd79a', 'Ms4a1', 'Pax5', 'Cd19'],
}

for ct, genes in MARKERS.items():
    avail = [g for g in genes if g in adata.var_names]
    if avail:
        sc.tl.score_genes(adata, avail, score_name=f'{ct}_score')
        print(f"  {ct}: {len(avail)}/{len(genes)} markers found")

score_cols = [f'{ct}_score' for ct in MARKERS if f'{ct}_score' in adata.obs.columns]
ct_names   = [s.replace('_score', '') for s in score_cols]

# Per-cluster mean scores
cluster_mean = adata.obs.groupby('leiden')[score_cols].mean()
cluster_mean.columns = ct_names

# Relative scoring: subtract per-cell-type global mean to prevent dominant
# cell types (Astrocyte / Neuron) from suppressing others — same approach as
# liver snRNA-seq pipeline (02_cell_type_annotation.py)
ct_global_mean  = cluster_mean.mean(axis=0)
cluster_relative = cluster_mean - ct_global_mean

# Unknown criteria: best relative score < 0.10 OR margin to 2nd < 0.05
UNKNOWN_MIN_SCORE  = 0.10
UNKNOWN_MIN_MARGIN = 0.05
print("\n--- Cluster annotation (relative scoring) ---")
cluster_ct = {}
for cl in sorted(cluster_relative.index, key=int):
    row      = cluster_relative.loc[cl]
    best_ct  = row.idxmax()
    best_rel = row.max()
    second   = row.drop(best_ct).max()
    margin   = best_rel - second
    if best_rel < UNKNOWN_MIN_SCORE or margin < UNKNOWN_MIN_MARGIN:
        cluster_ct[cl] = 'Unknown'
    else:
        cluster_ct[cl] = best_ct
    print(f"  Cluster {cl}: {cluster_ct[cl]:16s} (rel={best_rel:.3f}, margin={margin:.3f})")

adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_ct)
print("\nCell type assignment:")
print(adata.obs['cell_type'].value_counts())

# =============================================================================
# 3b. All-cells UMAP: cell type + Plin2 expression
# =============================================================================
print("\nGenerating all-cells UMAP figures...")

# ── colour palette (stable across runs) ──────────────────────────────────────
CT_PALETTE = {
    'Microglia':      '#E53935',
    'Macrophage':     '#F4511E',
    'Astrocyte':      '#43A047',
    'Oligodendrocyte':'#1E88E5',
    'OPC':            '#039BE5',
    'Neuron':         '#8E24AA',
    'Endothelial':    '#FFB300',
    'Pericyte':       '#6D4C41',
    'T_cell':         '#00ACC1',
    'B_cell':         '#26A69A',
    'Unknown':        '#BDBDBD',
}
cell_types_present = list(adata.obs['cell_type'].unique())
palette_use = {ct: CT_PALETTE.get(ct, '#BDBDBD') for ct in cell_types_present}

umap_xy = adata.obsm['X_umap']

fig_all, axes_all = plt.subplots(1, 2, figsize=(18, 7))
fig_all.suptitle('GSE226211: All cells — Cortical Stab Wound (CTRL only)',
                 fontsize=13, fontweight='bold')

# Panel A: cell type
ax = axes_all[0]
for ct in sorted(cell_types_present, key=lambda x: (x == 'Unknown', x)):
    mask = adata.obs['cell_type'] == ct
    ax.scatter(umap_xy[mask, 0], umap_xy[mask, 1],
               c=palette_use[ct], s=2, alpha=0.5, label=f'{ct} ({mask.sum():,})',
               rasterized=True)
ax.set_title('Cell Type', fontsize=12, fontweight='bold')
ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
ax.legend(fontsize=7, markerscale=3, ncol=2,
          loc='lower right', framealpha=0.7)
ax.set_aspect('equal', adjustable='datalim')

# Panel B: Plin2 expression
ax2 = axes_all[1]
if 'Plin2' in adata.var_names:
    plin2_all = np.asarray(adata[:, 'Plin2'].X.todense()).flatten()
    sc2 = ax2.scatter(umap_xy[:, 0], umap_xy[:, 1],
                      c=plin2_all, cmap='YlOrRd', s=2, alpha=0.6,
                      vmin=0, vmax=np.percentile(plin2_all, 99), rasterized=True)
    plt.colorbar(sc2, ax=ax2, shrink=0.7, label='Plin2 (log-norm)')
    ax2.set_title('Plin2 Expression', fontsize=12, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Plin2 not found', ha='center', transform=ax2.transAxes)
ax2.set_xlabel('UMAP1'); ax2.set_ylabel('UMAP2')
ax2.set_aspect('equal', adjustable='datalim')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig0_all_cells_umap.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: fig0_all_cells_umap.png")

# Score heatmap: clusters × cell types (diagnostic)
fig_h, ax_h = plt.subplots(figsize=(14, 7))
sns.heatmap(cluster_relative.T, cmap='RdBu_r', center=0, ax=ax_h,
            linewidths=0.3, annot=True, fmt='.2f', annot_kws={'fontsize': 6})
ax_h.set_title('Relative cell type scores per Leiden cluster\n(raw mean − global mean)')
ax_h.set_xlabel('Leiden cluster')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig0b_celltype_score_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: fig0b_celltype_score_heatmap.png")

# =============================================================================
# 3c. Dotplot: marker genes by cell type (ref: liver dotplot_markers_by_cluster)
# =============================================================================
print("\nGenerating dotplot of marker genes by cell type...")

DOTPLOT_MARKERS = {
    'Microglia':      ['Tmem119', 'P2ry12', 'Cx3cr1', 'Hexb'],
    'Macrophage':     ['Ccr2', 'Lyz2', 'Ms4a7'],
    'Astrocyte':      ['Gfap', 'Aqp4', 'Aldh1l1', 'S100b'],
    'Oligodendrocyte':['Mbp', 'Plp1', 'Mog', 'Cnp'],
    'OPC':            ['Pdgfra', 'Cspg4', 'Sox10'],
    'Neuron':         ['Rbfox3', 'Snap25', 'Syt1', 'Stmn2'],
    'Endothelial':    ['Pecam1', 'Cldn5', 'Flt1'],
    'Pericyte':       ['Pdgfrb', 'Rgs5', 'Kcnj8'],
    'T_cell':         ['Cd3e', 'Cd3d', 'Cd8a'],
    'B_cell':         ['Cd79a', 'Ms4a1', 'Pax5'],
}

# Ordered cell types for y-axis (Unknown last)
CT_ORDER = [ct for ct in DOTPLOT_MARKERS if ct in adata.obs['cell_type'].values]
if 'Unknown' in adata.obs['cell_type'].values:
    CT_ORDER.append('Unknown')

flat_markers = [g for genes in DOTPLOT_MARKERS.values()
                for g in genes if g in adata.var_names]

sc.pl.dotplot(adata, flat_markers, groupby='cell_type', categories_order=CT_ORDER,
              use_raw=True, standard_scale='var', show=False,
              title='Marker gene expression by cell type (GSE226211, cortical stab wound)',
              figsize=(max(14, len(flat_markers) * 0.55), len(CT_ORDER) * 0.65 + 1.5))
plt.savefig(RESULTS_DIR / 'fig0c_dotplot_markers_by_celltype.png',
            bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: fig0c_dotplot_markers_by_celltype.png")

# =============================================================================
# 3d. Cell type proportions by condition (ref: liver cell_type_proportions.png)
# =============================================================================
print("\nGenerating cell type proportion figures...")

prop = (adata.obs
        .groupby(['condition', 'cell_type'])
        .size()
        .reset_index(name='n'))
total = (adata.obs.groupby('condition').size().reset_index(name='total'))
prop  = prop.merge(total, on='condition')
prop['proportion'] = prop['n'] / prop['total']
prop.to_csv(RESULTS_DIR / 'cell_type_proportions.csv', index=False)

COND_ORDER = [c for c in ['Intact', '3dpi_CTRL', '5dpi_CTRL']
              if c in prop['condition'].values]

# ── Bar chart (side-by-side, like liver script) ───────────────────────────────
pivot = (prop.pivot(index='cell_type', columns='condition', values='proportion')
         .reindex(columns=COND_ORDER)
         .fillna(0))
# Sort rows by total abundance
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

fig_p, ax_p = plt.subplots(figsize=(13, 5))
pivot.plot(kind='bar', ax=ax_p, width=0.75,
           color=['#4CAF50', '#FF9800', '#E53935'][:len(COND_ORDER)])
ax_p.set_title('Cell type proportions: Intact vs 3dpi vs 5dpi (GSE226211)',
               fontsize=12, fontweight='bold')
ax_p.set_ylabel('Proportion of all cells')
ax_p.set_xlabel('')
ax_p.tick_params(axis='x', rotation=35)
plt.setp(ax_p.get_xticklabels(), ha='right')
ax_p.legend(title='Condition', fontsize=9)
ax_p.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig0d_cell_type_proportions.png',
            bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: fig0d_cell_type_proportions.png")

# ── Stacked bar (100 %) ───────────────────────────────────────────────────────
pivot_cond = (prop.pivot(index='condition', columns='cell_type', values='proportion')
              .reindex(index=COND_ORDER)
              .fillna(0))
# Sort columns by overall abundance
col_order_stack = pivot_cond.sum(axis=0).sort_values(ascending=False).index
pivot_cond = pivot_cond[col_order_stack]

fig_s, ax_s = plt.subplots(figsize=(7, 5))
colors_stack = [CT_PALETTE.get(ct, '#BDBDBD') for ct in col_order_stack]
pivot_cond.plot(kind='bar', stacked=True, ax=ax_s,
                color=colors_stack, edgecolor='white', linewidth=0.3, width=0.6)
ax_s.set_title('Cell type composition (stacked 100%)', fontsize=12, fontweight='bold')
ax_s.set_ylabel('Proportion')
ax_s.set_xlabel('')
ax_s.tick_params(axis='x', rotation=0)
ax_s.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left', title='Cell type')
ax_s.set_ylim(0, 1)
ax_s.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig0e_cell_type_proportions_stacked.png',
            bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: fig0e_cell_type_proportions_stacked.png")

print("\nCell type proportions summary:")
print(prop[prop['condition'] == 'Intact'].sort_values('proportion', ascending=False)
      [['cell_type', 'n', 'proportion']].to_string(index=False))

# =============================================================================
# 4. Focus on Microglia — LDAM scoring
# =============================================================================
print("\nFocusing on Microglia...")
mg = adata[adata.obs['cell_type'].isin(['Microglia', 'Macrophage'])].copy()
print(f"Myeloid cells (Microglia + Macrophage): {mg.shape[0]}")

# Sub-cluster microglia — resolution=0.3, random_state=0 (aligned with script 03)
sc.pp.neighbors(mg, n_pcs=20, random_state=0,
                use_rep='X_pca' if 'X_pca' in mg.obsm else None)
sc.tl.leiden(mg, resolution=0.3, key_added='mg_subcluster', random_state=0)
sc.tl.umap(mg, random_state=0)
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

# Save myeloid object for downstream scripts (03_plin2_by_subcluster.py)
mg_h5ad = RESULTS_DIR / 'mg_myeloid.h5ad'
mg.write_h5ad(mg_h5ad)
print(f"Saved myeloid h5ad: {mg_h5ad}")

print(f"\nAll results saved to: {RESULTS_DIR}")
