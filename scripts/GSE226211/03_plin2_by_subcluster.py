#!/usr/bin/env python3
"""
GSE226211: Plin2 expression across microglia subclusters
- CTRL-only samples (Intact + 3dpi_CTRL + 5dpi_CTRL), INH excluded
- Reduced resolution (0.3) for fewer, more interpretable subclusters
- Per-subcluster Plin2 quantification + top marker genes for Plin2-high clusters
- Outputs: figure + CSV + marker gene table for 07_plin2_microglia_comparison.md
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.io import mmread
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR   = Path("/Users/maxue/Documents/vscode/tbi/data/GSE226211")
RESULTS_DIR = Path("/Users/maxue/Documents/vscode/tbi/results/GSE226211")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 150,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans'
})

# ── Sample metadata (INH samples removed entirely) ────────────────────────────
# GSM7068150/151 (3dpi_INH) and GSM7068154/155/161 (5dpi_INH) excluded
SAMPLE_META = {
    'GSM7068147_MUC13721':   {'condition': 'Intact',    'timepoint': 'Intact'},
    'GSM7068148_MUC13722':   {'condition': '3dpi_CTRL', 'timepoint': '3dpi'},
    'GSM7068149_MUC13723':   {'condition': '3dpi_CTRL', 'timepoint': '3dpi'},
    'GSM7068152_MUC13726':   {'condition': 'Intact',    'timepoint': 'Intact'},
    'GSM7068153_MUC13727':   {'condition': '5dpi_CTRL', 'timepoint': '5dpi'},
    'GSM7068156_MUC13731':   {'condition': 'Intact',    'timepoint': 'Intact'},
    'GSM7068157_MUC13732':   {'condition': 'Intact',    'timepoint': 'Intact'},
    'GSM7068158_MUC18415':   {'condition': '5dpi_CTRL', 'timepoint': '5dpi'},
    'GSM7068159_MUC29190':   {'condition': 'Intact',    'timepoint': 'Intact'},
    'GSM7068160_21L008532':  {'condition': '5dpi_CTRL', 'timepoint': '5dpi'},
}

# ── 1. Load myeloid object (saved by 01_scrna_ldam_analysis.py) ───────────────
MG_H5AD = RESULTS_DIR / 'mg_myeloid.h5ad'

if MG_H5AD.exists():
    print(f"Loading pre-computed myeloid object from {MG_H5AD} ...")
    mg = sc.read_h5ad(MG_H5AD)
    print(f"Myeloid cells: {mg.shape[0]}  subclusters: {mg.obs['mg_subcluster'].nunique()}")
else:
    # Fallback: rebuild from scratch (run 01_scrna_ldam_analysis.py first for alignment)
    print("WARNING: mg_myeloid.h5ad not found — rebuilding from raw data (results may differ from script 01)")
    adatas = []
    for sid, meta in SAMPLE_META.items():
        barcodes = pd.read_csv(DATA_DIR / f"{sid}_barcodes.tsv.gz",
                                header=None, sep='\t')[0].values
        features = pd.read_csv(DATA_DIR / f"{sid}_features.tsv.gz",
                                header=None, sep='\t')
        mtx = mmread(DATA_DIR / f"{sid}_matrix.mtx.gz")
        adata = sc.AnnData(
            X=mtx.T.tocsr(),
            obs=pd.DataFrame(index=[f"{sid}_{b}" for b in barcodes]),
            var=pd.DataFrame(index=features[1].values)
        )
        adata.var_names_make_unique()
        adata.obs['sample']    = sid
        adata.obs['condition'] = meta['condition']
        adata.obs['timepoint'] = meta['timepoint']
        adatas.append(adata)
        print(f"  {sid}: {adata.shape[0]} cells  [{meta['condition']}]")

    adata = sc.concat(adatas, join='inner')
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata_hvg = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50, random_state=0)
    sc.pp.neighbors(adata_hvg, n_pcs=30, random_state=0)
    sc.tl.umap(adata_hvg, random_state=0)
    sc.tl.leiden(adata_hvg, resolution=0.8, random_state=0)
    adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
    adata.obs['leiden']  = adata_hvg.obs['leiden']

    MARKER_SETS = {
        'Microglia':      ['Tmem119', 'P2ry12', 'Cx3cr1', 'Hexb', 'Csf1r', 'Aif1'],
        'Astrocyte':      ['Gfap', 'Aqp4', 'Aldh1l1', 'S100b', 'Slc1a3'],
        'Oligodendrocyte':['Mbp', 'Plp1', 'Mog', 'Cnp'],
        'OPC':            ['Pdgfra', 'Cspg4', 'Sox10'],
        'Neuron':         ['Rbfox3', 'Snap25', 'Syt1', 'Stmn2'],
        'Endothelial':    ['Pecam1', 'Cldn5', 'Flt1'],
        'Macrophage':     ['Ccr2', 'Lyz2', 'Ms4a7'],
    }
    for ct, genes in MARKER_SETS.items():
        avail = [g for g in genes if g in adata.var_names]
        if avail:
            sc.tl.score_genes(adata, avail, score_name=f'{ct}_score')
    score_cols = [f'{ct}_score' for ct in MARKER_SETS if f'{ct}_score' in adata.obs.columns]
    cluster_ct = {}
    for cl in adata.obs['leiden'].unique():
        mask = adata.obs['leiden'] == cl
        best = max(score_cols, key=lambda s: adata.obs.loc[mask, s].mean())
        cluster_ct[cl] = best.replace('_score', '')
    adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_ct)

    mg = adata[adata.obs['cell_type'].isin(['Microglia', 'Macrophage'])].copy()
    sc.pp.neighbors(mg, n_pcs=20, random_state=0,
                    use_rep='X_pca' if 'X_pca' in mg.obsm else None)
    sc.tl.leiden(mg, resolution=0.3, key_added='mg_subcluster', random_state=0)
    sc.tl.umap(mg, random_state=0)
    print(f"Myeloid cells: {mg.shape[0]}  subclusters: {mg.obs['mg_subcluster'].nunique()}")

n_cl = mg.obs['mg_subcluster'].nunique()

# ── Gene scoring ───────────────────────────────────────────────────────────────
LDAM_GENES = ['Lpl', 'Apoe', 'Cd36', 'Fabp5', 'Fabp4', 'Plin2', 'Plin3', 'Lipa', 'Soat1', 'Mgll']
DAM_GENES  = ['Trem2', 'Tyrobp', 'Cst7', 'Spp1', 'Itgax', 'Axl', 'Lgals3', 'Clec7a', 'Gpnmb', 'Igf1']
HOMEO_GENES = ['Tmem119', 'P2ry12', 'Cx3cr1', 'Hexb', 'Csf1r', 'Siglech']
for name, genes in [('LDAM', LDAM_GENES), ('DAM', DAM_GENES), ('Homeostatic', HOMEO_GENES)]:
    avail = [g for g in genes if g in mg.var_names]
    if len(avail) >= 2:
        sc.tl.score_genes(mg, avail, score_name=f'{name}_score')

# ── 5. Per-subcluster Plin2 statistics ────────────────────────────────────────
print("\nComputing per-subcluster Plin2 statistics...")

if 'Plin2' not in mg.var_names:
    raise ValueError("Plin2 not found in data!")

plin2_expr = np.asarray(mg[:, 'Plin2'].X.todense()).flatten()
mg.obs['Plin2_expr'] = plin2_expr

COND_COLORS = {'Intact': '#4CAF50', '3dpi_CTRL': '#FF9800', '5dpi_CTRL': '#E53935'}

rows = []
for cl in sorted(mg.obs['mg_subcluster'].unique(), key=int):
    mask = mg.obs['mg_subcluster'] == cl
    vals  = mg.obs.loc[mask, 'Plin2_expr']
    comp  = mg.obs.loc[mask, 'condition'].value_counts(normalize=True)
    row = {
        'cluster':       cl,
        'n_cells':       int(mask.sum()),
        'plin2_mean':    float(vals.mean()),
        'plin2_median':  float(vals.median()),
        'plin2_pct_pos': float((vals > 0).mean()),  # fraction expressing
        'frac_Intact':    float(comp.get('Intact',    0)),
        'frac_3dpi':      float(comp.get('3dpi_CTRL', 0)),
        'frac_5dpi':      float(comp.get('5dpi_CTRL', 0)),
    }
    for sc_name in ['LDAM_score', 'DAM_score', 'Homeostatic_score']:
        if sc_name in mg.obs.columns:
            row[sc_name] = float(mg.obs.loc[mask, sc_name].mean())
    rows.append(row)

plin2_df = pd.DataFrame(rows).sort_values('plin2_mean', ascending=False)
plin2_df.to_csv(RESULTS_DIR / 'plin2_by_subcluster.csv', index=False)
print(plin2_df[['cluster','n_cells','plin2_mean','plin2_pct_pos',
                 'LDAM_score','DAM_score','frac_Intact','frac_3dpi','frac_5dpi']].to_string(index=False))

# ── 6. Figure: Plin2 expression by subcluster ─────────────────────────────────
print("\nGenerating Plin2 subcluster figure...")

# Sort clusters by mean Plin2
cl_order = list(plin2_df['cluster'])   # highest → lowest Plin2

plot_df = pd.DataFrame({
    'Subcluster': mg.obs['mg_subcluster'].values,
    'Condition':  mg.obs['condition'].values,
    'Plin2':      mg.obs['Plin2_expr'].values,
})
plot_df['Subcluster'] = pd.Categorical(plot_df['Subcluster'], categories=cl_order)
plot_df = plot_df.sort_values('Subcluster')

fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(3, 2, height_ratios=[2.5, 1.2, 1.2],
                        hspace=0.45, wspace=0.35)

# ── Panel A: Violin plot of Plin2 per subcluster (all cells, colored by condition)
ax_violin = fig.add_subplot(gs[0, :])

# Use native matplotlib violin to colour by condition
positions = range(len(cl_order))
parts = ax_violin.violinplot(
    [plot_df.loc[plot_df['Subcluster'] == cl, 'Plin2'].values for cl in cl_order],
    positions=list(positions),
    showmedians=True, showextrema=False, widths=0.7
)
for pc in parts['bodies']:
    pc.set_facecolor('#A0C4FF')
    pc.set_alpha(0.7)
    pc.set_edgecolor('grey')
parts['cmedians'].set_color('#1a1aff')
parts['cmedians'].set_linewidth(2)

# Overlay: mean dot per condition
for ci, cl in enumerate(cl_order):
    for cond, color in COND_COLORS.items():
        cvals = plot_df.loc[(plot_df['Subcluster'] == cl) &
                            (plot_df['Condition']  == cond), 'Plin2'].values
        if len(cvals) > 0:
            ax_violin.scatter(ci, cvals.mean(), color=color, s=55,
                               zorder=5, edgecolors='white', linewidths=0.5)

# Annotate: % Plin2+ and LDAM score on top of violin
for ci, cl in enumerate(cl_order):
    row = plin2_df[plin2_df['cluster'] == cl].iloc[0]
    ldam = row.get('LDAM_score', np.nan)
    pct  = row['plin2_pct_pos']
    n    = row['n_cells']
    ax_violin.text(ci, ax_violin.get_ylim()[1] * 0.97 if ci == 0 else ax_violin.get_ylim()[1],
                   f"{pct:.0%}\nn={n}", ha='center', va='top', fontsize=8, color='#333333')

ax_violin.set_xticks(list(positions))
ax_violin.set_xticklabels([f"Cluster {cl}" for cl in cl_order], rotation=45, ha='right', fontsize=10)
ax_violin.set_ylabel('Plin2 (log-normalised)', fontsize=12)
ax_violin.set_title('Plin2 Expression Across Microglia Subclusters (GSE226211, Cortical Stab Wound)',
                     fontsize=13, fontweight='bold')
ax_violin.set_xlim(-0.6, len(cl_order) - 0.4)
ax_violin.grid(axis='y', alpha=0.3)

legend_handles = [mpatches.Patch(color=c, label=l) for l, c in COND_COLORS.items()]
ax_violin.legend(handles=legend_handles, title='Condition (mean dot)',
                 loc='upper right', fontsize=9, title_fontsize=9)

# ── Panel B: Bar — mean Plin2 per cluster ─────────────────────────────────────
ax_bar = fig.add_subplot(gs[1, 0])
bar_colors = plt.cm.YlOrRd(
    np.linspace(0.3, 0.9, len(cl_order))
)[::-1]   # darker = higher Plin2
bars = ax_bar.bar(range(len(cl_order)),
                  [plin2_df[plin2_df['cluster'] == cl]['plin2_mean'].values[0] for cl in cl_order],
                  color=bar_colors, edgecolor='grey', linewidth=0.5)
ax_bar.set_xticks(range(len(cl_order)))
ax_bar.set_xticklabels([f"C{cl}" for cl in cl_order], rotation=0, fontsize=9)
ax_bar.set_ylabel('Mean Plin2', fontsize=10)
ax_bar.set_title('Mean Plin2 per Subcluster', fontsize=11, fontweight='bold')
ax_bar.grid(axis='y', alpha=0.3)

# ── Panel C: Scatter — LDAM score vs mean Plin2 ───────────────────────────────
ax_scatter = fig.add_subplot(gs[1, 1])
if 'LDAM_score' in plin2_df.columns:
    scatter_colors = [
        '#E53935' if (plin2_df.loc[plin2_df['cluster'] == cl, 'frac_5dpi'].values[0] +
                      plin2_df.loc[plin2_df['cluster'] == cl, 'frac_3dpi'].values[0]) > 0.5
        else '#4CAF50'
        for cl in plin2_df['cluster']
    ]
    sc_plot = ax_scatter.scatter(
        plin2_df['LDAM_score'], plin2_df['plin2_mean'],
        s=plin2_df['n_cells'] / 20,
        c=scatter_colors, alpha=0.8, edgecolors='grey', linewidths=0.5
    )
    for _, row in plin2_df.iterrows():
        ax_scatter.annotate(f"C{row['cluster']}", (row['LDAM_score'], row['plin2_mean']),
                            textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax_scatter.set_xlabel('LDAM Score', fontsize=10)
    ax_scatter.set_ylabel('Mean Plin2', fontsize=10)
    ax_scatter.set_title('LDAM Score vs Plin2 Expression\n(bubble size = n_cells)',
                         fontsize=10, fontweight='bold')
    ax_scatter.grid(alpha=0.3)
    legend_handles2 = [
        mpatches.Patch(color='#E53935', label='Injury-dominant (>50% 3+5dpi)'),
        mpatches.Patch(color='#4CAF50', label='Intact-dominant')
    ]
    ax_scatter.legend(handles=legend_handles2, fontsize=8, loc='upper left')

# ── Panel D: Stacked bar — condition composition per cluster ───────────────────
ax_comp = fig.add_subplot(gs[2, 0])
comp_data = plin2_df[['cluster','frac_Intact','frac_3dpi','frac_5dpi']].set_index('cluster')
comp_data.index = [f"C{c}" for c in comp_data.index]
comp_data.columns = ['Intact', '3dpi', '5dpi']
comp_data.plot(kind='bar', stacked=True, ax=ax_comp,
               color=['#4CAF50', '#FF9800', '#E53935'], edgecolor='white', linewidth=0.3)
ax_comp.set_ylabel('Fraction', fontsize=10)
ax_comp.set_title('Condition Composition per Subcluster', fontsize=11, fontweight='bold')
ax_comp.tick_params(axis='x', rotation=0)
ax_comp.legend(fontsize=8, loc='upper right')
ax_comp.grid(axis='y', alpha=0.3)

# ── Panel E: % Plin2+ cells per cluster ───────────────────────────────────────
ax_pct = fig.add_subplot(gs[2, 1])
pct_vals = [plin2_df[plin2_df['cluster'] == cl]['plin2_pct_pos'].values[0] * 100
            for cl in cl_order]
pct_colors = plt.cm.YlOrRd(np.array(pct_vals) / 100)
ax_pct.bar(range(len(cl_order)), pct_vals,
           color=pct_colors, edgecolor='grey', linewidth=0.5)
ax_pct.set_xticks(range(len(cl_order)))
ax_pct.set_xticklabels([f"C{cl}" for cl in cl_order], rotation=0, fontsize=9)
ax_pct.set_ylabel('% Plin2+ cells', fontsize=10)
ax_pct.set_title('Fraction of Plin2-Expressing Cells per Subcluster', fontsize=11, fontweight='bold')
ax_pct.axhline(50, color='grey', linestyle='--', alpha=0.5, label='50%')
ax_pct.set_ylim(0, 105)
ax_pct.grid(axis='y', alpha=0.3)

out_path = RESULTS_DIR / 'fig7_plin2_by_subcluster.png'
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# ── Panel UMAP: cluster / Plin2 / condition ────────────────────────────────────
print("\nGenerating UMAP figure...")

umap_xy = mg.obsm['X_umap']
n_cl_final = mg.obs['mg_subcluster'].nunique()

# assign per-cluster palette
cluster_ids   = sorted(mg.obs['mg_subcluster'].unique(), key=int)
cmap_cl       = plt.cm.get_cmap('tab10', len(cluster_ids))
cluster_color = {cl: cmap_cl(i) for i, cl in enumerate(cluster_ids)}

# mark top-2 Plin2 clusters
top2_preview = list(plin2_df['cluster'].iloc[:2])

fig_u, axes_u = plt.subplots(1, 3, figsize=(18, 5.5))
fig_u.suptitle('GSE226211 Microglia UMAP (CTRL only, INH excluded)', fontsize=13, fontweight='bold')

# --- subplot 1: subclusters ---
ax = axes_u[0]
for cl in cluster_ids:
    mask_cl = mg.obs['mg_subcluster'] == cl
    label   = f"C{cl}*" if cl in top2_preview else f"C{cl}"
    ax.scatter(umap_xy[mask_cl, 0], umap_xy[mask_cl, 1],
               c=[cluster_color[cl]], s=4, alpha=0.7, label=label, rasterized=True)
# annotate cluster centroid
for cl in cluster_ids:
    mask_cl = mg.obs['mg_subcluster'] == cl
    cx, cy  = umap_xy[mask_cl, 0].mean(), umap_xy[mask_cl, 1].mean()
    weight  = 'bold' if cl in top2_preview else 'normal'
    color   = '#cc0000' if cl in top2_preview else 'black'
    ax.text(cx, cy, f"C{cl}", ha='center', va='center', fontsize=8,
            fontweight=weight, color=color,
            bbox=dict(facecolor='white', alpha=0.55, edgecolor='none', pad=1))
ax.set_title(f'Subclusters (n={n_cl_final})\n* = top-2 Plin2', fontsize=11)
ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
ax.legend(fontsize=7, ncol=2, markerscale=2, loc='lower right')
ax.set_aspect('equal', adjustable='datalim')

# --- subplot 2: Plin2 expression ---
ax2 = axes_u[1]
plin2_vals = mg.obs['Plin2_expr'].values
sc2 = ax2.scatter(umap_xy[:, 0], umap_xy[:, 1],
                  c=plin2_vals, cmap='YlOrRd', s=4, alpha=0.8,
                  vmin=0, vmax=np.percentile(plin2_vals, 98), rasterized=True)
plt.colorbar(sc2, ax=ax2, shrink=0.7, label='Plin2 (log-norm)')
ax2.set_title('Plin2 Expression', fontsize=11)
ax2.set_xlabel('UMAP1'); ax2.set_ylabel('UMAP2')
ax2.set_aspect('equal', adjustable='datalim')

# --- subplot 3: condition ---
ax3 = axes_u[2]
for cond, color in COND_COLORS.items():
    mask_c = mg.obs['condition'] == cond
    ax3.scatter(umap_xy[mask_c, 0], umap_xy[mask_c, 1],
                c=color, s=4, alpha=0.6, label=cond, rasterized=True)
ax3.set_title('Condition', fontsize=11)
ax3.set_xlabel('UMAP1'); ax3.set_ylabel('UMAP2')
ax3.legend(fontsize=9, markerscale=3)
ax3.set_aspect('equal', adjustable='datalim')

umap_path = RESULTS_DIR / 'fig8_umap_subclusters.png'
plt.tight_layout()
plt.savefig(umap_path, bbox_inches='tight', dpi=180)
plt.close()
print(f"Saved UMAP: {umap_path}")

# ── 7. Print summary table for markdown ───────────────────────────────────────
print("\n\n=== SUMMARY TABLE (for markdown) ===")
print(f"{'Cluster':<10} {'n_cells':<9} {'Plin2_mean':<12} {'Plin2%+':<10} {'LDAM':>8} {'DAM':>8} {'Intact%':>9} {'Inj%':>7}")
print("-" * 75)
for _, row in plin2_df.iterrows():
    inj_pct = (row['frac_3dpi'] + row['frac_5dpi']) * 100
    print(
        f"  Cluster {row['cluster']:<3} {int(row['n_cells']):<9} "
        f"{row['plin2_mean']:<12.4f} {row['plin2_pct_pos']*100:<10.1f} "
        f"{row.get('LDAM_score', float('nan')):>8.3f} "
        f"{row.get('DAM_score', float('nan')):>8.3f} "
        f"{row['frac_Intact']*100:>9.1f} {inj_pct:>7.1f}"
    )

# ── 8. Marker genes for Plin2-high clusters ───────────────────────────────────
print("\n\nExtracting marker genes for Plin2-high clusters...")

# Identify top-2 Plin2 clusters
top2_cl = list(plin2_df['cluster'].iloc[:2])
print(f"Top-2 Plin2 clusters: {top2_cl}")

# Label: Plin2_high vs rest
mg.obs['plin2_group'] = mg.obs['mg_subcluster'].apply(
    lambda x: 'Plin2_high' if x in top2_cl else 'Other'
)

# Rank genes
sc.tl.rank_genes_groups(mg, groupby='plin2_group', groups=['Plin2_high'],
                        reference='Other', method='wilcoxon', n_genes=200)

# Extract results
marker_names  = mg.uns['rank_genes_groups']['names']['Plin2_high']
marker_scores = mg.uns['rank_genes_groups']['scores']['Plin2_high']
marker_logfc  = mg.uns['rank_genes_groups']['logfoldchanges']['Plin2_high']
marker_pvals  = mg.uns['rank_genes_groups']['pvals_adj']['Plin2_high']

marker_df = pd.DataFrame({
    'gene':    marker_names,
    'score':   marker_scores,
    'log2FC':  marker_logfc,
    'padj':    marker_pvals,
})
marker_df = marker_df[marker_df['padj'] < 0.05].head(100)
marker_df.to_csv(RESULTS_DIR / 'plin2_high_markers.csv', index=False)

print("\nTop 40 marker genes of Plin2-high clusters:")
print(f"{'Gene':<14} {'log2FC':>8} {'padj':>12}")
print("-" * 38)
for _, r in marker_df.head(40).iterrows():
    print(f"  {r['gene']:<12} {r['log2FC']:>8.3f} {r['padj']:>12.2e}")

# Key gene expression in Plin2-high vs other
COMPARE_GENES = [
    'Plin2','Lpl','Cd36','Apoe','Fabp5','Spp1','Cst7','Lgals3',
    'Trem2','Tyrobp','Axl','Mki67','Lamp1','Ctsb','Ctsd','Nceh1',
    'Cyp46a1','Abca1','Abcg1','P2ry12','Tmem119','Il1b','Il10',
    'Megf10','Gulp1','Hmox1','Ch25h','Igf1','Clec7a','Gpnmb',
]
print("\n\n=== KEY GENE EXPRESSION: Plin2-high vs Other ===")
print(f"{'Gene':<14} {'Plin2_high_mean':>18} {'Other_mean':>12} {'log2FC':>8} {'padj':>12}")
print("-" * 68)

ph_mask  = mg.obs['plin2_group'] == 'Plin2_high'
oth_mask = mg.obs['plin2_group'] == 'Other'
pseudo = 0.01
compare_rows = []
for gene in COMPARE_GENES:
    if gene not in mg.var_names:
        continue
    ph_expr  = float(np.asarray(mg[ph_mask,  gene].X.mean(axis=0)).flatten()[0])
    oth_expr = float(np.asarray(mg[oth_mask, gene].X.mean(axis=0)).flatten()[0])
    fc = np.log2((ph_expr + pseudo) / (oth_expr + pseudo))
    # get padj from marker table if available
    padj_val = marker_df.loc[marker_df['gene'] == gene, 'padj'].values
    padj_str = f"{padj_val[0]:.2e}" if len(padj_val) > 0 else "n.s."
    print(f"  {gene:<12} {ph_expr:>18.4f} {oth_expr:>12.4f} {fc:>8.3f} {padj_str:>12}")
    compare_rows.append({'gene': gene, 'plin2_high_mean': ph_expr,
                         'other_mean': oth_expr, 'log2FC': fc, 'padj': padj_str})

pd.DataFrame(compare_rows).to_csv(RESULTS_DIR / 'plin2_high_key_genes.csv', index=False)

print(f"\nSaved: plin2_high_markers.csv, plin2_high_key_genes.csv")
print("\nDone.")