#!/usr/bin/env python3
"""
05_plot_lysosomal_heatmap.py
重新生成 fig9_lysosomal_markers_heatmap.png
依赖：results/GSE226211/lysosomal_comparison_table.csv（由 04_lysosomal_markers_comparison.py 生成）
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("/Users/maxue/Documents/vscode/tbi/results/GSE226211")

TARGET_GENES = ['Lamp1','Ctsd','Ctsb','Naglu','Lgals3','Lipa','Npc2','Nceh1','Anxa5','Sqstm1','Map1lc3b']
GENE_DESC_EN = {
    'Lamp1':    'Lysosomal membrane',
    'Ctsd':     'Lysosomal protease (Meh: activity↓)',
    'Ctsb':     'Lysosomal protease (Meh: transcript↑)',
    'Naglu':    'Glycosidase (Meh: activity↓)',
    'Lgals3':   'Lysosomal damage marker (Meh↑)',
    'Lipa':     'Lysosomal lipase (Meh↑)',
    'Npc2':     'Lysosomal cholesterol carrier (Meh↑)',
    'Nceh1':    'CE hydrolase (Meh↑)',
    'Anxa5':    'Phago-lysosomal (Meh↑)',
    'Sqstm1':   'p62 autophagy receptor (Meh: protein↑)',
    'Map1lc3b': 'LC3 autophagosome (Meh: protein↑)',
}

result_df = pd.read_csv(RESULTS / 'lysosomal_comparison_table.csv')

def parse_cell(s):
    if pd.isna(s) or str(s) in ('n/f', ''):
        return np.nan, False
    s = str(s).strip()
    sig = s.endswith('*')
    s = s.rstrip('*')
    sign = 1 if s.startswith('↑') else -1
    try:
        return sign * float(s[1:]), sig
    except:
        return np.nan, False

col_map = {
    'G1': 'G1_Meh_TBI_vs_Sham',
    'G2': 'G2_Meh_DAM3_vs_TBImic',
    'G3': 'G3_226_3dpi_vs_Intact',
    'G4': 'G4_226_5dpi_vs_Intact',
    'G5': 'G5_226_Plin2high_vs_Other',
    'G6': 'G6_163_D1',
    'G7': 'G7_163_D4',
    'G8': 'G8_163_D7',
    'G9': 'G9_163_D14',
}

n_genes, n_cols = len(TARGET_GENES), 9
fc_mat  = np.full((n_genes, n_cols), np.nan)
sig_mat = np.zeros((n_genes, n_cols), dtype=bool)

for i, gene in enumerate(TARGET_GENES):
    row = result_df[result_df['Gene'] == gene].iloc[0]
    for j, (_, col) in enumerate(col_map.items()):
        fc, sig = parse_cell(row[col])
        fc_mat[i, j]  = fc
        sig_mat[i, j] = sig

fc_plot = np.clip(fc_mat, -4, 4)

col_labels = [
    'Meh\nTBI/Sham', 'Meh\nDAM3/TBI',
    'GSE226211\n3dpi/Intact', 'GSE226211\n5dpi/Intact', 'GSE226211\nPlin2hi/Other',
    'GSE163691\nD1', 'GSE163691\nD4', 'GSE163691\nD7', 'GSE163691\nD14'
]

# ── Figure & axes layout ───────────────────────────────────────────────────────
cell_h, cell_w = 0.9, 1.5
fig_w = cell_w * n_cols + 6.5
fig_h = cell_h * n_genes + 4.5

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# ── Adjust heatmap position ────────────────────────────────────────────────────
fig.subplots_adjust(top=0.75, bottom=0.12, left=0.30, right=0.88)

# ── Heatmap ────────────────────────────────────────────────────────────────────
im = ax.imshow(fc_plot, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.025)
cbar.set_label('log2FC (capped ±4)', fontsize=13)
cbar.ax.tick_params(labelsize=11)

# Stars + FC values
for i in range(n_genes):
    for j in range(n_cols):
        if sig_mat[i, j]:
            fc = fc_mat[i, j]
            ax.text(j, i, '★', ha='center', va='center', fontsize=14,
                    color='white' if abs(fc) > 1.5 else '#111111')
        fc = fc_mat[i, j]
        if not np.isnan(fc):
            txt_color = 'white' if abs(fc) > 2.5 else '#333'
            ax.text(j, i + 0.36, f"{fc:+.1f}", ha='center', va='center',
                    fontsize=9, color=txt_color)

# Tick labels
ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=12, ha='center', linespacing=1.4)
ax.xaxis.set_tick_params(length=0, pad=8)

ax.set_yticks(range(n_genes))
ax.set_yticklabels([f"{g}  —  {GENE_DESC_EN[g]}" for g in TARGET_GENES], fontsize=11.5)
ax.yaxis.set_tick_params(length=0)

# Cell grid lines
for i in range(n_genes + 1):
    ax.axhline(i - 0.5, color='white', lw=0.8)
for j in range(n_cols + 1):
    ax.axvline(j - 0.5, color='white', lw=0.8)

# Group dividers
ax.axvline(1.5, color='black', lw=2)
ax.axvline(4.5, color='black', lw=2)

# ── Group labels: above heatmap top edge ───────────────────────────────────────
# Adjust y value (1.03) to move labels closer/farther from heatmap top
GROUP_INFO = [
    (1.0 / n_cols, 'Mehrabani 2025'),
    (3.0 / n_cols, 'GSE226211 (scRNA-seq)'),
    (7.0 / n_cols, 'GSE163691 (bulk DESeq2)'),
]
for xfrac, label in GROUP_INFO:
    ax.annotate(
        label,
        xy=(xfrac, 1.03), xycoords='axes fraction',
        ha='center', va='bottom',
        fontsize=12, color='#222', fontstyle='italic', fontweight='bold'
    )

# ── Main title: in figure coords ───────────────────────────────────────────────
# Adjust y (0.89) to move title up/down
fig.text(
    0.53, 0.79,
    'Mehrabani 2025 Lysosomal Dysfunction Marker Genes\n'
    'Expression Changes in Three Cortical Stab-Wound TBI Datasets  (★ padj<0.05)',
    ha='center', va='bottom',
    fontsize=14, fontweight='bold', linespacing=1.5
)

# ── Save ───────────────────────────────────────────────────────────────────────
out = RESULTS / 'fig9_lysosomal_markers_heatmap.png'
plt.savefig(out, bbox_inches='tight', dpi=160)
plt.close()
print(f"Saved: {out}")
