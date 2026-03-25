#!/usr/bin/env python3
"""
07_plot_lysosomal_extended_heatmap.py
单独重新生成 fig10_lysosomal_extended_heatmap.png
依赖：results/GSE226211/lysosomal_extended_table.csv（由 06 生成）

运行：
  python3 scripts/GSE226211/07_plot_lysosomal_extended_heatmap.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("/Users/maxue/Documents/vscode/tbi/results/GSE226211")

# ── Gene categories (order determines row order in heatmap) ───────────────────
GENE_CATEGORIES = {
    'Lysosomal membrane': {
        'Lamp1': 'Lysosomal membrane protein 1 (Meh: IF marker)',
        'Lamp2': 'Lysosomal membrane protein 2',
        'Cd63':  'Lysosome-phagosome fusion marker',
        'Cd68':  'Lysosomal macrophage marker',
    },
    'Cathepsins': {
        'Ctsb': 'Cysteine protease (Meh: transcript↑)',
        'Ctsd': 'Aspartyl protease (Meh: activity↓)',
        'Ctsl': 'Cysteine protease',
        'Ctse': 'Aspartyl protease',
    },
    'Lysosomal hydrolases': {
        'Lipa':   'Lysosomal acid lipase CE/TG (Meh↑)',
        'Gba':    'Glucocerebrosidase',
        'Galc':   'Galactosylceramidase (myelin lipid)',
        'Hexa':   'Hexosaminidase A',
        'Gusb':   'Beta-glucuronidase',
        'Naglu':  'N-acetylglucosaminidase (Meh: activity↓)',
        'Lgals3': 'Galectin-3, lysosomal damage marker (Meh↑)',
        'Anxa5':  'Phago-lysosomal coupling (Meh↑)',
    },
    'Lysosomal acidification': {
        'Atp6v0d1': 'V-ATPase subunit d1',
        'Atp6v1h':  'V-ATPase subunit h',
        'Lamtor1':  'mTORC1-lysosome platform (Ragulator)',
    },
    'Intralysosomal cholesterol': {
        'Npc1':  'Lysosomal cholesterol export pump',
        'Npc2':  'Intralysosomal cholesterol carrier (Meh↑)',
        'Nceh1': 'Neutral CE hydrolase (Meh↑)',
    },
    'Cholesterol export / esterification': {
        'Cyp46a1': 'Cholesterol→24-OHC (primary export)',
        'Abca1':   'Cellular cholesterol efflux',
        'Abcg1':   'Cellular cholesterol efflux',
        'Osbpl1a': 'ER-lysosome sterol transfer',
        'Soat1':   'Cholesterol esterification → CE/LD',
        'Ch25h':   '25-hydroxylase (oxysterol bypass)',
    },
    'Phagocytic uptake': {
        'Cd36':  'Lipid/myelin receptor (Meh: myelin uptake)',
        'Msr1':  'Scavenger receptor (myelin uptake)',
        'Axl':   'TAM receptor, myelin phagocytosis',
        'Mertk': 'TAM receptor, myelin phagocytosis',
    },
    'Autophagy-lysosome': {
        'Sqstm1':   'p62 autophagy receptor (Meh: protein↑)',
        'Map1lc3b': 'LC3 autophagosome marker (Meh: protein↑)',
        'Becn1':    'Autophagy initiation (Beclin-1)',
        'Tfeb':     'Lysosome biogenesis TF (CLEAR network)',
        'Tfec':     'TFEB family (phagocytic activation)',
        'Plin2':    'Lipid droplet coat protein (LD marker)',
        'Il1b':     'Inflammation / NF-κB target',
    },
}

TARGET_GENES = [g for cat in GENE_CATEGORIES.values() for g in cat]
GENE_DESC    = {g: d for cat in GENE_CATEGORIES.values() for g, d in cat.items()}
GENE_CAT     = {g: cat for cat, genes in GENE_CATEGORIES.items() for g in genes}

# ── Load pre-computed CSV ─────────────────────────────────────────────────────
df = pd.read_csv(RESULTS / 'lysosomal_extended_table.csv')

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
    'G1': 'G1', 'G2': 'G2', 'G3': 'G3', 'G4': 'G4', 'G5': 'G5',
    'G6': 'G6', 'G7': 'G7', 'G8': 'G8', 'G9': 'G9',
}

n_genes, n_cols = len(TARGET_GENES), 9
fc_mat  = np.full((n_genes, n_cols), np.nan)
sig_mat = np.zeros((n_genes, n_cols), dtype=bool)

for i, gene in enumerate(TARGET_GENES):
    rows = df[df['Gene'] == gene]
    if len(rows) == 0:
        continue
    row = rows.iloc[0]
    for j, col in enumerate(['G1','G2','G3','G4','G5','G6','G7','G8','G9']):
        fc, sig = parse_cell(row[col])
        fc_mat[i, j]  = fc
        sig_mat[i, j] = sig

fc_plot = np.clip(fc_mat, -4, 4)

col_labels = [
    'Meh\nTBI/Sham', 'Meh\nDAM3/TBI',
    'GSE226211\n3dpi/Intact', 'GSE226211\n5dpi/Intact', 'GSE226211\nPlin2hi/Other',
    'GSE163691\nD1', 'GSE163691\nD4', 'GSE163691\nD7', 'GSE163691\nD14'
]

# ── Layout ────────────────────────────────────────────────────────────────────
cell_h, cell_w = 0.55, 1.4
fig_w = cell_w * n_cols + 7.0
fig_h = cell_h * n_genes + 5.0

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# ── Adjust heatmap position ────────────────────────────────────────────────────
fig.subplots_adjust(top=0.82, bottom=0.08, left=0.38, right=0.88)

# ── Heatmap ───────────────────────────────────────────────────────────────────
im = ax.imshow(fc_plot, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.025)
cbar.set_label('log2FC (capped ±4)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Stars + FC values
for i in range(n_genes):
    for j in range(n_cols):
        if sig_mat[i, j]:
            fc = fc_mat[i, j]
            ax.text(j, i, '★', ha='center', va='center', fontsize=11,
                    color='white' if abs(fc) > 1.5 else '#111')
        fc = fc_mat[i, j]
        if not np.isnan(fc):
            ax.text(j, i + 0.35, f"{fc:+.1f}", ha='center', va='center',
                    fontsize=7.5, color='white' if abs(fc) > 2.5 else '#444')

# x-axis
ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=11, ha='center', linespacing=1.3)
ax.xaxis.set_tick_params(length=0, pad=6)

# y-axis
ax.set_yticks(range(n_genes))
ax.set_yticklabels([f"{g}  —  {GENE_DESC[g]}" for g in TARGET_GENES], fontsize=9)
ax.yaxis.set_tick_params(length=0)

# Cell grid
for i in range(n_genes + 1):
    ax.axhline(i - 0.5, color='white', lw=0.6)
for j in range(n_cols + 1):
    ax.axvline(j - 0.5, color='white', lw=0.6)

# Column group dividers
ax.axvline(1.5, color='black', lw=2)
ax.axvline(4.5, color='black', lw=2)

# Row category dividers + category labels
cat_colors = plt.cm.Set2(np.linspace(0, 1, len(GENE_CATEGORIES)))
idx = 0
for ci, (cat, genes) in enumerate(GENE_CATEGORIES.items()):
    n = len(genes)
    mid = idx + n / 2 - 0.5
    boundary = idx + n - 0.5

    # dashed separator between categories
    if ci < len(GENE_CATEGORIES) - 1:
        ax.axhline(boundary, color='#555', lw=1.2, linestyle='--')

    # category label on the left
    ax.annotate(
        cat,
        xy=(-0.01, mid), xycoords=('axes fraction', 'data'),
        ha='right', va='center',
        fontsize=9, color=cat_colors[ci],
        fontweight='bold',
        annotation_clip=False
    )
    idx += n

# Group header labels above heatmap
# Adjust y value (1.025) to move labels closer/farther from heatmap top
for xfrac, label in [
    (1.0 / n_cols, 'Mehrabani 2025'),
    (3.7 / n_cols, 'GSE226211 (scRNA-seq)'),
    (7.0 / n_cols, 'GSE163691 (bulk DESeq2)'),
]:
    ax.annotate(label, xy=(xfrac, 1.01), xycoords='axes fraction',
                ha='center', va='bottom',
                fontsize=11.5, color='#111', fontstyle='italic', fontweight='bold')

# Main title
# Adjust y (0.89) to move title up/down relative to figure
fig.text(
    0.55, 0.84,
    '08_lysosomal_function_analysis.md — All Genes: 9-Group Expression Comparison\n'
    '(★ padj<0.05;  dashed lines = functional category boundaries)',
    ha='center', va='bottom',
    fontsize=12, fontweight='bold', linespacing=1.5
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = RESULTS / 'fig10_lysosomal_extended_heatmap.png'
plt.savefig(out, bbox_inches='tight', dpi=160)
plt.close()
print(f"Saved: {out}")
