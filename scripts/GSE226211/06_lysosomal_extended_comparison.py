#!/usr/bin/env python3
"""
06_lysosomal_extended_comparison.py
扩展版：对比 08_lysosomal_function_analysis.md 中出现的所有基因（~40个，8个功能类别）
在9个比较组中的表达变化。

比较组（同 04）：
  G1: Mehrabani TableS4 — TBI D3 vs Sham (全体 CD11b+)
  G2: Mehrabani TableS6 — DAM3 vs 全体 TBI 微
  G3: GSE226211 — 3dpi_CTRL vs Intact (全体微)
  G4: GSE226211 — 5dpi_CTRL vs Intact (全体微)
  G5: GSE226211 — Plin2-high (C8+C4) vs Other (res=0.3)
  G6-G9: GSE163691 — D1/D4/D7/D14 injury vs sham

输出:
  results/GSE226211/lysosomal_extended_table.csv
  results/GSE226211/fig10_lysosomal_extended_heatmap.png
  10_lysosomal_extended_comparison.md
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.io import mmread
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE    = Path("/Users/maxue/Documents/vscode/tbi")
DATA226 = BASE / "data/GSE226211"
DATA163 = BASE / "data/GSE163691"
SUPPL   = BASE / "data/Mehrabani2025_suppl"
RESULTS = BASE / "results/GSE226211"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── All genes from 08_lysosomal_function_analysis.md, organized by category ───
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

# Flat list preserving category order
TARGET_GENES = [g for cat in GENE_CATEGORIES.values() for g in cat]
GENE_DESC    = {g: d for cat in GENE_CATEGORIES.values() for g, d in cat.items()}
GENE_CAT     = {g: cat for cat, genes in GENE_CATEGORIES.items() for g in genes}

print(f"Total genes to compare: {len(TARGET_GENES)}")

# ── Helper ─────────────────────────────────────────────────────────────────────
def fmt(fc, padj, threshold=0.05):
    if fc is None or pd.isna(fc):
        return 'n/f'
    sig = '*' if (padj is not None and not pd.isna(padj) and padj < threshold) else ''
    arrow = '↑' if fc > 0 else '↓'
    return f"{arrow}{abs(fc):.2f}{sig}"

# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Mehrabani (xlsx)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 1: Mehrabani 2025 ===")
s4 = pd.read_excel(SUPPL / 'TableS4_TBI_vs_SHAM_DEGs.xlsx', sheet_name='TBI_vs_SHAM_DEGs', header=1)
s4.columns = ['p_val','avg_log2FC','pct1','pct2','p_val_adj','log10_pval','log10_padj','gene']
s4 = s4.dropna(subset=['gene']); s4['gene'] = s4['gene'].astype(str)

s6 = pd.read_excel(SUPPL / 'TableS6_deg_microglia_sublcluster.xlsx',
                   sheet_name='DE_DAM3_vs_all_microglia_TBI', header=1)
s6 = s6.dropna(subset=['gene']); s6['gene'] = s6['gene'].astype(str)

def lkup_meh(df, gene, fc='avg_log2FC', p='p_val_adj'):
    r = df[df['gene'].str.lower() == gene.lower()]
    return (float(r.iloc[0][fc]), float(r.iloc[0][p])) if len(r) > 0 else (np.nan, np.nan)

# ══════════════════════════════════════════════════════════════════════════════
# Part 2: GSE163691
# ══════════════════════════════════════════════════════════════════════════════
print("=== Part 2: GSE163691 ===")
gse163 = {}
for tp in ['D1','D4','D7','D14']:
    df = pd.read_csv(DATA163 / f'GSE163691_{tp}_injury_sham-Diff.txt.gz', sep='\t')
    gse163[tp] = df.dropna(subset=['padj','log2FoldChange'])
    print(f"  {tp}: {len(gse163[tp])} genes")

def lkup_163(tp, gene):
    df = gse163[tp]
    r = df[df['symbol'].str.lower() == gene.lower()]
    if len(r) == 0:
        r = df[df['SYMBOL'].str.lower() == gene.lower()]
    return (float(r.iloc[0]['log2FoldChange']), float(r.iloc[0]['padj'])) if len(r) > 0 else (np.nan, np.nan)

# ══════════════════════════════════════════════════════════════════════════════
# Part 3: GSE226211 — load pre-computed myeloid object (01_scrna_ldam_analysis.py)
# Ensures identical clustering as scripts 01, 03, 04 (random_state=0, res=0.3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 3: GSE226211 (mg_myeloid.h5ad) ===")

mg = sc.read_h5ad(RESULTS / 'mg_myeloid.h5ad')
print(f"  Myeloid cells: {mg.shape[0]}  subclusters: {mg.obs['mg_subcluster'].nunique()}")
print(f"  Conditions: {mg.obs['condition'].value_counts().to_dict()}")

plin2_expr = np.asarray(mg[:, 'Plin2'].X.todense()).flatten()
mg.obs['Plin2_expr'] = plin2_expr
cl_plin2 = mg.obs.groupby('mg_subcluster')['Plin2_expr'].mean().sort_values(ascending=False)
top2_cl  = list(cl_plin2.index[:2])
mg.obs['plin2_group'] = mg.obs['mg_subcluster'].apply(
    lambda x: 'Plin2_high' if x in top2_cl else 'Other')
print(f"  Plin2-high clusters: {top2_cl} (mean: {cl_plin2.iloc[0]:.3f}, {cl_plin2.iloc[1]:.3f})")

# G3: 3dpi vs Intact
print("  DEG G3: 3dpi_CTRL vs Intact...")
mg3 = mg[mg.obs['condition'].isin(['Intact','3dpi_CTRL'])].copy()
sc.tl.rank_genes_groups(mg3, groupby='condition', groups=['3dpi_CTRL'],
                        reference='Intact', method='wilcoxon', n_genes=len(mg.var_names))
res_3dpi = {mg3.uns['rank_genes_groups']['names']['3dpi_CTRL'][i]:
            (float(mg3.uns['rank_genes_groups']['logfoldchanges']['3dpi_CTRL'][i]),
             float(mg3.uns['rank_genes_groups']['pvals_adj']['3dpi_CTRL'][i]))
            for i in range(len(mg3.uns['rank_genes_groups']['names']['3dpi_CTRL']))}

# G4: 5dpi vs Intact
print("  DEG G4: 5dpi_CTRL vs Intact...")
mg5 = mg[mg.obs['condition'].isin(['Intact','5dpi_CTRL'])].copy()
sc.tl.rank_genes_groups(mg5, groupby='condition', groups=['5dpi_CTRL'],
                        reference='Intact', method='wilcoxon', n_genes=len(mg.var_names))
res_5dpi = {mg5.uns['rank_genes_groups']['names']['5dpi_CTRL'][i]:
            (float(mg5.uns['rank_genes_groups']['logfoldchanges']['5dpi_CTRL'][i]),
             float(mg5.uns['rank_genes_groups']['pvals_adj']['5dpi_CTRL'][i]))
            for i in range(len(mg5.uns['rank_genes_groups']['names']['5dpi_CTRL']))}

# G5: Plin2-high vs Other
print("  DEG G5: Plin2-high vs Other...")
sc.tl.rank_genes_groups(mg, groupby='plin2_group', groups=['Plin2_high'],
                        reference='Other', method='wilcoxon', n_genes=len(mg.var_names))
res_ph = {mg.uns['rank_genes_groups']['names']['Plin2_high'][i]:
          (float(mg.uns['rank_genes_groups']['logfoldchanges']['Plin2_high'][i]),
           float(mg.uns['rank_genes_groups']['pvals_adj']['Plin2_high'][i]))
          for i in range(len(mg.uns['rank_genes_groups']['names']['Plin2_high']))}

def lkup_226(res, gene):
    if gene in res: return res[gene]
    for k, v in res.items():
        if k.lower() == gene.lower(): return v
    return np.nan, np.nan

# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Build comparison table
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 4: Building table ===")

rows = []
for gene in TARGET_GENES:
    row = {'Gene': gene, 'Category': GENE_CAT[gene], 'Description': GENE_DESC[gene]}

    fc, p = lkup_meh(s4, gene);  row['G1_fc']=fc; row['G1_p']=p; row['G1']=fmt(fc,p)
    fc, p = lkup_meh(s6, gene);  row['G2_fc']=fc; row['G2_p']=p; row['G2']=fmt(fc,p)
    fc, p = lkup_226(res_3dpi, gene); row['G3_fc']=fc; row['G3_p']=p; row['G3']=fmt(fc,p)
    fc, p = lkup_226(res_5dpi, gene); row['G4_fc']=fc; row['G4_p']=p; row['G4']=fmt(fc,p)
    fc, p = lkup_226(res_ph,   gene); row['G5_fc']=fc; row['G5_p']=p; row['G5']=fmt(fc,p)
    for i, tp in enumerate(['D1','D4','D7','D14'], start=6):
        fc, p = lkup_163(tp, gene)
        row[f'G{i}_fc']=fc; row[f'G{i}_p']=p; row[f'G{i}']=fmt(fc,p)
    rows.append(row)

df = pd.DataFrame(rows)
disp_cols = ['Gene','Category','Description',
             'G1','G2','G3','G4','G5','G6','G7','G8','G9']
df[disp_cols].to_csv(RESULTS / 'lysosomal_extended_table.csv', index=False)
print(df[disp_cols].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Part 5: Heatmap (category-grouped, two panels side by side)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 5: Heatmap ===")

fc_cols   = ['G1_fc','G2_fc','G3_fc','G4_fc','G5_fc','G6_fc','G7_fc','G8_fc','G9_fc']
padj_cols = ['G1_p', 'G2_p', 'G3_p', 'G4_p', 'G5_p', 'G6_p', 'G7_p', 'G8_p', 'G9_p']

fc_mat   = df[fc_cols].values.astype(float)
padj_mat = df[padj_cols].values.astype(float)
fc_plot  = np.clip(fc_mat, -4, 4)

n_genes, n_cols = len(TARGET_GENES), 9
cell_h, cell_w  = 0.55, 1.4
fig_w = cell_w * n_cols + 7.0
fig_h = cell_h * n_genes + 5.0

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.subplots_adjust(top=0.82, bottom=0.08, left=0.38, right=0.88)

im = ax.imshow(fc_plot, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.025)
cbar.set_label('log2FC (capped ±4)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Stars + FC values
for i in range(n_genes):
    for j in range(n_cols):
        if not np.isnan(padj_mat[i,j]) and padj_mat[i,j] < 0.05:
            fc = fc_mat[i,j]
            ax.text(j, i, '★', ha='center', va='center', fontsize=11,
                    color='white' if abs(fc) > 1.5 else '#111')
        fc = fc_mat[i,j]
        if not np.isnan(fc):
            ax.text(j, i+0.35, f"{fc:+.1f}", ha='center', va='center',
                    fontsize=7.5, color='white' if abs(fc) > 2.5 else '#444')

# x-axis
col_labels = [
    'Meh\nTBI/Sham','Meh\nDAM3/TBI',
    'GSE226211\n3dpi/Intact','GSE226211\n5dpi/Intact','GSE226211\nPlin2hi/Other',
    'GSE163691\nD1','GSE163691\nD4','GSE163691\nD7','GSE163691\nD14'
]
ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=11, ha='center', linespacing=1.3)
ax.xaxis.set_tick_params(length=0, pad=6)

# y-axis: gene name + description
ax.set_yticks(range(n_genes))
ax.set_yticklabels([f"{g}  —  {GENE_DESC[g]}" for g in TARGET_GENES], fontsize=9)
ax.yaxis.set_tick_params(length=0)

# Cell grid
for i in range(n_genes+1):
    ax.axhline(i-0.5, color='white', lw=0.6)
for j in range(n_cols+1):
    ax.axvline(j-0.5, color='white', lw=0.6)

# Group dividers (column)
ax.axvline(1.5, color='black', lw=2)
ax.axvline(4.5, color='black', lw=2)

# Category dividers (row) + category labels
cat_boundaries = []
cat_mids = []
idx = 0
for cat, genes in GENE_CATEGORIES.items():
    n = len(genes)
    cat_boundaries.append(idx + n - 0.5)
    cat_mids.append(idx + n/2 - 0.5)
    idx += n

for b in cat_boundaries[:-1]:
    ax.axhline(b, color='#555', lw=1.2, linestyle='--')

# Category labels on the left (outside axes)
cat_colors = plt.cm.Set2(np.linspace(0, 1, len(GENE_CATEGORIES)))
for i, (cat, mid) in enumerate(zip(GENE_CATEGORIES.keys(), cat_mids)):
    ax.annotate(
        cat,
        xy=(-0.01, mid), xycoords=('axes fraction', 'data'),
        ha='right', va='center',
        fontsize=9, color=cat_colors[i],
        fontweight='bold',
        annotation_clip=False
    )

# Group header labels above heatmap
for xfrac, label in [
    (1.0/n_cols, 'Mehrabani 2025'),
    (3.0/n_cols, 'GSE226211 (scRNA-seq)'),
    (7.0/n_cols, 'GSE163691 (bulk DESeq2)'),
]:
    ax.annotate(label, xy=(xfrac, 1.025), xycoords='axes fraction',
                ha='center', va='bottom',
                fontsize=11.5, color='#111', fontstyle='italic', fontweight='bold')

# Main title
fig.text(0.63, 0.89,
         '08_lysosomal_function_analysis.md — All Genes: 9-Group Expression Comparison\n'
         '(★ padj<0.05;  dashed lines = functional category boundaries)',
         ha='center', va='bottom', fontsize=12, fontweight='bold', linespacing=1.5)

out = RESULTS / 'fig10_lysosomal_extended_heatmap.png'
plt.savefig(out, bbox_inches='tight', dpi=160)
plt.close()
print(f"Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Write markdown
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 6: Writing markdown ===")

top2_str = '+'.join([f'C{c}' for c in top2_cl])
md_path = BASE / '10_lysosomal_extended_comparison.md'

with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# 10. 溶酶体相关基因完整比较（08分析文件所有基因）\n\n")
    f.write("**来源**：`08_lysosomal_function_analysis.md` 中出现的所有基因  \n")
    f.write(f"**基因总数**：{len(TARGET_GENES)} 个，分 {len(GENE_CATEGORIES)} 个功能类别  \n")
    f.write("**分析日期**：2026-03-17  \n\n")

    f.write("## 比较组\n\n")
    f.write("| 编号 | 数据集 | 比较 | 层级 |\n|------|--------|------|------|\n")
    f.write("| G1 | Mehrabani 2025 TableS4 | TBI D3 vs Sham | 全体 CD11b+ (scRNA-seq) |\n")
    f.write("| G2 | Mehrabani 2025 TableS6 | DAM3 vs 全体 TBI 微 | 小胶质亚群 |\n")
    f.write("| G3 | GSE226211 | 3dpi_CTRL vs Intact | 全体微 (Wilcoxon) |\n")
    f.write("| G4 | GSE226211 | 5dpi_CTRL vs Intact | 全体微 (Wilcoxon) |\n")
    f.write(f"| G5 | GSE226211 | Plin2-high ({top2_str}) vs Other | res=0.3 亚群 |\n")
    f.write("| G6 | GSE163691 | D1 injury vs sham | bulk DESeq2 |\n")
    f.write("| G7 | GSE163691 | D4 injury vs sham | bulk DESeq2 |\n")
    f.write("| G8 | GSE163691 | D7 injury vs sham | bulk DESeq2 |\n")
    f.write("| G9 | GSE163691 | D14 injury vs sham | bulk DESeq2 |\n\n")
    f.write("> ↑/↓ + log2FC，* = padj<0.05；n/f = 数据中未检出\n\n")

    for cat, genes in GENE_CATEGORIES.items():
        f.write(f"## {cat}\n\n")
        f.write("| 基因 | 描述 | G1<br>Meh TBI | G2<br>Meh DAM3 | G3<br>226 3dpi | G4<br>226 5dpi | G5<br>226 Plin2hi | G6<br>D1 | G7<br>D4 | G8<br>D7 | G9<br>D14 |\n")
        f.write("|" + "---|"*11 + "\n")
        for gene in genes:
            r = df[df['Gene']==gene].iloc[0]
            f.write(f"| **{gene}** | {GENE_DESC[gene]} "
                    f"| {r['G1']} | {r['G2']} | {r['G3']} | {r['G4']} | {r['G5']} "
                    f"| {r['G6']} | {r['G7']} | {r['G8']} | {r['G9']} |\n")
        f.write("\n")

    f.write("## 数据来源\n\n")
    f.write("| 文件 | 内容 |\n|------|------|\n")
    f.write("| `data/Mehrabani2025_suppl/TableS4_Mehrabani2025.xlsx` | Mehrabani TBI D3 vs Sham |\n")
    f.write("| `data/Mehrabani2025_suppl/TableS6_Mehrabani2025.xlsx` | Mehrabani DAM3 vs TBI microglia |\n")
    f.write("| `data/GSE226211/` | 10个CTRL+Intact样本（INH excluded）|\n")
    f.write("| `data/GSE163691/GSE163691_DX_injury_sham-Diff.txt.gz` | bulk DESeq2 D1-D14 |\n")
    f.write("| `results/GSE226211/lysosomal_extended_table.csv` | 完整数值表 |\n")
    f.write("| `results/GSE226211/fig10_lysosomal_extended_heatmap.png` | 热图 |\n")
    f.write("| `scripts/GSE226211/06_lysosomal_extended_comparison.py` | 本脚本 |\n")

print(f"Saved: {md_path}")
print("\nDone.")
