#!/usr/bin/env python3
"""
04_lysosomal_markers_comparison.py
对比 Mehrabani 2025 溶酶体障碍标志基因在以下9个比较组中的表达变化：

Mehrabani (论文 TableS4/S6):
  G1: TBI D3 vs Sham (全体 CD11b+ 细胞)
  G2: DAM3 vs 全体 TBI 小胶质 (TableS6)

GSE226211 (CTRL-only, INH excluded):
  G3: 全体微 Intact vs 3dpi_CTRL
  G4: 全体微 Intact vs 5dpi_CTRL
  G5: Plin2-high (C8+C4) vs 其余全体微 (resolution=0.3)

GSE163691:
  G6: D1 injury vs sham
  G7: D4 injury vs sham
  G8: D7 injury vs sham
  G9: D14 injury vs sham

输出: results/GSE226211/lysosomal_comparison_table.csv
      09_lysosomal_markers_comparison.md
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import mmread
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE      = Path("/Users/maxue/Documents/vscode/tbi")
DATA226   = BASE / "data/GSE226211"
DATA163   = BASE / "data/GSE163691"
SUPPL     = BASE / "data/Mehrabani2025_suppl"
RESULTS   = BASE / "results/GSE226211"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Target genes (from Mehrabani 2025 table in 08_lysosomal_function_analysis.md)
TARGET_GENES = [
    'Lamp1',    # 溶酶体膜标志
    'Ctsd',     # 组织蛋白酶D（活性降低）
    'Ctsb',     # 组织蛋白酶B（转录上调）
    'Naglu',    # N-乙酰葡萄糖胺酶（活性降低）
    'Lgals3',   # 半乳糖凝集素3（溶酶体损伤标志）
    'Lipa',     # 溶酶体酸性脂酶
    'Npc2',     # 尼曼匹克C2
    'Nceh1',    # 中性CE水解酶
    'Anxa5',    # 膜联蛋白A5
    'Sqstm1',   # p62（自噬受体）
    'Map1lc3b', # LC3
]

GENE_DESC = {
    'Lamp1':    '溶酶体膜标志',
    'Ctsd':     '溶酶体蛋白酶（Meh↓活性）',
    'Ctsb':     '溶酶体蛋白酶（Meh↑转录）',
    'Naglu':    '糖苷水解酶（Meh↓活性）',
    'Lgals3':   '溶酶体损伤标志（Meh↑）',
    'Lipa':     '溶酶体脂肪酶（Meh↑）',
    'Npc2':     '溶酶体胆固醇转运（Meh↑）',
    'Nceh1':    'CE水解酶（Meh↑）',
    'Anxa5':    '吞噬-溶酶体关联（Meh↑）',
    'Sqstm1':   'p62自噬受体（Meh↑蛋白）',
    'Map1lc3b': 'LC3自噬体（Meh↑蛋白）',
}

# ── Helper: format a result cell ──────────────────────────────────────────────
def fmt(fc, padj, threshold=0.05):
    if fc is None or pd.isna(fc):
        return 'n/f'
    sig = '*' if (padj is not None and not pd.isna(padj) and padj < threshold) else ''
    arrow = '↑' if fc > 0 else '↓'
    return f"{arrow}{abs(fc):.2f}{sig}"

# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Mehrabani 2025 (从 xlsx 直接读取)
# ══════════════════════════════════════════════════════════════════════════════
print("=== Part 1: Loading Mehrabani 2025 data ===")

# G1: TableS4 - TBI D3 vs Sham
s4 = pd.read_excel(SUPPL / 'TableS4_TBI_vs_SHAM_DEGs.xlsx',
                   sheet_name='TBI_vs_SHAM_DEGs', header=1)
s4.columns = ['p_val','avg_log2FC','pct1','pct2','p_val_adj','log10_pval','log10_padj','gene']
s4 = s4.dropna(subset=['gene'])
s4['gene'] = s4['gene'].astype(str)

# G2: TableS6 - DAM3 vs all TBI microglia
s6 = pd.read_excel(SUPPL / 'TableS6_deg_microglia_sublcluster.xlsx',
                   sheet_name='DE_DAM3_vs_all_microglia_TBI', header=1)
s6 = s6.dropna(subset=['gene'])
s6['gene'] = s6['gene'].astype(str)

def lookup_mehrabani(df, gene, fc_col='avg_log2FC', padj_col='p_val_adj'):
    r = df[df['gene'].str.lower() == gene.lower()]
    if len(r) > 0:
        return float(r.iloc[0][fc_col]), float(r.iloc[0][padj_col])
    return None, None

print(f"  S4 shape: {s4.shape}, S6 shape: {s6.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# Part 2: GSE163691 (直接读取 DEG 文件)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 2: Loading GSE163691 DEGs ===")

gse163_dfs = {}
for tp in ['D1','D4','D7','D14']:
    df = pd.read_csv(DATA163 / f'GSE163691_{tp}_injury_sham-Diff.txt.gz', sep='\t')
    df = df.dropna(subset=['padj','log2FoldChange'])
    gse163_dfs[tp] = df
    print(f"  {tp}: {df.shape[0]} genes")

def lookup_gse163(tp, gene):
    df = gse163_dfs[tp]
    r = df[df['symbol'].str.lower() == gene.lower()]
    if len(r) == 0:
        r = df[df['SYMBOL'].str.lower() == gene.lower()]
    if len(r) > 0:
        return float(r.iloc[0]['log2FoldChange']), float(r.iloc[0]['padj'])
    return None, None

# ══════════════════════════════════════════════════════════════════════════════
# Part 3: GSE226211 — load pre-computed myeloid object (01_scrna_ldam_analysis.py)
# Ensures identical clustering/UMAP as scripts 01, 03 (random_state=0, res=0.3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 3: Loading GSE226211 (mg_myeloid.h5ad) ===")

mg = sc.read_h5ad(RESULTS / 'mg_myeloid.h5ad')
print(f"  Myeloid cells: {mg.shape[0]}  subclusters: {mg.obs['mg_subcluster'].nunique()}")
print(f"  Conditions: {mg.obs['condition'].value_counts().to_dict()}")

n_cl = mg.obs['mg_subcluster'].nunique()

# Plin2-high clusters (top-2 by mean expression)
plin2_expr = np.asarray(mg[:, 'Plin2'].X.todense()).flatten()
mg.obs['Plin2_expr'] = plin2_expr
cl_plin2 = mg.obs.groupby('mg_subcluster')['Plin2_expr'].mean().sort_values(ascending=False)
top2_cl  = list(cl_plin2.index[:2])
print(f"  Top-2 Plin2 clusters: {top2_cl} (mean: {cl_plin2.iloc[0]:.3f}, {cl_plin2.iloc[1]:.3f})")
mg.obs['plin2_group'] = mg.obs['mg_subcluster'].apply(
    lambda x: 'Plin2_high' if x in top2_cl else 'Other'
)

# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Differential expression — GSE226211 three comparisons
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 4: GSE226211 DEG analysis ===")

# G3: 3dpi_CTRL vs Intact (all microglia)
print("  G3: 3dpi_CTRL vs Intact...")
mg_3dpi = mg[mg.obs['condition'].isin(['Intact', '3dpi_CTRL'])].copy()
sc.tl.rank_genes_groups(mg_3dpi, groupby='condition', groups=['3dpi_CTRL'],
                        reference='Intact', method='wilcoxon', n_genes=len(mg.var_names))
res_3dpi = {
    mg_3dpi.uns['rank_genes_groups']['names']['3dpi_CTRL'][i]:
    (float(mg_3dpi.uns['rank_genes_groups']['logfoldchanges']['3dpi_CTRL'][i]),
     float(mg_3dpi.uns['rank_genes_groups']['pvals_adj']['3dpi_CTRL'][i]))
    for i in range(len(mg_3dpi.uns['rank_genes_groups']['names']['3dpi_CTRL']))
}

# G4: 5dpi_CTRL vs Intact (all microglia)
print("  G4: 5dpi_CTRL vs Intact...")
mg_5dpi = mg[mg.obs['condition'].isin(['Intact', '5dpi_CTRL'])].copy()
sc.tl.rank_genes_groups(mg_5dpi, groupby='condition', groups=['5dpi_CTRL'],
                        reference='Intact', method='wilcoxon', n_genes=len(mg.var_names))
res_5dpi = {
    mg_5dpi.uns['rank_genes_groups']['names']['5dpi_CTRL'][i]:
    (float(mg_5dpi.uns['rank_genes_groups']['logfoldchanges']['5dpi_CTRL'][i]),
     float(mg_5dpi.uns['rank_genes_groups']['pvals_adj']['5dpi_CTRL'][i]))
    for i in range(len(mg_5dpi.uns['rank_genes_groups']['names']['5dpi_CTRL']))
}

# G5: Plin2-high vs Other (all microglia)
print("  G5: Plin2-high vs Other...")
sc.tl.rank_genes_groups(mg, groupby='plin2_group', groups=['Plin2_high'],
                        reference='Other', method='wilcoxon', n_genes=len(mg.var_names))
res_plin2 = {
    mg.uns['rank_genes_groups']['names']['Plin2_high'][i]:
    (float(mg.uns['rank_genes_groups']['logfoldchanges']['Plin2_high'][i]),
     float(mg.uns['rank_genes_groups']['pvals_adj']['Plin2_high'][i]))
    for i in range(len(mg.uns['rank_genes_groups']['names']['Plin2_high']))
}

def lookup_226(res_dict, gene):
    if gene in res_dict:
        return res_dict[gene]
    # case-insensitive fallback
    for k, v in res_dict.items():
        if k.lower() == gene.lower():
            return v
    return None, None

# ══════════════════════════════════════════════════════════════════════════════
# Part 5: Build comparison table
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 5: Building comparison table ===")

rows = []
for gene in TARGET_GENES:
    row = {'Gene': gene, '功能描述': GENE_DESC.get(gene, '')}

    # G1: Mehrabani TBI vs Sham
    fc, padj = lookup_mehrabani(s4, gene)
    row['G1_Meh_TBI_vs_Sham'] = fmt(fc, padj)
    row['G1_fc'] = fc; row['G1_padj'] = padj

    # G2: Mehrabani DAM3 vs TBI microglia
    fc, padj = lookup_mehrabani(s6, gene)
    row['G2_Meh_DAM3_vs_TBImic'] = fmt(fc, padj)
    row['G2_fc'] = fc; row['G2_padj'] = padj

    # G3: GSE226211 3dpi vs Intact
    fc, padj = lookup_226(res_3dpi, gene)
    row['G3_226_3dpi_vs_Intact'] = fmt(fc, padj)
    row['G3_fc'] = fc; row['G3_padj'] = padj

    # G4: GSE226211 5dpi vs Intact
    fc, padj = lookup_226(res_5dpi, gene)
    row['G4_226_5dpi_vs_Intact'] = fmt(fc, padj)
    row['G4_fc'] = fc; row['G4_padj'] = padj

    # G5: GSE226211 Plin2-high vs Other
    fc, padj = lookup_226(res_plin2, gene)
    row['G5_226_Plin2high_vs_Other'] = fmt(fc, padj)
    row['G5_fc'] = fc; row['G5_padj'] = padj

    # G6-G9: GSE163691 timepoints
    for i, tp in enumerate(['D1','D4','D7','D14'], start=6):
        fc, padj = lookup_gse163(tp, gene)
        row[f'G{i}_163_{tp}'] = fmt(fc, padj)
        row[f'G{i}_fc'] = fc; row[f'G{i}_padj'] = padj

    rows.append(row)

result_df = pd.DataFrame(rows)

# Display columns
display_cols = ['Gene', '功能描述',
    'G1_Meh_TBI_vs_Sham', 'G2_Meh_DAM3_vs_TBImic',
    'G3_226_3dpi_vs_Intact', 'G4_226_5dpi_vs_Intact', 'G5_226_Plin2high_vs_Other',
    'G6_163_D1', 'G7_163_D4', 'G8_163_D7', 'G9_163_D14']

print("\n" + "="*120)
print(result_df[display_cols].to_string(index=False))
print("="*120)

# Save CSV
csv_path = RESULTS / 'lysosomal_comparison_table.csv'
result_df[display_cols].to_csv(csv_path, index=False)
print(f"\nSaved CSV: {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Generate heatmap figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 6: Generating heatmap ===")

fc_cols   = ['G1_fc','G2_fc','G3_fc','G4_fc','G5_fc','G6_fc','G7_fc','G8_fc','G9_fc']
padj_cols = ['G1_padj','G2_padj','G3_padj','G4_padj','G5_padj','G6_padj','G7_padj','G8_padj','G9_padj']

col_labels = [
    'Meh\nTBI/Sham', 'Meh\nDAM3/TBI',
    'GSE226211\n3dpi/Intact', 'GSE226211\n5dpi/Intact', 'GSE226211\nPlin2hi/Other',
    'GSE163691\nD1', 'GSE163691\nD4', 'GSE163691\nD7', 'GSE163691\nD14'
]

fc_matrix   = result_df[fc_cols].values.astype(float)
padj_matrix = result_df[padj_cols].values.astype(float)

# Cap FC for colour scale
fc_plot = np.clip(fc_matrix, -4, 4)

n_genes = len(TARGET_GENES)   # 11
n_cols  = len(col_labels)     # 9
cell_h, cell_w = 0.7, 1.1    # inches per cell

fig_w = cell_w * n_cols + 5.0   # extra room for y-labels + colorbar
fig_h = cell_h * n_genes + 2.5  # extra room for x-labels + title

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
im = ax.imshow(fc_plot, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')

cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('log2FC (capped ±4)', fontsize=10)

# Mark significance
for i in range(n_genes):
    for j in range(n_cols):
        padj = padj_matrix[i, j]
        fc   = fc_matrix[i, j]
        if not np.isnan(padj) and padj < 0.05:
            ax.text(j, i, '★', ha='center', va='center', fontsize=11,
                    color='white' if abs(fc) > 1.5 else '#222222')

ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=10, ha='center')
ax.xaxis.set_tick_params(length=0)

ax.set_yticks(range(n_genes))
ax.set_yticklabels(
    [f"{g}  {GENE_DESC[g]}" for g in TARGET_GENES],
    fontsize=10, ha='right'
)
ax.yaxis.set_tick_params(length=0)

ax.set_title(
    'Mehrabani 2025 溶酶体障碍标志基因\n在三个刺伤 TBI 数据集中的表达变化  (★ padj<0.05)',
    fontsize=12, fontweight='bold', pad=14
)

# Group dividers
ax.axvline(1.5, color='black', lw=2)
ax.axvline(4.5, color='black', lw=2)

# Group labels above x-axis (use axes-fraction coords)
for x, label, color in [
    (0.5/n_cols, 'Mehrabani 2025', '#1a1a1a'),
    (3.0/n_cols, 'GSE226211 (scRNA-seq)', '#1a1a1a'),
    (7.0/n_cols, 'GSE163691 (bulk DESeq2)', '#1a1a1a'),
]:
    ax.annotate(label, xy=(x, 1.02), xycoords='axes fraction',
                ha='center', va='bottom', fontsize=9,
                color=color, fontstyle='italic')

# Grid lines for readability
for i in range(n_genes + 1):
    ax.axhline(i - 0.5, color='white', lw=0.5)
for j in range(n_cols + 1):
    ax.axvline(j - 0.5, color='white', lw=0.5)

plt.tight_layout(rect=[0, 0, 1, 1])
fig_path = RESULTS / 'fig9_lysosomal_markers_heatmap.png'
plt.savefig(fig_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved figure: {fig_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Part 7: Write markdown table
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part 7: Writing 09_lysosomal_markers_comparison.md ===")

md_path = BASE / '09_lysosomal_markers_comparison.md'

with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# 09. Mehrabani 2025 溶酶体障碍标志基因：多数据集横向比较\n\n")
    f.write("**分析日期**：2026-03-17  \n")
    f.write("**参考文献**：Mehrabani-Tabari et al., *Nat Neurosci* 2025 (PMID:41282218)  \n\n")

    f.write("## 比较组说明\n\n")
    f.write("| 编号 | 数据集 | 比较 | 细胞类型/分析层级 |\n")
    f.write("|------|--------|------|-------------------|\n")
    f.write("| G1 | Mehrabani 2025 TableS4 | TBI D3 vs Sham | 全体 CD11b+ 细胞 (scRNA-seq) |\n")
    f.write("| G2 | Mehrabani 2025 TableS6 | DAM3 vs 全体 TBI 小胶质 | 小胶质亚群 |\n")
    f.write("| G3 | GSE226211 | 3dpi_CTRL vs Intact | 全体微 (Wilcoxon) |\n")
    f.write("| G4 | GSE226211 | 5dpi_CTRL vs Intact | 全体微 (Wilcoxon) |\n")

    top2_str = '+'.join([f'C{c}' for c in top2_cl])
    f.write(f"| G5 | GSE226211 | Plin2-high ({top2_str}) vs 其余小胶质 | 亚群分析 res=0.3 |\n")
    f.write("| G6 | GSE163691 | D1 injury vs sham | 全脑 bulk RNA-seq (DESeq2) |\n")
    f.write("| G7 | GSE163691 | D4 injury vs sham | 全脑 bulk RNA-seq (DESeq2) |\n")
    f.write("| G8 | GSE163691 | D7 injury vs sham | 全脑 bulk RNA-seq (DESeq2) |\n")
    f.write("| G9 | GSE163691 | D14 injury vs sham | 全脑 bulk RNA-seq (DESeq2) |\n\n")

    f.write("> 格式说明：↑/↓ + log2FC 数值，* = padj < 0.05；n/f = 数据中未检出该基因\n\n")
    f.write("> GSE226211 log2FC: Wilcoxon rank-sum；GSE163691 log2FC: DESeq2\n\n")

    f.write("## 溶酶体障碍标志基因比较表\n\n")

    header = "| 基因 | 功能（Mehrabani方向） | G1<br>Meh TBI/Sham | G2<br>Meh DAM3/TBI | G3<br>226 3dpi/Int | G4<br>226 5dpi/Int | G5<br>226 Plin2hi | G6<br>163 D1 | G7<br>163 D4 | G8<br>163 D7 | G9<br>163 D14 |"
    f.write(header + "\n")
    f.write("|" + "---|" * header.count("|") + "\n")

    for _, row in result_df.iterrows():
        f.write(
            f"| **{row['Gene']}** | {row['功能描述']} "
            f"| {row['G1_Meh_TBI_vs_Sham']} | {row['G2_Meh_DAM3_vs_TBImic']} "
            f"| {row['G3_226_3dpi_vs_Intact']} | {row['G4_226_5dpi_vs_Intact']} | {row['G5_226_Plin2high_vs_Other']} "
            f"| {row['G6_163_D1']} | {row['G7_163_D4']} | {row['G8_163_D7']} | {row['G9_163_D14']} |\n"
        )

    f.write("\n## GSE226211 亚群信息\n\n")
    f.write(f"- 分群参数：Leiden resolution = 0.3，共 {n_cl} 个亚群\n")
    f.write(f"- Plin2-high 定义：均值 Plin2 最高的 top-2 亚群 = **{top2_str}**\n")
    f.write(f"  - {top2_cl[0]}: mean Plin2 = {cl_plin2.iloc[0]:.3f}, n = {int((mg.obs['mg_subcluster']==top2_cl[0]).sum())}\n")
    f.write(f"  - {top2_cl[1]}: mean Plin2 = {cl_plin2.iloc[1]:.3f}, n = {int((mg.obs['mg_subcluster']==top2_cl[1]).sum())}\n\n")

    f.write("## 关键基因原始数值（log2FC，padj）\n\n")
    f.write("### GSE226211 G3/G4/G5\n\n")
    f.write("| 基因 | G3 3dpi FC | G3 padj | G4 5dpi FC | G4 padj | G5 Plin2hi FC | G5 padj |\n")
    f.write("|------|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for _, row in result_df.iterrows():
        def fmt_num(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 'n/f'
            return f"{v:.4f}"
        def fmt_p(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 'n/f'
            return f"{v:.2e}"
        f.write(f"| **{row['Gene']}** | {fmt_num(row['G3_fc'])} | {fmt_p(row['G3_padj'])} | {fmt_num(row['G4_fc'])} | {fmt_p(row['G4_padj'])} | {fmt_num(row['G5_fc'])} | {fmt_p(row['G5_padj'])} |\n")

    f.write("\n### GSE163691 G6-G9\n\n")
    f.write("| 基因 | D1 FC | D1 padj | D4 FC | D4 padj | D7 FC | D7 padj | D14 FC | D14 padj |\n")
    f.write("|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for _, row in result_df.iterrows():
        def fmt_num(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 'n/f'
            return f"{v:.4f}"
        def fmt_p(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 'n/f'
            return f"{v:.2e}"
        f.write(f"| **{row['Gene']}** | {fmt_num(row['G6_fc'])} | {fmt_p(row['G6_padj'])} | {fmt_num(row['G7_fc'])} | {fmt_p(row['G7_padj'])} | {fmt_num(row['G8_fc'])} | {fmt_p(row['G8_padj'])} | {fmt_num(row['G9_fc'])} | {fmt_p(row['G9_padj'])} |\n")

    f.write("\n## 数据来源\n\n")
    f.write("| 数据 | 文件 |\n|------|------|\n")
    f.write("| Mehrabani TableS4 | `data/Mehrabani2025_suppl/TableS4_Mehrabani2025.xlsx` |\n")
    f.write("| Mehrabani TableS6 | `data/Mehrabani2025_suppl/TableS6_Mehrabani2025.xlsx` |\n")
    f.write("| GSE226211 | `data/GSE226211/` (10个CTRL+Intact样本) |\n")
    f.write("| GSE163691 | `data/GSE163691/GSE163691_DX_injury_sham-Diff.txt.gz` |\n")
    f.write("| 分析脚本 | `scripts/GSE226211/04_lysosomal_markers_comparison.py` |\n")
    f.write("| 热图 | `results/GSE226211/fig9_lysosomal_markers_heatmap.png` |\n")
    f.write("| CSV | `results/GSE226211/lysosomal_comparison_table.csv` |\n")

print(f"Saved markdown: {md_path}")
print("\nDone.")
