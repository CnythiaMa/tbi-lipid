"""
肝细胞炎症性表型重编程分析
验证假设：脑外伤后肝细胞基因表达特征向 Kupffer 细胞方向偏移

方法：
  1. 从 control 核中，用 one-vs-rest Wilcoxon 定义 Kupffer 和 Hepatocyte 特征签名
  2. 用 scanpy score_genes（改良 Seurat AddModuleScore）对肝细胞评分
  3. 计算身份比值，统计 control vs 7dpi 的偏移显著性
  4. 检查哪些 Kupffer marker 基因在肝细胞中显著上调
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

sc.settings.verbosity = 0

RESULT_DIR = '/Users/maxue/Documents/vscode/tbi/results/liver_snRNA-seq'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures')
TAB_DIR    = os.path.join(RESULT_DIR, 'tables')

adata = sc.read_h5ad(os.path.join(RESULT_DIR, 'liver_snRNA_final.h5ad'))
print(f"Loaded: {adata.n_obs} nuclei")

# ══════════════════════════════════════════════════════════════════════
# STEP 1: 从 Control 核定义纯净的细胞类型特征签名（one-vs-rest）
# ══════════════════════════════════════════════════════════════════════
def marker_genes_one_vs_rest(adata_sub, cell_type, n_genes=100,
                              min_fc=1.0, min_pct=0.10):
    """
    Wilcoxon one-vs-rest marker 基因筛选
    - 只考虑在目标细胞类型中表达比例 > min_pct 的基因
    - 只保留 log2FC >= min_fc 的基因（目标 vs 其余均值比）
    返回按 log2FC 降序排列的 DataFrame
    """
    target = adata_sub.obs['cell_type_broad'] == cell_type
    rest   = ~target

    X_all = adata_sub.raw.X
    if issparse(X_all): X_all = X_all.toarray()

    pct_target = (X_all[target, :] > 0).mean(axis=0)
    gene_mask  = pct_target > min_pct
    genes_use  = np.array(adata_sub.raw.var_names)[gene_mask]
    X_filt     = X_all[:, gene_mask]

    X_t = X_filt[target, :]
    X_r = X_filt[rest,   :]

    results = []
    for i, gene in enumerate(genes_use):
        stat, pval = stats.mannwhitneyu(X_t[:, i], X_r[:, i], alternative='greater')
        mean_t = X_t[:, i].mean()
        mean_r = X_r[:, i].mean()
        log2fc = np.log2((mean_t + 1e-4) / (mean_r + 1e-4))
        results.append({
            'gene': gene, 'log2FC': log2fc, 'pval': pval,
            'pct_target': pct_target[gene_mask][i],
            'mean_target': mean_t, 'mean_rest': mean_r,
        })

    df = pd.DataFrame(results).sort_values('log2FC', ascending=False)
    return df[df['log2FC'] >= min_fc].head(n_genes)

ctl = adata[adata.obs['condition'] == 'control'].copy()

print("Computing Kupffer cell markers (one-vs-rest in control)...")
kup_markers = marker_genes_one_vs_rest(ctl, 'Kupffer_cell', n_genes=100, min_fc=1.0, min_pct=0.15)
print("Computing Hepatocyte markers (one-vs-rest in control)...")
hep_markers = marker_genes_one_vs_rest(ctl, 'Hepatocyte',   n_genes=100, min_fc=1.0, min_pct=0.20)

kup_markers.to_csv(os.path.join(TAB_DIR, 'kupffer_signature_genes.csv'), index=False)
hep_markers.to_csv(os.path.join(TAB_DIR, 'hepatocyte_signature_genes.csv'), index=False)
print(f"Kupffer signature: {len(kup_markers)} genes | Hepatocyte signature: {len(hep_markers)} genes")

kup_sig = kup_markers['gene'].tolist()
hep_sig = hep_markers['gene'].tolist()

# ══════════════════════════════════════════════════════════════════════
# STEP 2: 在肝细胞核上打分（score_genes = 改良 Seurat AddModuleScore）
#
# 算法说明：
#   对每个核，计算目标基因集的平均表达量，减去等量随机对照基因
#   （从与目标基因表达水平相近的 bin 中随机抽取）的平均表达量。
#   这个"对照减法"消除了细胞整体测序深度对得分的干扰，使得分可
#   在不同核之间直接比较。正得分 = 目标基因高于背景期望。
# ══════════════════════════════════════════════════════════════════════
hep = adata[adata.obs['cell_type_broad'] == 'Hepatocyte'].copy()

valid_kup = [g for g in kup_sig if g in hep.raw.var_names]
valid_hep = [g for g in hep_sig if g in hep.raw.var_names]
print(f"\nValid genes for scoring — Kupffer: {len(valid_kup)}, Hepatocyte: {len(valid_hep)}")

sc.tl.score_genes(hep, valid_kup, score_name='KupfferScore', use_raw=True)
sc.tl.score_genes(hep, valid_hep, score_name='HepatocyteScore', use_raw=True)
hep.obs['IdentityRatio'] = hep.obs['KupfferScore'] - hep.obs['HepatocyteScore']

# ── 统计检验 ──────────────────────────────────────────────────────
print("\n=== 肝细胞身份偏移统计 ===")
results_rows = []
for score in ['KupfferScore', 'HepatocyteScore', 'IdentityRatio']:
    ctl_v = hep.obs[score][hep.obs['condition'] == 'control']
    dpi_v = hep.obs[score][hep.obs['condition'] == '7dpi']
    _, p  = stats.mannwhitneyu(dpi_v, ctl_v, alternative='two-sided')
    delta = dpi_v.mean() - ctl_v.mean()
    print(f"  {score}:")
    print(f"    control={ctl_v.mean():.4f}±{ctl_v.std():.4f}  "
          f"7dpi={dpi_v.mean():.4f}±{dpi_v.std():.4f}  Δ={delta:+.4f}  p={p:.2e}")
    results_rows.append({'score': score, 'control_mean': ctl_v.mean(),
                         '7dpi_mean': dpi_v.mean(), 'delta': delta, 'pval': p})

pd.DataFrame(results_rows).to_csv(os.path.join(TAB_DIR, 'hepatocyte_identity_shift_stats.csv'), index=False)

# ── 可视化：Violin + Histogram ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = {'KupfferScore': '#e74c3c', 'HepatocyteScore': '#2980b9', 'IdentityRatio': '#8e44ad'}

for ax, score in zip(axes, ['KupfferScore', 'HepatocyteScore', 'IdentityRatio']):
    ctl_v = hep.obs[score][hep.obs['condition'] == 'control'].values
    dpi_v = hep.obs[score][hep.obs['condition'] == '7dpi'].values
    parts = ax.violinplot([ctl_v, dpi_v], positions=[0, 1],
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(colors[score]); pc.set_alpha(0.6)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    _, p = stats.mannwhitneyu(dpi_v, ctl_v, alternative='two-sided')
    y_top = max(np.percentile(ctl_v, 99), np.percentile(dpi_v, 99)) * 1.15
    ax.plot([0, 1], [y_top, y_top], color='black', linewidth=1)
    ax.text(0.5, y_top * 1.02, f'p={p:.1e}', ha='center', fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Control', '7dpi'])
    ax.set_title(score); ax.set_ylabel('Score')
    ax.axhline(0, color='grey', linewidth=0.6, linestyle='--')

plt.suptitle('Hepatocyte identity shift toward Kupffer cell phenotype after TBI\n'
             '(score_genes: target gene set mean − matched random control mean)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'hepatocyte_identity_shift_violin.png'), dpi=150, bbox_inches='tight')
plt.close()

# 直方图
fig, ax = plt.subplots(figsize=(8, 4))
ctl_r = hep.obs['IdentityRatio'][hep.obs['condition'] == 'control']
dpi_r = hep.obs['IdentityRatio'][hep.obs['condition'] == '7dpi']
ax.hist(ctl_r, bins=100, alpha=0.5, color='steelblue', label=f'Control (n={len(ctl_r)})', density=True)
ax.hist(dpi_r, bins=100, alpha=0.5, color='coral',     label=f'7dpi (n={len(dpi_r)})',    density=True)
ax.axvline(ctl_r.mean(), color='steelblue', linestyle='--', linewidth=1.5, label=f'ctl mean={ctl_r.mean():.3f}')
ax.axvline(dpi_r.mean(), color='coral',     linestyle='--', linewidth=1.5, label=f'7dpi mean={dpi_r.mean():.3f}')
ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)
_, p_r = stats.mannwhitneyu(dpi_r, ctl_r, alternative='two-sided')
ax.set_xlabel('IdentityRatio = KupfferScore − HepatocyteScore\n正值→更像Kupffer；负值→更像Hepatocyte')
ax.set_ylabel('Density')
ax.set_title(f'Hepatocyte identity shift distribution  (p={p_r:.2e})')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'hepatocyte_identity_ratio_histogram.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Kupffer marker 基因在肝细胞 DE 中的表现
# ══════════════════════════════════════════════════════════════════════
hep_de = pd.read_csv(os.path.join(TAB_DIR, 'DE_Hepatocyte_7dpi_vs_control.csv'))
kup_in_hep = hep_de[hep_de['gene'].isin(kup_sig)].sort_values('log2FC', ascending=False)
kup_in_hep.to_csv(os.path.join(TAB_DIR, 'kupffer_markers_in_hepatocyte_DE.csv'), index=False)

print(f"\n全部 {len(kup_in_hep)} 个 Kupffer marker 基因在肝细胞 DE 中均上调")
print(f"其中 FDR<0.1 且 log2FC>0.3: {len(kup_in_hep[(kup_in_hep['FDR']<0.1)&(kup_in_hep['log2FC']>0.3)])}")

# ── 关键功能基因检查 ──────────────────────────────────────────────
mac_genes = ['Adgre1','Mrc1','Timd4','Csf1r','Msr1','Clec4f','Cd163','Spi1',
             'Cd36','Cybb','Sirpa','Hk2','Hk3','Pla2g4a','Trem2','Lgals3']
mac_in_de = hep_de[hep_de['gene'].isin(mac_genes)].sort_values('log2FC', ascending=False)
mac_in_de.to_csv(os.path.join(TAB_DIR, 'macrophage_functional_genes_in_hepatocytes.csv'), index=False)
print("\n关键巨噬细胞功能基因在肝细胞中的变化：")
print(mac_in_de[['gene','log2FC','FDR','significant']].to_string(index=False))

# ── 热图 ────────────────────────────────────────────────────────
top_genes = kup_in_hep[kup_in_hep['log2FC'] > 0].head(25)['gene'].tolist()
top_genes  = [g for g in top_genes if g in hep.raw.var_names]
if top_genes:
    sc.pl.heatmap(hep, top_genes, groupby='condition', use_raw=True,
                  standard_scale='var', cmap='RdBu_r', show=False,
                  figsize=(14, max(5, len(top_genes) * 0.32)))
    plt.suptitle('Top Kupffer marker genes expression in Hepatocytes: control vs 7dpi', y=1.01)
    plt.savefig(os.path.join(FIG_DIR, 'hepatocyte_kupffer_markers_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

# ── barplot: top Kupffer 基因在肝细胞中的 log2FC ──────────────
top30 = kup_in_hep.head(30)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(top30)), top30['log2FC'].values,
        color=['#e74c3c' if v > 0 else '#2980b9' for v in top30['log2FC']], alpha=0.8)
ax.set_yticks(range(len(top30))); ax.set_yticklabels(top30['gene'].tolist(), fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('log2FC in Hepatocytes (7dpi / control)')
ax.set_title('Kupffer cell signature genes upregulated in Hepatocytes at 7dpi TBI')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'kupffer_markers_in_hepatocyte_barplot.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# ── UMAP 叠加得分 ──────────────────────────────────────────────
for score in ['KupfferScore', 'HepatocyteScore', 'IdentityRatio']:
    sc.pl.umap(hep, color=score, show=False, cmap='RdBu_r', vcenter=0,
               title=f'Hepatocytes: {score}')
    plt.savefig(os.path.join(FIG_DIR, f'umap_hepatocyte_{score}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

print("\n=== 肝细胞身份偏移分析完成 ===")
