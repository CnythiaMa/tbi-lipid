#!/usr/bin/env python3
"""
GSE163691 Analysis Part 2: D7-peak gene functional annotation
and comprehensive hypothesis-testing analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/volume/data/xyma/jsonl/training/tbi/data/GSE163691")
RESULTS_DIR = Path("/volume/data/xyma/jsonl/training/tbi/results/GSE163691")

plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans'
})

# Load data
tpm = pd.read_csv(DATA_DIR / "GSE163691_TPM.txt.gz", sep='\t', index_col=0)
diff_d7 = pd.read_csv(DATA_DIR / "GSE163691_D7_injury_sham-Diff.txt.gz", sep='\t')
id_to_symbol = dict(zip(diff_d7['ID'], diff_d7['symbol']))
tpm['symbol'] = tpm.index.map(id_to_symbol)
tpm = tpm.dropna(subset=['symbol'])
tpm['mean_expr'] = tpm.drop(columns=['symbol']).mean(axis=1)
tpm = tpm.sort_values('mean_expr', ascending=False).drop_duplicates(subset='symbol')
tpm = tpm.set_index('symbol').drop(columns=['mean_expr'])

timepoints = ['D1', 'D4', 'D7', 'D14']
diff_results = {}
for tp in timepoints:
    df = pd.read_csv(DATA_DIR / f"GSE163691_{tp}_injury_sham-Diff.txt.gz", sep='\t')
    df = df.dropna(subset=['symbol']).drop_duplicates(subset='symbol')
    df = df.set_index('symbol')
    diff_results[tp] = df

# Also load sex-specific
sex_diff = {}
for sex in ['Male', 'Female']:
    sex_diff[sex] = {}
    for tp in timepoints:
        df = pd.read_csv(DATA_DIR / f"GSE163691_{sex}_{tp}_injury_sham-Diff.txt.gz", sep='\t')
        df = df.dropna(subset=['symbol']).drop_duplicates(subset='symbol')
        df = df.set_index('symbol')
        sex_diff[sex][tp] = df

# =============================================================================
# 1. Comprehensive temporal pattern classification
# =============================================================================
print("Classifying temporal expression patterns...")

# For each gene with significant change at any timepoint, classify its pattern
all_genes_sig = set()
for tp in timepoints:
    sig = diff_results[tp][(diff_results[tp]['padj'] < 0.05) &
                           (abs(diff_results[tp]['log2FoldChange']) > 0.5)]
    all_genes_sig.update(sig.index)

print(f"Total genes significantly changed at any timepoint: {len(all_genes_sig)}")

patterns = []
for gene in all_genes_sig:
    fcs = []
    padjs = []
    for tp in timepoints:
        if gene in diff_results[tp].index:
            fcs.append(diff_results[tp].loc[gene, 'log2FoldChange'])
            padjs.append(diff_results[tp].loc[gene, 'padj'])
        else:
            fcs.append(0)
            padjs.append(1)

    # Classify pattern
    peak_tp = timepoints[np.argmax([abs(f) for f in fcs])]
    is_up = fcs[np.argmax([abs(f) for f in fcs])] > 0

    # Special patterns
    d7_sig = padjs[2] < 0.05 and abs(fcs[2]) > 0.5
    d14_sig = padjs[3] < 0.05 and abs(fcs[3]) > 0.5

    # D7 peak + D14 resolve
    d7_peak_resolve = (d7_sig and fcs[2] > 1 and
                       (not d14_sig or abs(fcs[3]) < abs(fcs[2]) * 0.5))

    # Sustained up (D1-D14)
    sustained_up = all(p < 0.05 and f > 0.5 for p, f in zip(padjs, fcs))

    # Late response (D14 specific)
    late_response = (padjs[3] < 0.05 and abs(fcs[3]) > 1 and
                     all(padjs[i] > 0.05 or abs(fcs[i]) < 0.5 for i in [0, 1]))

    # Acute only (D1 peak, resolves)
    acute_only = (padjs[0] < 0.05 and abs(fcs[0]) > 1 and
                  (padjs[2] > 0.05 or abs(fcs[2]) < abs(fcs[0]) * 0.3))

    pattern = 'other'
    if d7_peak_resolve:
        pattern = 'D7_peak_resolve'
    elif sustained_up:
        pattern = 'sustained_up'
    elif late_response:
        pattern = 'late_D14'
    elif acute_only:
        pattern = 'acute_D1'
    elif peak_tp == 'D4' and is_up:
        pattern = 'subacute_D4'

    patterns.append({
        'gene': gene,
        'D1_fc': fcs[0], 'D4_fc': fcs[1], 'D7_fc': fcs[2], 'D14_fc': fcs[3],
        'D1_padj': padjs[0], 'D4_padj': padjs[1], 'D7_padj': padjs[2], 'D14_padj': padjs[3],
        'peak_timepoint': peak_tp,
        'direction': 'up' if is_up else 'down',
        'pattern': pattern,
    })

pattern_df = pd.DataFrame(patterns)
pattern_counts = pattern_df['pattern'].value_counts()
print("\nTemporal pattern classification:")
for p, c in pattern_counts.items():
    print(f"  {p}: {c} genes")

pattern_df.to_csv(RESULTS_DIR / 'temporal_pattern_classification.csv', index=False)

# =============================================================================
# 2. Curated lipid/LDAM gene deep analysis
# =============================================================================
print("\n\nDeep analysis of lipid-related genes...")

# Comprehensive lipid gene list
LIPID_COMPREHENSIVE = {
    # LD formation
    'LD_formation': ['Plin2', 'Plin3', 'Plin4', 'Plin5', 'Dgat1', 'Dgat2',
                     'Soat1', 'Acat1', 'Acat2', 'Fasn', 'Acaca'],
    # FA uptake & transport
    'FA_uptake': ['Cd36', 'Fabp5', 'Fabp4', 'Fabp7', 'Fatp1', 'Slc27a1',
                  'Slc27a4', 'Lpl', 'Vldlr', 'Ldlr', 'Lrp1'],
    # Lipoproteins
    'Lipoprotein': ['Apoe', 'Apoc1', 'Apob', 'Abca1', 'Abcg1'],
    # Lipolysis / LD clearance
    'Lipolysis': ['Lipa', 'Mgll', 'Pnpla2', 'Abhd5', 'Ces1d', 'Nceh1'],
    # FA oxidation
    'FA_oxidation': ['Cpt1a', 'Cpt2', 'Acox1', 'Acadl', 'Acadm', 'Acads',
                     'Ppara', 'Pparg', 'Ppargc1a'],
    # Cholesterol
    'Cholesterol': ['Hmgcr', 'Hmgcs1', 'Cyp46a1', 'Nr1h3', 'Nr1h2', 'Abca1', 'Abcg1'],
    # DHA/PUFA related
    'DHA_PUFA': ['Mfsd2a', 'Elovl2', 'Elovl4', 'Elovl5', 'Fads1', 'Fads2',
                 'Pla2g4a', 'Alox5', 'Alox12', 'Alox15', 'Ptgs1', 'Ptgs2'],
    # Sphingolipid
    'Sphingolipid': ['Smpd1', 'Smpd3', 'Asah1', 'Cers2', 'Cers6', 'Sphk1', 'Sgpl1'],
    # Myelin lipids
    'Myelin_lipid': ['Plp1', 'Mbp', 'Mag', 'Mog', 'Cnp', 'Ugt8a'],
    # LD regulators
    'LD_regulators': ['Grn', 'Trem2', 'Hdac3', 'Tfeb', 'Atg7', 'Becn1'],
}

print("\nDetailed log2FC for all lipid-related genes:")
print(f"{'Category':<18} {'Gene':<10} {'D1':>8} {'D4':>8} {'D7':>8} {'D14':>8} {'Pattern'}")
print("-" * 80)

lipid_detail_rows = []
for category, genes in LIPID_COMPREHENSIVE.items():
    for gene in genes:
        if gene not in tpm.index:
            continue
        fcs = []
        sigs = []
        for tp in timepoints:
            if gene in diff_results[tp].index:
                fc = diff_results[tp].loc[gene, 'log2FoldChange']
                padj = diff_results[tp].loc[gene, 'padj']
                fcs.append(fc)
                sig = '***' if padj < 0.001 else ('**' if padj < 0.01 else ('*' if padj < 0.05 else ''))
                sigs.append(sig)
            else:
                fcs.append(np.nan)
                sigs.append('')

        fc_strs = [f"{f:+.1f}{s}" if not np.isnan(f) else "NA" for f, s in zip(fcs, sigs)]

        # Determine pattern
        if gene in pattern_df['gene'].values:
            pat = pattern_df[pattern_df['gene'] == gene]['pattern'].values[0]
        else:
            pat = 'not_sig'

        print(f"  {category:<16} {gene:<10} {fc_strs[0]:>8} {fc_strs[1]:>8} {fc_strs[2]:>8} {fc_strs[3]:>8}  {pat}")

        lipid_detail_rows.append({
            'category': category, 'gene': gene,
            'D1_fc': fcs[0], 'D4_fc': fcs[1], 'D7_fc': fcs[2], 'D14_fc': fcs[3],
            'pattern': pat
        })

lipid_detail_df = pd.DataFrame(lipid_detail_rows)
lipid_detail_df.to_csv(RESULTS_DIR / 'lipid_genes_detailed.csv', index=False)

# =============================================================================
# 3. Figure 12: Comprehensive lipid gene heatmap (categorized)
# =============================================================================
print("\nGenerating Figure 12: Categorized lipid gene heatmap...")

# Build FC matrix with category labels
all_lipid_genes = []
category_labels = []
for cat, genes in LIPID_COMPREHENSIVE.items():
    for g in genes:
        if g in tpm.index and g not in all_lipid_genes:
            all_lipid_genes.append(g)
            category_labels.append(cat)

fc_mat = pd.DataFrame(index=all_lipid_genes, columns=timepoints, dtype=float)
for gene in all_lipid_genes:
    for tp in timepoints:
        if gene in diff_results[tp].index:
            fc_mat.loc[gene, tp] = diff_results[tp].loc[gene, 'log2FoldChange']

# Create figure with category color bar
fig, (ax_cat, ax_heat) = plt.subplots(1, 2, figsize=(14, max(8, len(all_lipid_genes) * 0.28)),
                                        gridspec_kw={'width_ratios': [0.5, 10]})

# Category colors
cat_colors = {
    'LD_formation': '#E53935', 'FA_uptake': '#FB8C00', 'Lipoprotein': '#FFD600',
    'Lipolysis': '#43A047', 'FA_oxidation': '#00ACC1', 'Cholesterol': '#1E88E5',
    'DHA_PUFA': '#8E24AA', 'Sphingolipid': '#6D4C41', 'Myelin_lipid': '#78909C',
    'LD_regulators': '#D81B60'
}

# Category color bar
for i, (gene, cat) in enumerate(zip(all_lipid_genes, category_labels)):
    ax_cat.barh(i, 1, color=cat_colors.get(cat, 'gray'), height=0.9)
ax_cat.set_yticks(range(len(all_lipid_genes)))
ax_cat.set_yticklabels(all_lipid_genes, fontsize=8)
ax_cat.set_xlim(0, 1)
ax_cat.set_xticks([])
ax_cat.invert_yaxis()
ax_cat.set_title('Category', fontsize=9)

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=k.replace('_', ' '))
                   for k, c in cat_colors.items() if k in set(category_labels)]
ax_cat.legend(handles=legend_elements, loc='lower left', fontsize=6,
              bbox_to_anchor=(-0.5, -0.15), ncol=2)

# Heatmap
vmax = 6
sns.heatmap(fc_mat.values.astype(float), cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
            annot=True, fmt='.1f', linewidths=0.3, ax=ax_heat,
            xticklabels=timepoints, yticklabels=False,
            cbar_kws={'label': 'log₂FC (Injury / Sham)', 'shrink': 0.5})
ax_heat.set_xlabel('Time Post-Injury')

plt.suptitle('GSE163691: Comprehensive Lipid Gene Expression in Microglia\nAfter Cortical Stab Wound (D1-D14)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig12_lipid_comprehensive_heatmap.png', bbox_inches='tight')
plt.close()
print("  Saved: fig12_lipid_comprehensive_heatmap.png")

# =============================================================================
# 4. Figure 13: LD formation vs clearance balance
# =============================================================================
print("\nGenerating Figure 13: LD formation vs clearance balance...")

formation_genes = ['Plin2', 'Plin3', 'Cd36', 'Lpl', 'Fabp5', 'Apoe', 'Soat1', 'Fasn', 'Dgat1']
clearance_genes = ['Lipa', 'Mgll', 'Pnpla2', 'Abhd5', 'Grn', 'Tfeb', 'Atg7', 'Becn1']

day_nums = [1, 4, 7, 14]

def avg_fc(genes):
    available = [g for g in genes if g in tpm.index]
    fcs = []
    for tp in timepoints:
        fc_vals = []
        for g in available:
            if g in diff_results[tp].index:
                fc_vals.append(diff_results[tp].loc[g, 'log2FoldChange'])
        fcs.append(np.mean(fc_vals) if fc_vals else 0)
    return fcs

form_fcs = avg_fc(formation_genes)
clear_fcs = avg_fc(clearance_genes)
balance = [f - c for f, c in zip(form_fcs, clear_fcs)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Formation vs Clearance
ax1.plot(day_nums, form_fcs, 'ro-', linewidth=2.5, markersize=10, label='LD Formation genes')
ax1.plot(day_nums, clear_fcs, 'bs-', linewidth=2.5, markersize=10, label='LD Clearance genes')
ax1.fill_between(day_nums, form_fcs, clear_fcs, alpha=0.15, color='red',
                  where=[f > c for f, c in zip(form_fcs, clear_fcs)])
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.axvline(x=7, color='red', linestyle='--', alpha=0.4, linewidth=2)
ax1.set_xlabel('Days Post-Injury', fontsize=12)
ax1.set_ylabel('Mean log₂FC', fontsize=12)
ax1.set_title('LD Formation vs Clearance Gene Expression', fontsize=12, fontweight='bold')
ax1.set_xticks(day_nums)
ax1.set_xticklabels(['D1', 'D4', 'D7', 'D14'])
ax1.legend(fontsize=10)
ax1.annotate('LD peak\n(observed)', xy=(7, max(form_fcs)), fontsize=9, color='red', ha='center')

# Panel B: Balance (Formation - Clearance)
colors = ['#E53935' if b > 0 else '#1E88E5' for b in balance]
ax2.bar(day_nums, balance, width=2, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_xlabel('Days Post-Injury', fontsize=12)
ax2.set_ylabel('Formation - Clearance (mean log₂FC)', fontsize=12)
ax2.set_title('Net LD Accumulation Tendency', fontsize=12, fontweight='bold')
ax2.set_xticks(day_nums)
ax2.set_xticklabels(['D1', 'D4', 'D7', 'D14'])
for i, (d, b) in enumerate(zip(day_nums, balance)):
    ax2.annotate(f'{b:.2f}', xy=(d, b), ha='center',
                 va='bottom' if b > 0 else 'top', fontweight='bold')

plt.suptitle('GSE163691: Lipid Droplet Dynamics — Transcriptomic Evidence\n'
             '(FACS-sorted Microglia, Cortical Stab Wound)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig13_ld_balance.png', bbox_inches='tight')
plt.close()
print("  Saved: fig13_ld_balance.png")

# =============================================================================
# 5. Figure 14: Hypothesis testing - DHA/PUFA and transport genes
# =============================================================================
print("\nGenerating Figure 14: DHA/PUFA and transport pathway...")

dha_genes = ['Mfsd2a', 'Elovl2', 'Elovl5', 'Fads1', 'Fads2', 'Alox5', 'Alox15', 'Ptgs2',
             'Pla2g4a', 'Pla2g7', 'Lpcat1', 'Lpcat2']
dha_genes_available = [g for g in dha_genes if g in tpm.index]

fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes = axes.flatten()

for idx, gene in enumerate(dha_genes_available[:12]):
    ax = axes[idx]
    for sex, color in [('M', '#2196F3'), ('F', '#E91E63')]:
        inj_means = []
        for tp in timepoints:
            inj_cols = [c for c in tpm.columns if c.startswith(f"{tp}_{sex}_injury")]
            sham_cols = [c for c in tpm.columns if c.startswith(f"{tp}_{sex}_sham")]
            inj_m = tpm.loc[gene, inj_cols].mean()
            sham_m = tpm.loc[gene, sham_cols].mean()
            inj_means.append(np.log2((inj_m + 1) / (sham_m + 1)))
        ax.plot(day_nums, inj_means, marker='o', color=color, linewidth=1.5,
                label='Male' if sex == 'M' else 'Female')

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=7, color='red', linestyle='--', alpha=0.3)
    ax.set_title(gene, fontweight='bold')
    ax.set_xticks(day_nums)
    ax.set_xticklabels(['D1', 'D4', 'D7', 'D14'])
    if idx == 0:
        ax.legend(fontsize=7)

# Hide unused axes
for idx in range(len(dha_genes_available), 12):
    axes[idx].set_visible(False)

plt.suptitle('GSE163691: DHA/PUFA Metabolism & Transport Genes in Microglia\n(log₂FC Injury vs Sham, by sex)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig14_dha_pufa_genes.png', bbox_inches='tight')
plt.close()
print("  Saved: fig14_dha_pufa_genes.png")

# =============================================================================
# 6. Figure 15: IL-6 axis deep dive with sex differences
# =============================================================================
print("\nGenerating Figure 15: IL-6 axis + acute phase response...")

il6_extended = ['Il6', 'Il6ra', 'Il6st', 'Jak1', 'Jak2', 'Stat3', 'Socs3',
                'Lcn2', 'Serpina3n', 'Hp', 'A2m', 'Saa3', 'Orm1']
il6_available = [g for g in il6_extended if g in tpm.index]

n_genes = len(il6_available)
ncols = 4
nrows = (n_genes + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
axes = axes.flatten()

for idx, gene in enumerate(il6_available):
    ax = axes[idx]

    # Injury TPM (combined sex)
    inj_means, inj_sems = [], []
    sham_means, sham_sems = [], []
    for tp in timepoints:
        inj_cols = [c for c in tpm.columns if c.startswith(f"{tp}_") and '_injury_' in c]
        sham_cols = [c for c in tpm.columns if c.startswith(f"{tp}_") and '_sham_' in c]
        iv = tpm.loc[gene, inj_cols].values.astype(float)
        sv = tpm.loc[gene, sham_cols].values.astype(float)
        inj_means.append(iv.mean())
        inj_sems.append(iv.std() / np.sqrt(len(iv)))
        sham_means.append(sv.mean())
        sham_sems.append(sv.std() / np.sqrt(len(sv)))

    ax.errorbar(day_nums, inj_means, yerr=inj_sems, marker='o', color='#E53935',
                linewidth=2, capsize=3, label='Injury')
    ax.errorbar(day_nums, sham_means, yerr=sham_sems, marker='s', color='#1E88E5',
                linewidth=2, capsize=3, label='Sham')

    # Significance stars
    for i, tp in enumerate(timepoints):
        if gene in diff_results[tp].index:
            padj = diff_results[tp].loc[gene, 'padj']
            if padj < 0.05:
                star = '***' if padj < 0.001 else ('**' if padj < 0.01 else '*')
                y_pos = max(inj_means[i], sham_means[i]) * 1.05
                ax.text(day_nums[i], y_pos, star, ha='center', fontweight='bold', fontsize=9)

    ax.set_title(gene, fontweight='bold', fontsize=11)
    ax.set_xticks(day_nums)
    ax.set_xticklabels(['D1', 'D4', 'D7', 'D14'])
    if idx == 0:
        ax.legend(fontsize=7)
    ax.set_ylabel('TPM')

for idx in range(len(il6_available), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('GSE163691: IL-6 / Acute Phase Response Genes in Microglia\n(Injury vs Sham, mean ± SEM)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig15_il6_acute_phase.png', bbox_inches='tight')
plt.close()
print("  Saved: fig15_il6_acute_phase.png")

# =============================================================================
# 7. Summary statistics for hypothesis testing
# =============================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS-RELEVANT FINDINGS")
print("=" * 70)

print("\n1. LIPID DROPLET DYNAMICS (D7 peak, D14 resolve)")
print("-" * 50)
# Check if LD formation genes peak before/at D7
print("   LD Formation genes peaking pattern:")
for g in ['Plin2', 'Plin3', 'Cd36', 'Lpl', 'Fabp5', 'Soat1']:
    if g in tpm.index:
        fcs = []
        for tp in timepoints:
            if g in diff_results[tp].index:
                fcs.append(diff_results[tp].loc[g, 'log2FoldChange'])
            else:
                fcs.append(0)
        peak = timepoints[np.argmax(fcs)]
        print(f"   {g:10s}: peaks at {peak} (FC: {max(fcs):.2f})")

print("\n   LD Clearance genes (Mgll downregulated = impaired clearance):")
for g in ['Mgll', 'Pnpla2', 'Lipa', 'Grn']:
    if g in tpm.index:
        fcs = []
        for tp in timepoints:
            if g in diff_results[tp].index:
                fcs.append(diff_results[tp].loc[g, 'log2FoldChange'])
            else:
                fcs.append(0)
        print(f"   {g:10s}: D1={fcs[0]:+.2f}, D4={fcs[1]:+.2f}, D7={fcs[2]:+.2f}, D14={fcs[3]:+.2f}")

print("\n2. IL-6 SIGNALING IN MICROGLIA")
print("-" * 50)
for g in ['Il6', 'Il6ra', 'Il6st', 'Stat3', 'Socs3']:
    if g in tpm.index:
        fcs = []
        for tp in timepoints:
            if g in diff_results[tp].index:
                fcs.append(diff_results[tp].loc[g, 'log2FoldChange'])
            else:
                fcs.append(0)
        print(f"   {g:10s}: D1={fcs[0]:+.2f}, D4={fcs[1]:+.2f}, D7={fcs[2]:+.2f}, D14={fcs[3]:+.2f}")

print("\n3. DHA/MFSD2A TRANSPORT")
print("-" * 50)
for g in ['Mfsd2a', 'Fabp7', 'Elovl2', 'Fads1', 'Fads2']:
    if g in tpm.index:
        fcs = []
        for tp in timepoints:
            if g in diff_results[tp].index:
                fcs.append(diff_results[tp].loc[g, 'log2FoldChange'])
            else:
                fcs.append(0)
        print(f"   {g:10s}: D1={fcs[0]:+.2f}, D4={fcs[1]:+.2f}, D7={fcs[2]:+.2f}, D14={fcs[3]:+.2f}")

print("\n4. MYELIN PHAGOCYTOSIS MARKERS (local lipid source)")
print("-" * 50)
phago_genes = ['Trem2', 'Tyrobp', 'Mertk', 'Axl', 'Gas6', 'Megf10', 'Cd68']
for g in phago_genes:
    if g in tpm.index:
        fcs = []
        for tp in timepoints:
            if g in diff_results[tp].index:
                fcs.append(diff_results[tp].loc[g, 'log2FoldChange'])
            else:
                fcs.append(0)
        print(f"   {g:10s}: D1={fcs[0]:+.2f}, D4={fcs[1]:+.2f}, D7={fcs[2]:+.2f}, D14={fcs[3]:+.2f}")

print("\n5. ACUTE PHASE PROTEINS (liver axis markers expressed in microglia)")
print("-" * 50)
for g in ['Lcn2', 'Serpina3n', 'A2m', 'Saa3', 'Orm1', 'Hp']:
    if g in tpm.index:
        fcs = []
        padjs_list = []
        for tp in timepoints:
            if g in diff_results[tp].index:
                fcs.append(diff_results[tp].loc[g, 'log2FoldChange'])
                padjs_list.append(diff_results[tp].loc[g, 'padj'])
            else:
                fcs.append(0)
                padjs_list.append(1)
        sig_str = ' | '.join([f"D{d}:{f:+.1f}{'*' if p<0.05 else ''}"
                              for d, f, p in zip([1,4,7,14], fcs, padjs_list)])
        print(f"   {g:12s}: {sig_str}")

print("\n" + "=" * 70)
print("DONE. All results saved to:", RESULTS_DIR)
