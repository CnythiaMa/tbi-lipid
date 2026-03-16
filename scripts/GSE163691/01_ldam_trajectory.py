#!/usr/bin/env python3
"""
GSE163691 Analysis: LDAM/DAM Gene Trajectory in Microglia After Cortical Stab Wound
Time points: D1, D4, D7, D14 (FACS-sorted microglia bulk RNA-seq)

Key question: Do LDAM signature genes peak at D7 and resolve by D14,
matching the observed lipid droplet dynamics?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/volume/data/xyma/jsonl/training/tbi/data/GSE163691")
RESULTS_DIR = Path("/volume/data/xyma/jsonl/training/tbi/results/GSE163691")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans'
})

# =============================================================================
# 1. Load Data
# =============================================================================
print("Loading TPM matrix...")
tpm = pd.read_csv(DATA_DIR / "GSE163691_TPM.txt.gz", sep='\t', index_col=0)
print(f"  TPM matrix: {tpm.shape[0]} genes × {tpm.shape[1]} samples")

# We need gene symbols. Load one Diff file to get Ensembl -> Symbol mapping
print("Building gene ID mapping...")
diff_d7 = pd.read_csv(DATA_DIR / "GSE163691_D7_injury_sham-Diff.txt.gz", sep='\t')
id_to_symbol = dict(zip(diff_d7['ID'], diff_d7['symbol']))

# Map Ensembl IDs to symbols
tpm['symbol'] = tpm.index.map(id_to_symbol)
tpm = tpm.dropna(subset=['symbol'])
# Handle duplicates: keep highest expressed
tpm['mean_expr'] = tpm.drop(columns=['symbol']).mean(axis=1)
tpm = tpm.sort_values('mean_expr', ascending=False).drop_duplicates(subset='symbol')
tpm = tpm.set_index('symbol').drop(columns=['mean_expr'])
print(f"  After symbol mapping: {tpm.shape[0]} genes")

# =============================================================================
# 2. Define Gene Sets
# =============================================================================

# LDAM signature (Marschallinger et al., 2020 Nat Neurosci)
LDAM_GENES = [
    # Lipid metabolism / storage
    'Lpl', 'Apoe', 'Cd36', 'Fabp5', 'Fabp4', 'Lipa', 'Plin2', 'Plin3',
    'Acsl1', 'Acsl3', 'Acsl5', 'Dgat1', 'Dgat2', 'Soat1',
    # Lipid droplet / neutral lipid
    'Acat1', 'Acat2', 'Abhd5', 'Mgll', 'Pnpla2',
    # Pro-inflammatory (LDAM-associated)
    'Nos2', 'Ptgs2', 'Il1b', 'Tnf', 'Cxcl10', 'Ccl2',
]

# DAM signature (Keren-Shaul et al., 2017 Cell)
DAM_GENES = [
    'Trem2', 'Tyrobp', 'Cst7', 'Spp1', 'Itgax', 'Axl', 'Lgals3',
    'Clec7a', 'Cd9', 'Csf1', 'Lyz2', 'Ctsl', 'Ctsb', 'Ctsd',
    'Gpnmb', 'Igf1', 'Lilrb4a',
]

# Homeostatic microglia markers (downregulated in reactive states)
HOMEOSTATIC_GENES = [
    'P2ry12', 'Tmem119', 'Cx3cr1', 'Siglech', 'Hexb',
    'Csf1r', 'Fcrls', 'Gpr34', 'Selplg',
]

# Lipid metabolism pathway genes (broader)
LIPID_METABOLISM = [
    # Fatty acid synthesis
    'Fasn', 'Acaca', 'Scd1', 'Scd2', 'Elovl1', 'Fads1', 'Fads2',
    # Fatty acid oxidation
    'Cpt1a', 'Acox1', 'Acadl', 'Acadm',
    # Cholesterol
    'Hmgcr', 'Hmgcs1', 'Ldlr', 'Abca1', 'Abcg1', 'Nr1h3', 'Cyp46a1',
    # Phospholipid
    'Pla2g4a', 'Pla2g7', 'Lpcat1', 'Lpcat2',
    # Sphingolipid
    'Smpd1', 'Asah1', 'Sphk1', 'Cers6',
    # DHA-related
    'Mfsd2a', 'Fabp7',
]

# Autophagy / lysosomal (LD clearance)
AUTOPHAGY_GENES = [
    'Atg5', 'Atg7', 'Becn1', 'Map1lc3a', 'Map1lc3b', 'Sqstm1',
    'Tfeb', 'Lamp1', 'Lamp2', 'Ctsd', 'Ctsl', 'Ctsb',
    'Grn', 'Hdac3',
]

# IL-6 signaling pathway
IL6_PATHWAY = [
    'Il6', 'Il6ra', 'Il6st', 'Jak1', 'Jak2', 'Stat3',
    'Socs1', 'Socs3', 'Nfkb1', 'Rela',
]

# TBI-liver axis related
LIVER_AXIS = [
    'Hp', 'Crp', 'Saa1', 'Saa3', 'Orm1', 'Fga', 'Fgb', 'Fgg',
    'Serpina1a', 'Serpina3n', 'A2m', 'Lcn2',
]

ALL_GENE_SETS = {
    'LDAM Signature': LDAM_GENES,
    'DAM Signature': DAM_GENES,
    'Homeostatic Microglia': HOMEOSTATIC_GENES,
    'Lipid Metabolism': LIPID_METABOLISM,
    'Autophagy/Lysosome': AUTOPHAGY_GENES,
    'IL-6 Pathway': IL6_PATHWAY,
    'Acute Phase/Liver Axis': LIVER_AXIS,
}

# =============================================================================
# 3. Parse sample metadata
# =============================================================================
metadata = []
for col in tpm.columns:
    parts = col.split('_')
    timepoint = parts[0]
    sex = parts[1]
    condition = parts[2]
    rep = parts[3]
    metadata.append({
        'sample': col, 'timepoint': timepoint,
        'sex': sex, 'condition': condition, 'rep': rep,
        'day_num': int(timepoint.replace('D', ''))
    })
meta_df = pd.DataFrame(metadata)

# =============================================================================
# 4. Compute log2FC (injury/sham) for each gene at each timepoint
# =============================================================================
print("\nComputing injury vs sham fold changes...")

timepoints = ['D1', 'D4', 'D7', 'D14']
day_nums = [1, 4, 7, 14]

def get_fc_profile(gene, sex='both'):
    """Get log2FC trajectory across timepoints for a gene."""
    if gene not in tpm.index:
        return [np.nan] * 4
    fcs = []
    for tp in timepoints:
        if sex == 'both':
            inj_cols = [c for c in tpm.columns if c.startswith(f"{tp}_") and '_injury_' in c]
            sham_cols = [c for c in tpm.columns if c.startswith(f"{tp}_") and '_sham_' in c]
        else:
            s = sex[0].upper()
            inj_cols = [c for c in tpm.columns if c.startswith(f"{tp}_{s}_injury")]
            sham_cols = [c for c in tpm.columns if c.startswith(f"{tp}_{s}_sham")]
        inj_mean = tpm.loc[gene, inj_cols].mean()
        sham_mean = tpm.loc[gene, sham_cols].mean()
        # log2FC with pseudocount
        fc = np.log2((inj_mean + 1) / (sham_mean + 1))
        fcs.append(fc)
    return fcs

# =============================================================================
# 5. Load official DESeq2 results for padj values
# =============================================================================
print("Loading DESeq2 differential expression results...")
diff_results = {}
for tp in timepoints:
    df = pd.read_csv(DATA_DIR / f"GSE163691_{tp}_injury_sham-Diff.txt.gz", sep='\t')
    df = df.dropna(subset=['symbol']).drop_duplicates(subset='symbol')
    df = df.set_index('symbol')
    diff_results[tp] = df

# =============================================================================
# 6. Figure 1: LDAM Gene Trajectory Heatmap
# =============================================================================
print("\nGenerating Figure 1: LDAM gene trajectory heatmap...")

def make_trajectory_heatmap(gene_list, title, filename, figsize=(12, None)):
    available = [g for g in gene_list if g in tpm.index]
    missing = [g for g in gene_list if g not in tpm.index]
    if missing:
        print(f"  Missing genes: {missing}")

    if not available:
        print(f"  No genes found for {title}")
        return None

    # Build FC matrix
    fc_matrix = pd.DataFrame(index=available, columns=timepoints)
    padj_matrix = pd.DataFrame(index=available, columns=timepoints)

    for gene in available:
        fcs = get_fc_profile(gene)
        fc_matrix.loc[gene] = fcs
        for tp in timepoints:
            if gene in diff_results[tp].index:
                padj_matrix.loc[gene, tp] = diff_results[tp].loc[gene, 'padj']
            else:
                padj_matrix.loc[gene, tp] = np.nan

    fc_matrix = fc_matrix.astype(float)
    padj_matrix = padj_matrix.astype(float)

    # Sort by D7 FC (descending)
    fc_matrix = fc_matrix.sort_values('D7', ascending=False)
    padj_matrix = padj_matrix.loc[fc_matrix.index]

    h = max(6, len(available) * 0.35)
    fig, ax = plt.subplots(figsize=(figsize[0], h))

    # Heatmap
    vmax = max(abs(fc_matrix.values.min()), abs(fc_matrix.values.max()), 2)
    sns.heatmap(fc_matrix, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt='.2f', linewidths=0.5, ax=ax,
                cbar_kws={'label': 'log₂FC (Injury / Sham)'})

    # Mark significant with stars
    for i, gene in enumerate(fc_matrix.index):
        for j, tp in enumerate(timepoints):
            pval = padj_matrix.loc[gene, tp]
            if pd.notna(pval) and pval < 0.05:
                marker = '***' if pval < 0.001 else ('**' if pval < 0.01 else '*')
                ax.text(j + 0.5, i + 0.85, marker, ha='center', va='center',
                        fontsize=7, color='black', fontweight='bold')

    ax.set_title(f'{title}\n(Injury vs Sham, FACS-sorted Microglia)', fontsize=12)
    ax.set_xlabel('Time Post-Injury')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    return fc_matrix

fc_ldam = make_trajectory_heatmap(LDAM_GENES, 'LDAM Signature Genes', 'fig1_ldam_heatmap.png')
fc_dam = make_trajectory_heatmap(DAM_GENES, 'DAM Signature Genes', 'fig2_dam_heatmap.png')
fc_homeo = make_trajectory_heatmap(HOMEOSTATIC_GENES, 'Homeostatic Microglia Markers', 'fig3_homeostatic_heatmap.png')
fc_lipid = make_trajectory_heatmap(LIPID_METABOLISM, 'Lipid Metabolism Genes', 'fig4_lipid_metabolism_heatmap.png')
fc_auto = make_trajectory_heatmap(AUTOPHAGY_GENES, 'Autophagy/Lysosome (LD Clearance)', 'fig5_autophagy_heatmap.png')
fc_il6 = make_trajectory_heatmap(IL6_PATHWAY, 'IL-6 Signaling Pathway', 'fig6_il6_pathway_heatmap.png')
fc_liver = make_trajectory_heatmap(LIVER_AXIS, 'Acute Phase / Liver Axis Genes', 'fig7_liver_axis_heatmap.png')

# =============================================================================
# 7. Figure 8: Key LDAM genes time-course line plots
# =============================================================================
print("\nGenerating Figure 8: Key gene time-course line plots...")

key_genes = {
    'Lipid Storage': ['Plin2', 'Plin3', 'Lpl', 'Cd36', 'Fabp5', 'Apoe'],
    'Lipid Synthesis': ['Fasn', 'Scd1', 'Scd2', 'Acsl1', 'Dgat1', 'Dgat2'],
    'LD Clearance': ['Grn', 'Lipa', 'Pnpla2', 'Mgll', 'Abhd5', 'Hdac3'],
    'Inflammation': ['Il1b', 'Tnf', 'Cxcl10', 'Ccl2', 'Ptgs2', 'Nos2'],
    'DAM Core': ['Trem2', 'Tyrobp', 'Spp1', 'Cst7', 'Itgax', 'Axl'],
    'IL-6 Axis': ['Il6', 'Il6ra', 'Il6st', 'Stat3', 'Socs3', 'Lcn2'],
}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for idx, (panel_name, genes) in enumerate(key_genes.items()):
    ax = axes[idx]
    for gene in genes:
        if gene not in tpm.index:
            continue
        # Plot injury mean ± SEM
        inj_means, inj_sems = [], []
        sham_means = []
        for tp in timepoints:
            inj_cols = [c for c in tpm.columns if c.startswith(f"{tp}_") and '_injury_' in c]
            sham_cols = [c for c in tpm.columns if c.startswith(f"{tp}_") and '_sham_' in c]
            inj_vals = tpm.loc[gene, inj_cols].values.astype(float)
            sham_vals = tpm.loc[gene, sham_cols].values.astype(float)
            inj_means.append(inj_vals.mean())
            inj_sems.append(inj_vals.std() / np.sqrt(len(inj_vals)))
            sham_means.append(sham_vals.mean())

        ax.errorbar(day_nums, inj_means, yerr=inj_sems,
                     marker='o', markersize=5, linewidth=1.5, capsize=3, label=gene)

    ax.set_xlabel('Days Post-Injury')
    ax.set_ylabel('TPM (Injury)')
    ax.set_title(panel_name, fontsize=12, fontweight='bold')
    ax.set_xticks(day_nums)
    ax.set_xticklabels(['D1', 'D4', 'D7', 'D14'])
    ax.legend(fontsize=8, loc='best')
    ax.axvline(x=7, color='red', linestyle='--', alpha=0.3, label='LD peak')

plt.suptitle('GSE163691: Gene Expression in FACS-sorted Microglia After Cortical Stab Wound\n(Injury group, mean ± SEM, n=4 per timepoint)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig8_key_genes_timecourse.png', bbox_inches='tight')
plt.close()
print("  Saved: fig8_key_genes_timecourse.png")

# =============================================================================
# 8. Figure 9: Sex differences in LDAM genes
# =============================================================================
print("\nGenerating Figure 9: Sex differences...")

sex_genes = ['Plin2', 'Lpl', 'Cd36', 'Apoe', 'Fabp5', 'Trem2', 'Spp1', 'Il6ra', 'Grn', 'Fasn', 'Scd2', 'Lcn2']
sex_genes = [g for g in sex_genes if g in tpm.index]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for idx, gene in enumerate(sex_genes):
    ax = axes[idx]
    for sex, color, marker in [('M', '#2196F3', 's'), ('F', '#E91E63', 'o')]:
        inj_means, inj_sems = [], []
        for tp in timepoints:
            inj_cols = [c for c in tpm.columns if c.startswith(f"{tp}_{sex}_injury")]
            sham_cols = [c for c in tpm.columns if c.startswith(f"{tp}_{sex}_sham")]
            inj_vals = tpm.loc[gene, inj_cols].values.astype(float)
            sham_vals = tpm.loc[gene, sham_cols].values.astype(float)
            # Show FC
            fc = np.log2((inj_vals.mean() + 1) / (sham_vals.mean() + 1))
            inj_means.append(fc)

        sex_label = 'Male' if sex == 'M' else 'Female'
        ax.plot(day_nums, inj_means, marker=marker, color=color,
                linewidth=1.5, markersize=6, label=sex_label)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=7, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Days Post-Injury')
    ax.set_ylabel('log₂FC')
    ax.set_title(gene, fontsize=11, fontweight='bold')
    ax.set_xticks(day_nums)
    ax.set_xticklabels(['D1', 'D4', 'D7', 'D14'])
    ax.legend(fontsize=8)

plt.suptitle('GSE163691: Sex Differences in LDAM/Key Genes\n(log₂FC Injury vs Sham)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig9_sex_differences.png', bbox_inches='tight')
plt.close()
print("  Saved: fig9_sex_differences.png")

# =============================================================================
# 9. Figure 10: Gene Set Scores over time
# =============================================================================
print("\nGenerating Figure 10: Gene set scores...")

def compute_geneset_score(gene_set, sex='both'):
    """Average z-score of genes in the set across samples."""
    available = [g for g in gene_set if g in tpm.index]
    if not available:
        return [np.nan] * 4
    scores = []
    for tp in timepoints:
        if sex == 'both':
            inj_cols = [c for c in tpm.columns if c.startswith(f"{tp}_") and '_injury_' in c]
            sham_cols = [c for c in tpm.columns if c.startswith(f"{tp}_") and '_sham_' in c]
        else:
            s = sex[0].upper()
            inj_cols = [c for c in tpm.columns if c.startswith(f"{tp}_{s}_injury")]
            sham_cols = [c for c in tpm.columns if c.startswith(f"{tp}_{s}_sham")]
        fcs = []
        for g in available:
            inj_m = tpm.loc[g, inj_cols].mean()
            sham_m = tpm.loc[g, sham_cols].mean()
            fcs.append(np.log2((inj_m + 1) / (sham_m + 1)))
        scores.append(np.mean(fcs))
    return scores

fig, ax = plt.subplots(figsize=(10, 7))

colors = ['#E53935', '#FB8C00', '#43A047', '#1E88E5', '#8E24AA', '#00ACC1', '#6D4C41']
for (name, genes), color in zip(ALL_GENE_SETS.items(), colors):
    scores = compute_geneset_score(genes)
    ax.plot(day_nums, scores, marker='o', linewidth=2.5, markersize=8,
            label=name, color=color)

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=7, color='red', linestyle='--', alpha=0.4, linewidth=2)
ax.annotate('LD peak (D7)', xy=(7, ax.get_ylim()[1]*0.9), fontsize=9, color='red', ha='center')
ax.set_xlabel('Days Post-Injury', fontsize=12)
ax.set_ylabel('Mean log₂FC (Gene Set Score)', fontsize=12)
ax.set_title('GSE163691: Gene Set Activation Dynamics in Microglia\nAfter Cortical Stab Wound', fontsize=13, fontweight='bold')
ax.set_xticks(day_nums)
ax.set_xticklabels(['D1', 'D4', 'D7', 'D14'])
ax.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig10_geneset_scores.png', bbox_inches='tight')
plt.close()
print("  Saved: fig10_geneset_scores.png")

# =============================================================================
# 10. Figure 11: D7-specific upregulated genes → pathway analysis
# =============================================================================
print("\nIdentifying D7-peak genes...")

# Find genes that peak at D7 (highest FC at D7, significantly upregulated)
d7_diff = diff_results['D7']
d14_diff = diff_results['D14']

# D7 sig up
d7_up = d7_diff[(d7_diff['padj'] < 0.05) & (d7_diff['log2FoldChange'] > 1)].copy()
print(f"  D7 significantly upregulated genes (padj<0.05, log2FC>1): {len(d7_up)}")

# Among D7 up, which return to baseline by D14?
d7_peak_genes = []
for gene in d7_up.index:
    if gene in d14_diff.index:
        d14_fc = d14_diff.loc[gene, 'log2FoldChange']
        d7_fc = d7_up.loc[gene, 'log2FoldChange']
        if abs(d14_fc) < abs(d7_fc) * 0.5:  # D14 FC < 50% of D7 FC
            d7_peak_genes.append({
                'gene': gene,
                'D7_log2FC': d7_fc,
                'D14_log2FC': d14_fc,
                'D7_padj': d7_up.loc[gene, 'padj'],
                'recovery_ratio': 1 - abs(d14_fc) / abs(d7_fc)
            })

d7_peak_df = pd.DataFrame(d7_peak_genes).sort_values('D7_log2FC', ascending=False)
print(f"  D7-peak genes (resolve by D14): {len(d7_peak_df)}")

# Save
d7_peak_df.to_csv(RESULTS_DIR / 'D7_peak_resolve_D14_genes.csv', index=False)
print(f"  Saved: D7_peak_resolve_D14_genes.csv")

# Check overlap with our gene sets
print("\n  D7-peak genes overlapping with LDAM signature:")
ldam_overlap = [g for g in d7_peak_df['gene'] if g in LDAM_GENES]
print(f"    {ldam_overlap}")

print("  D7-peak genes overlapping with DAM signature:")
dam_overlap = [g for g in d7_peak_df['gene'] if g in DAM_GENES]
print(f"    {dam_overlap}")

print("  D7-peak genes overlapping with Lipid Metabolism:")
lipid_overlap = [g for g in d7_peak_df['gene'] if g in LIPID_METABOLISM]
print(f"    {lipid_overlap}")

# =============================================================================
# 11. Figure 11: Volcano plot D7
# =============================================================================
print("\nGenerating Figure 11: D7 volcano plot with lipid genes highlighted...")

fig, ax = plt.subplots(figsize=(12, 9))

d7_plot = d7_diff.copy()
d7_plot['neg_log10_padj'] = -np.log10(d7_plot['padj'].clip(lower=1e-300))
d7_plot = d7_plot.dropna(subset=['log2FoldChange', 'neg_log10_padj'])

# Background
ax.scatter(d7_plot['log2FoldChange'], d7_plot['neg_log10_padj'],
           c='lightgray', s=5, alpha=0.5, edgecolors='none')

# Significant
sig = d7_plot[(d7_plot['padj'] < 0.05) & (abs(d7_plot['log2FoldChange']) > 1)]
ax.scatter(sig['log2FoldChange'], sig['neg_log10_padj'],
           c='steelblue', s=8, alpha=0.6, edgecolors='none')

# Highlight gene sets
all_highlight = set(LDAM_GENES + DAM_GENES + LIPID_METABOLISM + IL6_PATHWAY + AUTOPHAGY_GENES)
highlight_colors = {
    'LDAM': ('#E53935', LDAM_GENES),
    'DAM': ('#FB8C00', DAM_GENES),
    'Lipid Met': ('#43A047', LIPID_METABOLISM),
    'IL-6': ('#8E24AA', IL6_PATHWAY),
    'Autophagy': ('#00ACC1', AUTOPHAGY_GENES),
}

for label, (color, genes) in highlight_colors.items():
    in_data = [g for g in genes if g in d7_plot.index]
    if in_data:
        subset = d7_plot.loc[in_data]
        ax.scatter(subset['log2FoldChange'], subset['neg_log10_padj'],
                   c=color, s=40, alpha=0.8, edgecolors='black', linewidths=0.5,
                   label=label, zorder=5)
        # Label significant ones
        sig_subset = subset[subset['padj'] < 0.05]
        for gene in sig_subset.index:
            ax.annotate(gene, (sig_subset.loc[gene, 'log2FoldChange'],
                              sig_subset.loc[gene, 'neg_log10_padj']),
                       fontsize=7, ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')

ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('log₂FC (Injury / Sham)', fontsize=12)
ax.set_ylabel('-log₁₀(padj)', fontsize=12)
ax.set_title('GSE163691: D7 Volcano Plot — Microglia After Cortical Stab Wound\nLipid/LDAM/DAM/IL-6 genes highlighted', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'fig11_D7_volcano.png', bbox_inches='tight')
plt.close()
print("  Saved: fig11_D7_volcano.png")

# =============================================================================
# 12. Summary Statistics Table
# =============================================================================
print("\nGenerating summary statistics...")

summary_rows = []
for set_name, genes in ALL_GENE_SETS.items():
    for gene in genes:
        if gene not in tpm.index:
            continue
        row = {'Gene_Set': set_name, 'Gene': gene}
        for tp in timepoints:
            if gene in diff_results[tp].index:
                row[f'{tp}_log2FC'] = diff_results[tp].loc[gene, 'log2FoldChange']
                row[f'{tp}_padj'] = diff_results[tp].loc[gene, 'padj']
            else:
                row[f'{tp}_log2FC'] = np.nan
                row[f'{tp}_padj'] = np.nan
        # Peak timepoint
        fcs = [row.get(f'{tp}_log2FC', 0) for tp in timepoints]
        if not all(np.isnan(f) if isinstance(f, float) else False for f in fcs):
            peak_idx = np.nanargmax([abs(f) for f in fcs])
            row['peak_timepoint'] = timepoints[peak_idx]
            row['peak_log2FC'] = fcs[peak_idx]
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_DIR / 'all_geneset_summary.csv', index=False)
print(f"  Saved: all_geneset_summary.csv ({len(summary_df)} gene-entries)")

# =============================================================================
# 13. Print key findings
# =============================================================================
print("\n" + "="*70)
print("KEY FINDINGS SUMMARY")
print("="*70)

for set_name, genes in [('LDAM', LDAM_GENES), ('DAM', DAM_GENES), ('IL-6', IL6_PATHWAY)]:
    print(f"\n--- {set_name} Signature ---")
    for gene in genes:
        if gene not in tpm.index:
            continue
        fcs = []
        sigs = []
        for tp in timepoints:
            if gene in diff_results[tp].index:
                fc = diff_results[tp].loc[gene, 'log2FoldChange']
                pval = diff_results[tp].loc[gene, 'padj']
                fcs.append(fc)
                sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
                sigs.append(sig)
            else:
                fcs.append(np.nan)
                sigs.append('NA')
        fc_str = ' | '.join([f"D{d}: {fc:+.2f}{s}" for d, fc, s in zip([1,4,7,14], fcs, sigs) if not np.isnan(fc)])
        print(f"  {gene:12s} {fc_str}")

print("\n" + "="*70)
print("DONE. All results saved to:", RESULTS_DIR)
print("="*70)
