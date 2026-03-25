# 10. 溶酶体相关基因完整比较（08分析文件所有基因）

**来源**：`08_lysosomal_function_analysis.md` 中出现的所有基因  
**基因总数**：39 个，分 8 个功能类别  
**分析日期**：2026-03-17  

## 比较组

| 编号 | 数据集 | 比较 | 层级 |
|------|--------|------|------|
| G1 | Mehrabani 2025 TableS4 | TBI D3 vs Sham | 全体 CD11b+ (scRNA-seq) |
| G2 | Mehrabani 2025 TableS6 | DAM3 vs 全体 TBI 微 | 小胶质亚群 |
| G3 | GSE226211 | 3dpi_CTRL vs Intact | 全体微 (Wilcoxon) |
| G4 | GSE226211 | 5dpi_CTRL vs Intact | 全体微 (Wilcoxon) |
| G5 | GSE226211 | Plin2-high (C7+C4) vs Other | res=0.3 亚群 |
| G6 | GSE163691 | D1 injury vs sham | bulk DESeq2 |
| G7 | GSE163691 | D4 injury vs sham | bulk DESeq2 |
| G8 | GSE163691 | D7 injury vs sham | bulk DESeq2 |
| G9 | GSE163691 | D14 injury vs sham | bulk DESeq2 |

> ↑/↓ + log2FC，* = padj<0.05；n/f = 数据中未检出

## Lysosomal membrane

| 基因 | 描述 | G1<br>Meh TBI | G2<br>Meh DAM3 | G3<br>226 3dpi | G4<br>226 5dpi | G5<br>226 Plin2hi | G6<br>D1 | G7<br>D4 | G8<br>D7 | G9<br>D14 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Lamp1** | Lysosomal membrane protein 1 (Meh: IF marker) | ↑1.13* | ↑0.42* | ↓0.12 | ↓0.11* | ↑0.05 | ↓0.26 | ↑1.34* | ↑1.62* | ↑1.51* |
| **Lamp2** | Lysosomal membrane protein 2 | ↑0.32* | ↑0.13* | ↓0.27* | ↑0.07 | ↓0.08* | ↓0.23 | ↑0.89 | ↑0.77 | ↑0.78 |
| **Cd63** | Lysosome-phagosome fusion marker | ↑2.38* | ↑0.97* | ↑0.67* | ↑0.62* | ↑0.31* | ↑2.17* | ↑3.27* | ↑2.02* | ↑2.62* |
| **Cd68** | Lysosomal macrophage marker | ↑1.54* | ↑0.74* | ↑0.19* | ↓0.12* | ↑0.60* | ↑0.39 | ↑1.68* | ↑1.32* | ↑0.73 |

## Cathepsins

| 基因 | 描述 | G1<br>Meh TBI | G2<br>Meh DAM3 | G3<br>226 3dpi | G4<br>226 5dpi | G5<br>226 Plin2hi | G6<br>D1 | G7<br>D4 | G8<br>D7 | G9<br>D14 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Ctsb** | Cysteine protease (Meh: transcript↑) | ↑2.47* | ↑0.77* | ↑0.42* | ↑0.40* | ↑0.52* | ↑1.65* | ↑2.73* | ↑2.26* | ↑2.24* |
| **Ctsd** | Aspartyl protease (Meh: activity↓) | ↑1.58* | ↑0.74* | ↓1.08* | ↓0.43* | ↓1.31* | ↓0.61 | ↑2.11* | ↑1.75* | ↑1.76* |
| **Ctsl** | Cysteine protease | ↑1.52* | ↑0.75* | ↓0.24 | ↑0.23* | ↓0.68* | ↑0.77* | ↑2.20* | ↑0.84* | ↑1.07* |
| **Ctse** | Aspartyl protease | n/f | ↑0.08* | ↑4.12 | ↑3.98 | ↑1.59 | ↑1.28 | ↑3.70* | ↑3.90* | ↑6.91* |

## Lysosomal hydrolases

| 基因 | 描述 | G1<br>Meh TBI | G2<br>Meh DAM3 | G3<br>226 3dpi | G4<br>226 5dpi | G5<br>226 Plin2hi | G6<br>D1 | G7<br>D4 | G8<br>D7 | G9<br>D14 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Lipa** | Lysosomal acid lipase CE/TG (Meh↑) | ↑1.74* | ↑0.64* | ↑0.45* | ↑0.46* | ↑0.99* | ↑0.25 | ↑2.22* | ↑2.00* | ↑1.75* |
| **Gba** | Glucocerebrosidase | ↑1.42* | ↑0.55* | ↑0.61* | ↑0.58* | ↑1.04* | ↑0.66 | ↑1.64* | ↑1.01* | ↑0.85 |
| **Galc** | Galactosylceramidase (myelin lipid) | ↑1.00* | ↑0.53* | ↑0.11 | ↑0.84* | ↑0.22* | ↑0.29 | ↑1.06* | ↑1.45* | ↑1.31* |
| **Hexa** | Hexosaminidase A | ↑0.64* | ↑0.17* | ↓0.56* | ↓0.54* | ↓0.30* | ↓0.97* | ↑0.84* | ↑1.27* | ↑1.10* |
| **Gusb** | Beta-glucuronidase | ↑1.11* | ↑0.60* | ↑0.36* | ↑0.28* | ↑0.68* | ↑0.60 | ↑1.85* | ↑1.55* | ↑0.93* |
| **Naglu** | N-acetylglucosaminidase (Meh: activity↓) | ↑1.16* | ↑0.33* | ↑0.07 | ↑0.16* | ↑0.25* | ↓0.61 | ↑1.26 | ↑1.14 | ↑0.22 |
| **Lgals3** | Galectin-3, lysosomal damage marker (Meh↑) | ↑5.45* | ↑0.82* | ↑6.51* | ↑4.43* | ↑6.26* | ↑6.23* | ↑6.96* | ↑3.07* | ↑5.92* |
| **Anxa5** | Phago-lysosomal coupling (Meh↑) | ↑3.20* | ↑0.89* | ↑2.50* | ↑1.39* | ↑3.22* | ↑2.87* | ↑2.30* | ↑2.24* | ↑2.61* |

## Lysosomal acidification

| 基因 | 描述 | G1<br>Meh TBI | G2<br>Meh DAM3 | G3<br>226 3dpi | G4<br>226 5dpi | G5<br>226 Plin2hi | G6<br>D1 | G7<br>D4 | G8<br>D7 | G9<br>D14 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Atp6v0d1** | V-ATPase subunit d1 | ↑0.79* | ↑0.43* | ↑0.61* | ↑0.36* | ↑0.93* | ↑0.66* | ↑0.51 | ↑1.19* | ↑0.30 |
| **Atp6v1h** | V-ATPase subunit h | ↓0.25* | ↓0.01* | ↓0.43 | ↑0.07* | ↑0.06* | ↑0.81* | ↑0.27 | ↑0.65* | ↑0.01 |
| **Lamtor1** | mTORC1-lysosome platform (Ragulator) | ↑1.08* | ↑0.31* | ↑0.65* | ↑0.25* | ↑0.78* | ↑0.19 | ↑0.76 | ↑1.21* | ↑0.95* |

## Intralysosomal cholesterol

| 基因 | 描述 | G1<br>Meh TBI | G2<br>Meh DAM3 | G3<br>226 3dpi | G4<br>226 5dpi | G5<br>226 Plin2hi | G6<br>D1 | G7<br>D4 | G8<br>D7 | G9<br>D14 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Npc1** | Lysosomal cholesterol export pump | ↑0.16* | ↓0.00* | ↓0.72 | ↑0.43* | ↓0.31 | ↓0.49 | ↑0.16 | ↑0.65 | ↑0.57 |
| **Npc2** | Intralysosomal cholesterol carrier (Meh↑) | ↑1.16* | ↑0.31* | ↑0.82* | ↑0.51* | ↑1.10* | ↑0.23 | ↑1.64* | ↑1.41* | ↑1.18* |
| **Nceh1** | Neutral CE hydrolase (Meh↑) | ↑2.99* | ↑0.93* | ↑1.31* | ↑1.68* | ↑1.50* | ↑2.61* | ↑2.02* | ↑2.22* | ↑1.97* |

## Cholesterol export / esterification

| 基因 | 描述 | G1<br>Meh TBI | G2<br>Meh DAM3 | G3<br>226 3dpi | G4<br>226 5dpi | G5<br>226 Plin2hi | G6<br>D1 | G7<br>D4 | G8<br>D7 | G9<br>D14 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Cyp46a1** | Cholesterol→24-OHC (primary export) | n/f | n/f | ↓1.41 | ↑0.65* | ↓0.84 | ↓2.94* | ↓4.17* | ↓4.02* | ↓3.76* |
| **Abca1** | Cellular cholesterol efflux | ↑0.02* | ↑0.18* | ↓0.52* | ↑0.98* | ↓0.28 | ↑1.00 | ↑1.14* | ↑1.39* | ↑1.71* |
| **Abcg1** | Cellular cholesterol efflux | ↑0.67* | ↑0.02* | ↑0.32* | ↑0.89* | ↑0.64* | ↑0.28 | ↑1.23 | ↑2.14* | ↑1.48 |
| **Osbpl1a** | ER-lysosome sterol transfer | ↓0.54* | ↓0.47* | ↓1.44* | ↑0.74* | ↓1.16* | ↓1.36 | ↓1.90* | ↓0.63 | ↓0.88 |
| **Soat1** | Cholesterol esterification → CE/LD | ↑1.28* | ↑0.65* | ↑0.43* | ↑0.97* | ↑0.93* | ↑1.37* | ↑1.71* | ↑1.82* | ↑1.75* |
| **Ch25h** | 25-hydroxylase (oxysterol bypass) | ↑6.52* | ↑2.31* | ↑3.73* | ↑3.86* | ↑2.51* | ↑7.28* | ↑6.91* | ↑1.48 | ↑5.74* |

## Phagocytic uptake

| 基因 | 描述 | G1<br>Meh TBI | G2<br>Meh DAM3 | G3<br>226 3dpi | G4<br>226 5dpi | G5<br>226 Plin2hi | G6<br>D1 | G7<br>D4 | G8<br>D7 | G9<br>D14 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Cd36** | Lipid/myelin receptor (Meh: myelin uptake) | ↑3.23* | ↑0.41* | ↑0.08 | ↓0.48 | ↑0.62* | ↑4.51* | ↑2.94* | ↑3.81* | ↑1.20 |
| **Msr1** | Scavenger receptor (myelin uptake) | ↑4.07* | ↑1.39* | ↑2.33* | ↑1.21* | ↑2.95* | ↑6.40* | ↑5.60* | ↑2.18* | ↑3.38* |
| **Axl** | TAM receptor, myelin phagocytosis | ↑2.39* | ↑0.60* | ↑2.78* | ↑2.47* | ↑2.45* | ↓0.76 | ↑1.62 | ↑2.49* | ↑3.05* |
| **Mertk** | TAM receptor, myelin phagocytosis | ↓1.17* | ↓1.20* | ↓2.25* | ↓0.29* | ↓2.44* | ↓1.47* | ↓0.99 | ↓0.69 | ↓0.28 |

## Autophagy-lysosome

| 基因 | 描述 | G1<br>Meh TBI | G2<br>Meh DAM3 | G3<br>226 3dpi | G4<br>226 5dpi | G5<br>226 Plin2hi | G6<br>D1 | G7<br>D4 | G8<br>D7 | G9<br>D14 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Sqstm1** | p62 autophagy receptor (Meh: protein↑) | ↑0.98* | ↑0.69* | ↑0.28* | ↑0.03 | ↑0.66* | ↑1.81* | ↑0.97* | ↑0.68 | ↑0.88 |
| **Map1lc3b** | LC3 autophagosome marker (Meh: protein↑) | ↑1.31* | ↑0.50* | ↑0.54* | ↑0.25* | ↑0.67* | ↓0.22 | ↑0.15 | ↑0.54 | ↑0.60 |
| **Becn1** | Autophagy initiation (Beclin-1) | ↓0.11* | ↓0.08* | ↑0.69* | ↑0.62* | ↑0.91* | ↑0.49 | ↓0.16 | ↑0.67 | ↑0.99 |
| **Tfeb** | Lysosome biogenesis TF (CLEAR network) | ↓0.77 | ↓0.87* | ↓0.12 | ↓0.26 | ↑0.11* | ↓1.05 | ↑0.08 | ↑0.37 | ↓0.51 |
| **Tfec** | TFEB family (phagocytic activation) | ↑1.79* | ↑0.10* | ↑1.41 | ↑0.33 | ↑1.66* | ↑1.32 | ↑2.86* | ↑1.94* | ↑1.62 |
| **Plin2** | Lipid droplet coat protein (LD marker) | ↑4.23* | ↑1.23* | ↑2.65* | ↑1.60* | ↑3.28* | ↑4.01* | ↑4.99* | ↑3.21* | ↑2.92* |
| **Il1b** | Inflammation / NF-κB target | ↑1.66* | n/f | ↑4.18* | ↑2.45* | ↑3.44* | ↑2.81* | ↑2.99* | ↑2.67* | ↑3.67* |

## 数据来源

| 文件 | 内容 |
|------|------|
| `data/Mehrabani2025_suppl/TableS4_Mehrabani2025.xlsx` | Mehrabani TBI D3 vs Sham |
| `data/Mehrabani2025_suppl/TableS6_Mehrabani2025.xlsx` | Mehrabani DAM3 vs TBI microglia |
| `data/GSE226211/` | 10个CTRL+Intact样本（INH excluded）|
| `data/GSE163691/GSE163691_DX_injury_sham-Diff.txt.gz` | bulk DESeq2 D1-D14 |
| `results/GSE226211/lysosomal_extended_table.csv` | 完整数值表 |
| `results/GSE226211/fig10_lysosomal_extended_heatmap.png` | 热图 |
| `scripts/GSE226211/06_lysosomal_extended_comparison.py` | 本脚本 |
