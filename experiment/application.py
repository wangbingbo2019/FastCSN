# -*- coding: utf-8 -*-
"""
ASD Differential Network Analysis using FastCSN (Edge-based Z-score Method)
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from itertools import combinations
from scipy.stats import ranksums, norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from fastcsn import FastCSN

# ==========================================================
# Parameters
# ==========================================================
DATA_DIR = "data/GSE129308_RAW"          # Place your 10x H5 files here

CONTROL_FILES = [
    "GSM6261344_Control-1-MAP2_filtered_feature_bc_matrix.h5",
    "GSM6261345_Control-2-MAP2_filtered_feature_bc_matrix.h5",
    "GSM6261346_Control-3-MAP2_filtered_feature_bc_matrix.h5",
]

ASD_FILES = [
    "GSM3704357_1-MAP2_filtered_feature_bc_matrix.h5",
    "GSM3704359_2-MAP2_filtered_feature_bc_matrix.h5",
    "GSM3704361_3-MAP2_filtered_feature_bc_matrix.h5",
]

N_HVG = 300
FASTCSN_RESOLUTION = 100
FASTCSN_Q = 0.1
Z_THRESHOLD = 3.5
TOP_DNG = 50
N_PERMUTATIONS = 0
RANDOM_SEED = 42
MAX_CELLS_PER_GROUP = 1000

os.makedirs("results", exist_ok=True)
np.random.seed(RANDOM_SEED)

# ==========================================================
# 1. Load and merge datasets
# ==========================================================
print("=" * 60)
print("Loading datasets...")
all_adatas = []
for f in CONTROL_FILES + ASD_FILES:
    path = os.path.join(DATA_DIR, f)
    print("Reading:", f)
    adata = sc.read_10x_h5(path)
    adata.var_names_make_unique()
    all_adatas.append(adata)

# Intersect common genes
common_genes = all_adatas[0].var_names
for a in all_adatas[1:]:
    common_genes = common_genes.intersection(a.var_names)
print(f"Common genes: {len(common_genes)}")

# Subset to common genes and add group labels
for i, a in enumerate(all_adatas):
    a.obsm['tmp'] = a[:, common_genes].X
    a = a[:, common_genes].copy()
    all_adatas[i] = a
    if i < len(CONTROL_FILES):
        a.obs['group'] = 'Control'
    else:
        a.obs['group'] = 'ASD'

adata = ad.concat(all_adatas, join='inner', index_unique='-')
print("Merged shape:", adata.shape)

# QC
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
print("After QC:", adata.shape)

# Normalization
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# HVG selection
print(f"Selecting top {N_HVG} HVGs...")
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
adata = adata[:, adata.var.highly_variable].copy()
print("After HVG:", adata.shape)

# ==========================================================
# 2. Downsample cells per group to avoid memory issues
# ==========================================================
print(f"Downsampling to max {MAX_CELLS_PER_GROUP} cells per group...")
asd_mask = adata.obs['group'] == 'ASD'
ctrl_mask = adata.obs['group'] == 'Control'
asd_idx_all = np.where(asd_mask)[0]
ctrl_idx_all = np.where(ctrl_mask)[0]
if len(asd_idx_all) > MAX_CELLS_PER_GROUP:
    asd_idx_sampled = np.random.choice(asd_idx_all, MAX_CELLS_PER_GROUP, replace=False)
else:
    asd_idx_sampled = asd_idx_all
if len(ctrl_idx_all) > MAX_CELLS_PER_GROUP:
    ctrl_idx_sampled = np.random.choice(ctrl_idx_all, MAX_CELLS_PER_GROUP, replace=False)
else:
    ctrl_idx_sampled = ctrl_idx_all
keep_idx = np.sort(np.concatenate([asd_idx_sampled, ctrl_idx_sampled]))
adata = adata[keep_idx].copy()
print(f"Downsampled shape: {adata.shape} (ASD: {len(asd_idx_sampled)}, Control: {len(ctrl_idx_sampled)})")

# Expression matrix (genes × cells)
X = adata.X.T
if not isinstance(X, np.ndarray):
    X = X.toarray()
genes = np.array(adata.var_names)
n_genes = len(genes)
print(f"Genes: {n_genes}, Cells: {X.shape[1]}")

# ==========================================================
# 3. Build gene pairs and run FastCSN
# ==========================================================
gene_pairs = list(combinations(range(n_genes), 2))
print(f"Total gene pairs: {len(gene_pairs)}")

print("\nRunning FastCSN precompute...")
fcsn = FastCSN(resolution=FASTCSN_RESOLUTION, q=FASTCSN_Q)
fcsn.precompute_on_full_data(X, gene_pairs)

cell_indices = np.arange(X.shape[1])
nets = fcsn.build_networks_for_subset(cell_indices)
print(f"Networks built: {len(nets)}")

# ==========================================================
# 4. Extract edge weight matrices (cells × gene_pairs)
# ==========================================================
groups = adata.obs["group"].values
asd_idx = np.where(groups == "ASD")[0]
ctrl_idx = np.where(groups == "Control")[0]
n_asd = len(asd_idx)
n_ctrl = len(ctrl_idx)
n_pairs = len(gene_pairs)

print(f"\nASD cells: {n_asd}, Control cells: {n_ctrl}")

W_asd = np.zeros((n_asd, n_pairs))
W_ctrl = np.zeros((n_ctrl, n_pairs))

print("Extracting edge weights...")
for p, (i, j) in enumerate(tqdm(gene_pairs)):
    for k, c in enumerate(asd_idx):
        W_asd[k, p] = nets[c][i, j]
    for k, c in enumerate(ctrl_idx):
        W_ctrl[k, p] = nets[c][i, j]

# ==========================================================
# 5. Compute edge-level z-scores
# ==========================================================
mean_asd = W_asd.mean(axis=0)
mean_ctrl = W_ctrl.mean(axis=0)
var_asd = W_asd.var(axis=0, ddof=1)
var_ctrl = W_ctrl.var(axis=0, ddof=1)

denom = np.sqrt(var_asd + var_ctrl)
denom[denom == 0] = 1e-6
z_scores = (mean_asd - mean_ctrl) / denom

diff_edge_mask = np.abs(z_scores) > Z_THRESHOLD
diff_edges = np.where(diff_edge_mask)[0]
print(f"\nDifferential edges (|z|>{Z_THRESHOLD}): {len(diff_edges)}")

# ==========================================================
# 6. Gene differential network scores
# ==========================================================
gene_scores = np.zeros(n_genes)
for p in diff_edges:
    i, j = gene_pairs[p]
    gene_scores[i] += np.abs(z_scores[p])
    gene_scores[j] += np.abs(z_scores[p])

dng_df = pd.DataFrame({
    'gene': genes,
    'score': gene_scores,
    'diff_edge_count': [np.sum([1 for p in diff_edges if gene_pairs[p][0]==g or gene_pairs[p][1]==g]) for g in range(n_genes)]
})
dng_df = dng_df.sort_values('score', ascending=False).reset_index(drop=True)
dng_df.to_csv("results/differential_network_genes.csv", index=False)

top_genes = dng_df.head(TOP_DNG)['gene'].values
with open("results/DNG_top50.txt", "w") as f:
    f.write("\n".join(top_genes))
print(f"Top {TOP_DNG} DNGs saved to results/DNG_top50.txt")

# ==========================================================
# 7. Edge significance (normal approximation)
# ==========================================================
if N_PERMUTATIONS > 0:
    print(f"\nRunning {N_PERMUTATIONS} permutations...")
    all_cells = np.arange(len(nets))
    perm_z_max = []
    for _ in tqdm(range(N_PERMUTATIONS)):
        np.random.shuffle(all_cells)
        perm_asd = all_cells[:n_asd]
        perm_ctrl = all_cells[n_asd:n_asd+n_ctrl]
        m_a = np.array([nets[c][gene_pairs[p][0], gene_pairs[p][1]] for p in range(n_pairs) for c in perm_asd]).reshape(n_pairs, n_asd).mean(axis=1)
        m_c = np.array([nets[c][gene_pairs[p][0], gene_pairs[p][1]] for p in range(n_pairs) for c in perm_ctrl]).reshape(n_pairs, n_ctrl).mean(axis=1)
        v_a = np.array([nets[c][gene_pairs[p][0], gene_pairs[p][1]] for p in range(n_pairs) for c in perm_asd]).reshape(n_pairs, n_asd).var(axis=1, ddof=1)
        v_c = np.array([nets[c][gene_pairs[p][0], gene_pairs[p][1]] for p in range(n_pairs) for c in perm_ctrl]).reshape(n_pairs, n_ctrl).var(axis=1, ddof=1)
        denom = np.sqrt(v_a + v_c + 1e-12)
        perm_z = (m_a - m_c) / denom
        perm_z_max.append(np.max(np.abs(perm_z)))
    perm_z_max = np.array(perm_z_max)
    emp_p = np.array([np.mean(perm_z_max >= abs(z)) for z in z_scores])
    edge_df = pd.DataFrame({
        'gene_i': [genes[i] for i,j in gene_pairs],
        'gene_j': [genes[j] for i,j in gene_pairs],
        'z_score': z_scores,
        'empirical_p': emp_p
    })
else:
    p_from_z = 2 * norm.sf(np.abs(z_scores))
    edge_df = pd.DataFrame({
        'gene_i': [genes[i] for i,j in gene_pairs],
        'gene_j': [genes[j] for i,j in gene_pairs],
        'z_score': z_scores,
        'p_nominal': p_from_z
    })

edge_df.to_csv("results/differential_edges.csv", index=False)
print("Differential edges saved to results/differential_edges.csv")

# ==========================================================
# 8. Save group-averaged network matrices
# ==========================================================
print("\nSaving group-averaged network matrices...")
avg_net_asd = np.zeros((n_genes, n_genes))
avg_net_ctrl = np.zeros((n_genes, n_genes))
for p, (i, j) in enumerate(gene_pairs):
    w_asd = mean_asd[p]
    w_ctrl = mean_ctrl[p]
    avg_net_asd[i, j] = w_asd
    avg_net_asd[j, i] = w_asd
    avg_net_ctrl[i, j] = w_ctrl
    avg_net_ctrl[j, i] = w_ctrl

df_asd = pd.DataFrame(avg_net_asd, index=genes, columns=genes)
df_ctrl = pd.DataFrame(avg_net_ctrl, index=genes, columns=genes)
df_asd.to_csv("results/average_network_ASD.csv")
df_ctrl.to_csv("results/average_network_Control.csv")
print("Group-averaged networks saved to results/")

# ==========================================================
# 9. Compare with conventional differential expression
# ==========================================================
print("\nPerforming traditional DE analysis (Wilcoxon)...")
de_pvals = []
de_logfc = []
for g in tqdm(range(n_genes)):
    asd_expr = X[g, asd_idx]
    ctrl_expr = X[g, ctrl_idx]
    stat, p = ranksums(asd_expr, ctrl_expr)
    de_pvals.append(p)
    de_logfc.append(np.log2(asd_expr.mean() + 1e-6) - np.log2(ctrl_expr.mean() + 1e-6))

de_fdr = multipletests(de_pvals, method='fdr_bh')[1]
de_df = pd.DataFrame({
    'gene': genes,
    'logFC': de_logfc,
    'pvalue': de_pvals,
    'FDR': de_fdr
})
de_df = de_df.sort_values('FDR')
de_df.to_csv("results/DE_results.csv", index=False)

deg = de_df[de_df['FDR'] < 0.05]['gene'].values
with open("results/DEG_list.txt", "w") as f:
    f.write("\n".join(deg))
print(f"DEGs (FDR<0.05): {len(deg)}")

# ==========================================================
# 10. Summary
# ==========================================================
print("\n" + "=" * 60)
print("Analysis Finished")
print("=" * 60)
print(f"Total genes: {n_genes}")
print(f"Differential edges: {len(diff_edges)}")
print(f"DEGs (FDR<0.05): {len(deg)}")
print(f"Top {TOP_DNG} DNGs saved for GO analysis")
print("Results in 'results/' folder:")
print("  differential_edges.csv")
print("  differential_network_genes.csv")
print("  DNG_top50.txt")
print("  DE_results.csv")
print("  DEG_list.txt")
print("  average_network_ASD.csv")
print("  average_network_Control.csv")