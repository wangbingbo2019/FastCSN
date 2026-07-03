"""
Scalability test across multiple datasets (single dataset example included)
"""

import time
import numpy as np
import pandas as pd
from scipy.io import mmread
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

warnings.filterwarnings('ignore')

from ocsn_official import build_ocsn, build_pearson, build_spearman, build_nmi
from loccsn_metacell import build_networks as build_loccsn_full
from fastcsn import FastCSN

# ==================== Configuration ====================
DATASETS = {
    "D4_COVID_Lung": {
        "path": r"F:\metacell_analysis\processed_data\COVID_Lung\pulmonary_alveolar_type_2_cell\counts_200genes.mtx",
        "sizes": [500,  2000, 5000, 10000, 19692],
    }
}

N_REPEATS = 5
Q = 0.1
GENE_PAIRS = 500
OUTPUT_CSV = "scalability_results.csv"
OUTPUT_PNG = "scalability_comparison.png"

# ==================== Load the first dataset to generate fixed gene pairs ====================
first_ds = list(DATASETS.keys())[0]
first_path = DATASETS[first_ds]["path"]
print(f"Loading {first_path} for gene pair generation...")
data = mmread(first_path)
if hasattr(data, 'toarray'):
    data = data.toarray()
else:
    data = np.array(data)
total_genes = data.shape[0]

np.random.seed(42)
gene_pairs = set()
while len(gene_pairs) < GENE_PAIRS:
    i, j = np.random.randint(0, total_genes, 2)
    if i != j:
        gene_pairs.add((i, j))
gene_pairs = list(gene_pairs)
print(f"Using {len(gene_pairs)} fixed gene pairs for all experiments.\n")

# ==================== Experiment logging ====================
records = []

for ds_name, ds_info in DATASETS.items():
    data_path = ds_info["path"]
    sizes = ds_info["sizes"]

    import os

    if not os.path.exists(data_path):
        print(f"Warning: {ds_name}: file not found, skipping...")
        continue

    print(f"\n{'=' * 60}")
    print(f"Dataset: {ds_name}")
    print(f"Loading {data_path}...")
    data = mmread(data_path)
    if hasattr(data, 'toarray'):
        data = data.toarray()
    else:
        data = np.array(data)
    total_cells = data.shape[1]
    sizes = [s for s in sizes if s <= total_cells]
    print(f"Data: {data.shape[0]} genes × {total_cells} cells")

    # ========== FastCSN: precompute once on the full dataset ==========
    print("\n  Precomputing FastCSN on full data (one-time cost)...")
    fcsn = FastCSN(resolution=100, q=Q)
    precompute_start = time.time()
    fcsn.precompute_on_full_data(data, gene_pairs)
    precompute_time = time.time() - precompute_start
    print(f"  FastCSN precomputation time: {precompute_time:.2f}s")

    # Run experiments for each cell subset size
    for N in sizes:
        print(f"\n  Cell count: {N}")
        for rep in range(N_REPEATS):
            print(f"    Rep {rep + 1}/{N_REPEATS}")
            idx = np.random.choice(total_cells, size=N, replace=False)
            X_sub = data[:, idx]

            # --- oCSN (baseline, recomputed each time) ---
            start = time.time()
            nets = build_ocsn(X_sub, gene_pairs, q=Q)
            elapsed = time.time() - start
            records.append({
                'dataset': ds_name, 'cells': N, 'rep': rep,
                'method': 'oCSN', 'time': elapsed,
                'nnz': nets[0].nnz if len(nets) > 0 else 0
            })
            print(f"      oCSN: {elapsed:.2f}s")

            # --- locCSN (baseline) ---
            start = time.time()
            nets = build_loccsn_full(X_sub, gene_pairs, q=Q, target_size=20, k_neighbors=100)
            elapsed = time.time() - start
            records.append({
                'dataset': ds_name, 'cells': N, 'rep': rep,
                'method': 'locCSN', 'time': elapsed,
                'nnz': nets[0].nnz if len(nets) > 0 else 0
            })
            print(f"      locCSN: {elapsed:.2f}s")

            # --- PearsonCSN ---
            start = time.time()
            nets = build_pearson(X_sub, gene_pairs, q=Q, thresh=0)
            elapsed = time.time() - start
            records.append({
                'dataset': ds_name, 'cells': N, 'rep': rep,
                'method': 'PearsonCSN', 'time': elapsed,
                'nnz': nets[0].nnz if len(nets) > 0 else 0
            })
            print(f"      PearsonCSN: {elapsed:.2f}s")

            # --- SpearmanCSN ---
            start = time.time()
            nets = build_spearman(X_sub, gene_pairs, q=Q, thresh=0)
            elapsed = time.time() - start
            records.append({
                'dataset': ds_name, 'cells': N, 'rep': rep,
                'method': 'SpearmanCSN', 'time': elapsed,
                'nnz': nets[0].nnz if len(nets) > 0 else 0
            })
            print(f"      SpearmanCSN: {elapsed:.2f}s")

            # --- NMICSN ---
            start = time.time()
            nets = build_nmi(X_sub, gene_pairs, q=Q, thresh=0)
            elapsed = time.time() - start
            records.append({
                'dataset': ds_name, 'cells': N, 'rep': rep,
                'method': 'NMICSN', 'time': elapsed,
                'nnz': nets[0].nnz if len(nets) > 0 else 0
            })
            print(f"      NMICSN: {elapsed:.2f}s")

            # --- FastCSN (uses precomputed tables, lookup only) ---
            start = time.time()
            nets = fcsn.build_networks_for_subset(idx)
            elapsed = time.time() - start
            records.append({
                'dataset': ds_name, 'cells': N, 'rep': rep,
                'method': 'FastCSN', 'time': elapsed,
                'nnz': nets[0].nnz if len(nets) > 0 else 0
            })
            print(f"      FastCSN (query only): {elapsed:.2f}s")

# ==================== Save results ====================
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to {OUTPUT_CSV}")

# ==================== Data cleaning: remove the first replicate of locCSN (often an outlier) ====================
df_clean = df[~((df['method'] == 'locCSN') & (df['rep'] == 0))].copy()

# ==================== Plotting (semi‑log: linear x, log y) ====================
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

# Plot per dataset
for ds_name in DATASETS:
    ds_df = df_clean[df_clean['dataset'] == ds_name]
    if len(ds_df) == 0:
        continue

    # Compute mean time per cell count and method
    summary = ds_df.groupby(['cells', 'method'])['time'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))

    # Colors and markers
    colors = {'FastCSN': '#F0E442', 'locCSN': '#CC79A7', 'oCSN': '#E69F00'}
    markers = {'oCSN': 'o', 'locCSN': 'D', 'PearsonCSN': 's',
               'SpearmanCSN': '^', 'NMICSN': 'v', 'FastCSN': 'p'}

    for method in summary['method'].unique():
        sub = summary[summary['method'] == method]
        color = colors.get(method, '#888888')
        marker = markers.get(method, 'o')
        lw = 1.8 if method in ['FastCSN', 'locCSN'] else 1.0
        ax.semilogy(sub['cells'], sub['time'],
                    color=color, marker=marker, label=method,
                    linewidth=lw, markersize=6, alpha=0.9, markeredgewidth=0.5)

    ax.set_xlabel('Number of cells', fontweight='bold', fontsize=10)
    ax.set_ylabel('Time (seconds, log scale)', fontweight='bold', fontsize=10)
    ax.set_title(f'Scalability Comparison — {ds_name}', fontweight='bold', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='both', labelsize=9)
    plt.tight_layout()
    plt.savefig(f"scalability_{ds_name}.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved: scalability_{ds_name}.png")

print("\nAll experiments completed successfully!")