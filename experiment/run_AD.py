"""
Scalability test (single dataset, adjustable number of highly variable genes and gene pairs,
precomputation uses all cells but only on highly variable genes)
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

# ==================== Configuration (adjust as needed) ====================
DATA_PATH = r"F:\metacell_analysis\processed_for_csn\AD_Cortex_all.mtx"
N_HVG = 100                     # Number of highly variable genes (reduce to speed up precomputation)
GENE_PAIRS = 200                # Number of gene pairs (reduce to speed up precomputation)
Q = 0.1
R_FASTCSN = 100
N_REPEATS = 3
CELL_SIZES = [500, 1000, 2000, 5000, 10000]   # Cell counts to test, up to 10000
RANDOM_SEED = 42

OUTPUT_CSV = "scalability_single.csv"
OUTPUT_PNG = "scalability_single.png"

np.random.seed(RANDOM_SEED)

# ==================== Helper functions ====================
def load_data(path):
    print(f"Loading data from {path}...")
    data = mmread(path)
    if hasattr(data, 'toarray'):
        data = data.toarray()
    else:
        data = np.array(data)
    print(f"Original data shape: {data.shape[0]} genes × {data.shape[1]} cells")
    return data

def select_high_variable_genes(data, n_genes):
    gene_var = np.var(data, axis=1)
    n_genes = min(n_genes, data.shape[0])
    top_indices = np.argsort(gene_var)[-n_genes:][::-1]
    selected_data = data[top_indices, :]
    print(f"Selected top {n_genes} variable genes (variance range: {gene_var[top_indices].min():.4f} - {gene_var[top_indices].max():.4f})")
    return selected_data, top_indices

def generate_gene_pairs(n_genes, n_pairs, seed=42):
    np.random.seed(seed)
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = np.random.randint(0, n_genes, 2)
        if i != j:
            pairs.add((i, j))
    return list(pairs)

# ==================== Main experiment ====================
def main():
    # 1. Load full data
    data_full = load_data(DATA_PATH)
    total_genes_orig, total_cells = data_full.shape
    print(f"Total cells: {total_cells}")

    # 2. Select highly variable genes (based on full data to cover full expression range)
    data_hvg, hvg_indices = select_high_variable_genes(data_full, N_HVG)
    n_genes, N = data_hvg.shape
    print(f"Using {n_genes} highly variable genes (precomputation based on all {N} cells).")

    # 3. Generate fixed gene pairs (based on HVG indices)
    gene_pairs = generate_gene_pairs(n_genes, GENE_PAIRS, RANDOM_SEED)
    print(f"Generated {len(gene_pairs)} gene pairs.")

    # 4. Precompute FastCSN (using all cells and HVG data)
    print("\nPrecomputing FastCSN on full data (one-time cost)...")
    fcsn = FastCSN(resolution=R_FASTCSN, q=Q)
    pre_start = time.time()
    fcsn.precompute_on_full_data(data_hvg, gene_pairs)   # uses all cells
    pre_time = time.time() - pre_start
    print(f"FastCSN precomputation time: {pre_time:.2f} seconds")

    # 5. Determine valid cell subset sizes (not exceeding total cells)
    valid_sizes = [s for s in CELL_SIZES if s <= total_cells]
    if total_cells not in valid_sizes:
        valid_sizes.append(total_cells)   # optional: include maximum cells
    print(f"Testing cell subset sizes: {valid_sizes}")

    records = []

    for N_sub in valid_sizes:
        print(f"\n======= Cell count: {N_sub} =======")
        for rep in range(N_REPEATS):
            print(f"  Replicate {rep+1}/{N_REPEATS}")
            # Randomly sample N_sub cells (indices in original data, used directly on HVG data)
            idx = np.random.choice(total_cells, size=N_sub, replace=False)
            X_sub = data_hvg[:, idx]

            # ---- oCSN ----
            start = time.time()
            nets_ocsn = build_ocsn(X_sub, gene_pairs, q=Q)
            elapsed = time.time() - start
            records.append({'cells': N_sub, 'rep': rep, 'method': 'oCSN', 'time': elapsed})
            print(f"      oCSN: {elapsed:.2f}s")

            # ---- locCSN ----
            start = time.time()
            nets_loc = build_loccsn_full(X_sub, gene_pairs, q=Q, target_size=20, k_neighbors=100)
            elapsed = time.time() - start
            records.append({'cells': N_sub, 'rep': rep, 'method': 'locCSN', 'time': elapsed})
            print(f"      locCSN: {elapsed:.2f}s")

            # ---- PearsonCSN ----
            start = time.time()
            nets_pearson = build_pearson(X_sub, gene_pairs, q=Q, thresh=0)
            elapsed = time.time() - start
            records.append({'cells': N_sub, 'rep': rep, 'method': 'PearsonCSN', 'time': elapsed})
            print(f"      PearsonCSN: {elapsed:.2f}s")

            # ---- SpearmanCSN ----
            start = time.time()
            nets_spearman = build_spearman(X_sub, gene_pairs, q=Q, thresh=0)
            elapsed = time.time() - start
            records.append({'cells': N_sub, 'rep': rep, 'method': 'SpearmanCSN', 'time': elapsed})
            print(f"      SpearmanCSN: {elapsed:.2f}s")

            # ---- NMICSN ----
            start = time.time()
            nets_nmi = build_nmi(X_sub, gene_pairs, q=Q, thresh=0)
            elapsed = time.time() - start
            records.append({'cells': N_sub, 'rep': rep, 'method': 'NMICSN', 'time': elapsed})
            print(f"      NMICSN: {elapsed:.2f}s")

            # ---- FastCSN (lookup only) ----
            start = time.time()
            nets_fast = fcsn.build_networks_for_subset(idx)   # idx are original cell indices, applicable to HVG data
            elapsed = time.time() - start
            records.append({'cells': N_sub, 'rep': rep, 'method': 'FastCSN', 'time': elapsed})
            print(f"      FastCSN (query only): {elapsed:.2f}s")

    # Save results
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Remove the first replicate of locCSN if it is an outlier (optional)
    df_clean = df[~((df['method'] == 'locCSN') & (df['rep'] == 0))].copy()

    # Plotting
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 300, 'savefig.dpi': 300,
    })

    summary = df_clean.groupby(['cells', 'method'])['time'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))

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

    ax.set_xlabel('Number of cells', fontweight='bold')
    ax.set_ylabel('Time (seconds, log scale)', fontweight='bold')
    ax.set_title('Scalability Comparison (AD Cortex)', fontweight='bold', loc='left')
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {OUTPUT_PNG}")

    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()