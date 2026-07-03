"""
Experiment 2: Sensitivity analysis of resolution R (full version, 5 repeats, comprehensive plots)
Metrics: Pearson r, Spearman ρ, L1 distance, recall of significant edges (top 1%)
Outputs:
  - sensitivity_results_5reps.csv       (raw data)
  - sensitivity_time.png                (dual-axis precomputation + query time)
  - sensitivity_combined_lines.png      (line plots for all four metrics combined)
  - sensitivity_boxplot_{metric}.png    (box plots per metric, faceted by cell count)
  - sensitivity_heatmap_{metric}.png    (heatmap per metric, R vs cell count)
  - sensitivity_scatter_{metric}.png    (scatter plots with error bands per metric)
Colors: N=1000 #c2a281, N=5000 #7f99c1, N=10000 #6cc4b3
"""

import time
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Import custom modules (adjust paths if necessary)
from ocsn_official import build_ocsn
from fastcsn import FastCSN

# ==================== Configuration ====================
DATA_PATH = r"F:\metacell_analysis\processed_data\COVID_Lung\pulmonary_alveolar_type_2_cell\counts_200genes.mtx"
GENE_PAIRS = 500
Q = 0.1
RESOLUTIONS = [10, 20, 50, 75, 100, 200]
CELL_SIZES = [1000, 5000, 10000]
N_REPEATS = 5                         # 5 replicates (reduced from 10)
TOP_PERCENT = 0.01
RANDOM_SEED = 42

OUTPUT_CSV = "sensitivity_results_5reps.csv"
OUTPUT_DIR = "./sensitivity_plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# New color scheme (soft tones for publication)
CELL_COLORS = {
    1000: '#c2a281',   # warm brown
    5000: '#7f99c1',   # muted blue
    10000: '#6cc4b3',  # mint green
}

np.random.seed(RANDOM_SEED)

# ==================== Helper functions ====================
def load_data(path):
    data = mmread(path)
    if hasattr(data, 'toarray'):
        data = data.toarray()
    return np.array(data)

def get_network_weights(networks, gene_pairs, n_cells):
    n_pairs = len(gene_pairs)
    weights = np.zeros((n_cells, n_pairs))
    for c, net in enumerate(networks):
        for idx, (i, j) in enumerate(gene_pairs):
            w = net[i, j] if net[i, j] != 0 else 0.0
            weights[c, idx] = w
    return weights

def evaluate_vs_ocsn(weights_test, weights_ocsn, top_k):
    n_cells = weights_test.shape[0]
    pearson_list, spearman_list, l1_list, recall_list = [], [], [], []
    for c in range(n_cells):
        v_test = weights_test[c]
        v_ocsn = weights_ocsn[c]
        if np.std(v_test) > 0 and np.std(v_ocsn) > 0:
            pearson_list.append(pearsonr(v_test, v_ocsn)[0])
            spearman_list.append(spearmanr(v_test, v_ocsn)[0])
        else:
            pearson_list.append(np.nan)
            spearman_list.append(np.nan)
        l1_list.append(np.mean(np.abs(v_test - v_ocsn)))
        ocsn_top = np.argsort(np.abs(v_ocsn))[-top_k:]
        test_top = np.argsort(np.abs(v_test))[-top_k:]
        recall = len(set(ocsn_top) & set(test_top)) / top_k if top_k > 0 else 0.0
        recall_list.append(recall)
    return {
        'pearson': np.nanmean(pearson_list),
        'spearman': np.nanmean(spearman_list),
        'l1': np.mean(l1_list),
        'recall': np.mean(recall_list)
    }

# ==================== Main experiment ====================
def run_experiment():
    data_full = load_data(DATA_PATH)
    total_genes, total_cells = data_full.shape
    print(f"Data shape: {total_genes} genes × {total_cells} cells")

    # Fixed gene pairs across all runs
    gene_pairs = set()
    while len(gene_pairs) < GENE_PAIRS:
        i, j = np.random.randint(0, total_genes, 2)
        if i != j:
            gene_pairs.add((i, j))
    gene_pairs = list(gene_pairs)
    print(f"Using {len(gene_pairs)} gene pairs.\n")

    records = []
    for N in CELL_SIZES:
        if N > total_cells:
            print(f"Skipping N={N} > total_cells")
            continue
        print(f"\n========== Cell count: {N} ==========")
        for rep in range(N_REPEATS):
            print(f"  Replicate {rep+1}/{N_REPEATS}")
            idx = np.random.choice(total_cells, size=N, replace=False)
            X_sub = data_full[:, idx]

            # oCSN baseline (computed once per replicate)
            nets_ocsn = build_ocsn(X_sub, gene_pairs, q=Q)
            weights_ocsn = get_network_weights(nets_ocsn, gene_pairs, N)
            top_k = max(1, int(TOP_PERCENT * GENE_PAIRS))

            for R in RESOLUTIONS:
                print(f"    R = {R}")
                fcsn = FastCSN(resolution=R, q=Q)
                pre_start = time.time()
                fcsn.precompute_on_full_data(X_sub, gene_pairs)
                pre_time = time.time() - pre_start

                query_start = time.time()
                nets_fast = fcsn.build_networks_for_subset(list(range(N)))
                query_time = time.time() - query_start

                weights_fast = get_network_weights(nets_fast, gene_pairs, N)
                metrics = evaluate_vs_ocsn(weights_fast, weights_ocsn, top_k)

                records.append({
                    'cells': N, 'R': R, 'rep': rep,
                    'precompute_time': pre_time,
                    'query_time': query_time,
                    'pearson': metrics['pearson'],
                    'spearman': metrics['spearman'],
                    'l1': metrics['l1'],
                    'recall': metrics['recall']
                })
                print(f"      pre={pre_time:.2f}s, query={query_time:.2f}s, pearson={metrics['pearson']:.4f}, recall={metrics['recall']:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")
    return df

# ==================== Plotting functions (publication-ready style) ====================
def set_style():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })
    sns.set_style("ticks")

def plot_time(df):
    """Combined time plot (dual axis)"""
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.set_xscale('log')
    ax1.set_xlabel('Resolution R', fontweight='bold')
    ax1.set_ylabel('Precomputation time (s)', color=CELL_COLORS[1000], fontweight='bold')
    for N in CELL_SIZES:
        sub = df[df['cells'] == N].groupby('R')['precompute_time'].mean().reset_index()
        ax1.plot(sub['R'], sub['precompute_time'], marker='o', color=CELL_COLORS[N],
                 linestyle='-', linewidth=1.5, label=f'N = {N} (pre)')
    ax1.tick_params(axis='y', labelcolor=CELL_COLORS[1000])
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Query time (s)', color=CELL_COLORS[5000], fontweight='bold')
    for N in CELL_SIZES:
        sub = df[df['cells'] == N].groupby('R')['query_time'].mean().reset_index()
        ax2.plot(sub['R'], sub['query_time'], marker='s', color=CELL_COLORS[N],
                 linestyle='--', linewidth=1.5, label=f'N = {N} (query)')
    ax2.tick_params(axis='y', labelcolor=CELL_COLORS[5000])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False, fontsize=8)
    ax1.set_title('Precomputation & Query Time', fontweight='bold', loc='left', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "sensitivity_time.png", bbox_inches='tight')
    plt.close()
    print("Saved: sensitivity_time.png")

def plot_combined_lines(df):
    """Combined line plots for all four metrics (2×2 subplots)"""
    metrics = ['pearson', 'spearman', 'l1', 'recall']
    ylabels = ["Pearson's r", "Spearman's ρ", 'L1 distance', 'Recall (top 1%)']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes[idx]
        ax.set_xscale('log')
        ax.set_xlabel('Resolution R', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        for N in CELL_SIZES:
            sub = df[df['cells'] == N]
            grouped = sub.groupby('R')[metric].agg(['mean', 'std']).reset_index()
            ax.errorbar(grouped['R'], grouped['mean'], yerr=grouped['std'],
                        marker='o', capsize=3, color=CELL_COLORS[N],
                        label=f'N = {N}', linewidth=1.5, elinewidth=1, capthick=1)
        ax.set_title(ylabel, fontweight='bold', loc='left', fontsize=11)
        ax.legend(frameon=False, loc='best', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        if metric in ['pearson', 'spearman']:
            ax.set_ylim(0.3, 1.02)
        elif metric == 'recall':
            ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "sensitivity_combined_lines.png", bbox_inches='tight')
    plt.close()
    print("Saved: sensitivity_combined_lines.png")

def plot_boxplots(df):
    """Box plots for each metric, faceted by cell count"""
    metrics = ['pearson', 'spearman', 'l1', 'recall']
    ylabels = ["Pearson's r", "Spearman's ρ", 'L1 distance', 'Recall (top 1%)']
    for metric, ylabel in zip(metrics, ylabels):
        fig, axes = plt.subplots(1, len(CELL_SIZES), figsize=(12, 4), sharey=False)
        if len(CELL_SIZES) == 1:
            axes = [axes]
        for ax, N in zip(axes, CELL_SIZES):
            sub = df[df['cells'] == N]
            sns.boxplot(data=sub, x='R', y=metric, color=CELL_COLORS[N], ax=ax,
                        showfliers=False, width=0.6, boxprops=dict(alpha=0.7))
            ax.set_title(f'N = {N} cells', fontweight='bold', loc='left', fontsize=11)
            ax.set_xlabel('Resolution R', fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        fig.suptitle(f'Impact of R on {ylabel}', fontweight='bold', y=1.02, fontsize=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + f"sensitivity_boxplot_{metric}.png", bbox_inches='tight')
        plt.close()
        print(f"Saved: sensitivity_boxplot_{metric}.png")

def plot_heatmaps(df):
    """Heatmaps for each metric (R vs cell count)"""
    metrics = ['pearson', 'spearman', 'l1', 'recall']
    for metric in metrics:
        pivot = df.groupby(['cells', 'R'])[metric].mean().unstack(level=0)
        cmap = 'RdYlBu_r' if metric in ['pearson', 'spearman'] else 'viridis'
        plt.figure(figsize=(5, 4))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap,
                    cbar_kws={'label': metric}, square=True)
        plt.title(f"{metric.capitalize()} (R vs cell count)", fontweight='bold')
        plt.xlabel('Cell count N', fontweight='bold')
        plt.ylabel('Resolution R', fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + f"sensitivity_heatmap_{metric}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: sensitivity_heatmap_{metric}.png")

def plot_scatter(df):
    """Scatter plots with error bands for each metric"""
    metrics = ['pearson', 'spearman', 'l1', 'recall']
    ylabels = ["Pearson's r", "Spearman's ρ", 'L1 distance', 'Recall (top 1%)']
    for metric, ylabel in zip(metrics, ylabels):
        fig, ax = plt.subplots(figsize=(6, 4))
        for N in CELL_SIZES:
            sub = df[df['cells'] == N]
            grouped = sub.groupby('R')[metric].agg(['mean', 'std']).reset_index()
            ax.errorbar(grouped['R'], grouped['mean'], yerr=grouped['std'],
                        marker='o', capsize=3, color=CELL_COLORS[N],
                        label=f'N = {N}', linewidth=1.5, linestyle='-')
        ax.set_xscale('log')
        ax.set_xlabel('Resolution R', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'Influence of R on {ylabel}', fontweight='bold', loc='left', fontsize=12)
        ax.legend(frameon=False, loc='best')
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + f"sensitivity_scatter_{metric}.png", bbox_inches='tight')
        plt.close()
        print(f"Saved: sensitivity_scatter_{metric}.png")

# ==================== Main entry point ====================
if __name__ == "__main__":
    # Run experiment or load existing results
    if not os.path.exists(OUTPUT_CSV):
        df = run_experiment()
    else:
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Loaded existing data from {OUTPUT_CSV}")

    set_style()
    plot_time(df)
    plot_combined_lines(df)
    plot_boxplots(df)
    plot_heatmaps(df)
    plot_scatter(df)

    print("\nAll sensitivity plots (5 repeats, new color scheme) generated successfully!")