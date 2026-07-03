"""
Experiment 1: Accuracy comparison (corrected version)
- Includes locCSN, unified colors, corrected recall
Metrics: Pearson r, Spearman r, L1 distance, recall of significant edges (top 1%)
"""

import time
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from ocsn_official import build_ocsn, build_pearson, build_spearman, build_nmi
from loccsn_metacell import build_networks as build_loccsn_full   # locCSN
from fastcsn import FastCSN

# ==================== Configuration ====================
DATA_PATH = r"F:\metacell_analysis\processed_data\COVID_Lung\pulmonary_alveolar_type_2_cell\counts_200genes.mtx"
GENE_PAIRS = 500
Q = 0.1
R_FASTCSN = 100
CELL_SIZES = [1000, 5000, 10000]
N_REPEATS = 3
TOP_PERCENT = 0.01
RANDOM_SEED = 42

OUTPUT_CSV = "accuracy_results_corrected.csv"
OUTPUT_FIG_PREFIX = "accuracy_corrected"

np.random.seed(RANDOM_SEED)

def load_data(path):
    print(f"Loading data from {path}...")
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
    pearson_list = []
    spearman_list = []
    l1_list = []
    recall_list = []
    for c in range(n_cells):
        v_test = weights_test[c]
        v_ocsn = weights_ocsn[c]
        # Skip if either vector has zero variance
        if np.std(v_test) > 0 and np.std(v_ocsn) > 0:
            pearson_list.append(pearsonr(v_test, v_ocsn)[0])
            spearman_list.append(spearmanr(v_test, v_ocsn)[0])
        else:
            pearson_list.append(np.nan)
            spearman_list.append(np.nan)
        # L1 distance
        l1_list.append(np.mean(np.abs(v_test - v_ocsn)))
        # Recall: top_k edges from oCSN covered by test method
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

def main():
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
    print(f"Using {len(gene_pairs)} fixed gene pairs.\n")

    records = []

    for N in CELL_SIZES:
        if N > total_cells:
            print(f"Skipping N={N} > total_cells")
            continue
        print(f"\n======= Cell count: {N} =======")
        for rep in range(N_REPEATS):
            print(f"  Replicate {rep+1}/{N_REPEATS}")
            idx = np.random.choice(total_cells, size=N, replace=False)
            X_sub = data_full[:, idx]

            # oCSN baseline
            nets_ocsn = build_ocsn(X_sub, gene_pairs, q=Q)
            weights_ocsn = get_network_weights(nets_ocsn, gene_pairs, N)
            top_k = max(1, int(TOP_PERCENT * GENE_PAIRS))

            # ---------- FastCSN ----------
            fcsn = FastCSN(resolution=R_FASTCSN, q=Q)
            start = time.time()
            fcsn.precompute_on_full_data(X_sub, gene_pairs)
            time_fit = time.time() - start
            start = time.time()
            nets_fast = fcsn.build_networks_for_subset(list(range(N)))
            time_query = time.time() - start
            time_fast = time_fit + time_query
            weights_fast = get_network_weights(nets_fast, gene_pairs, N)
            metrics = evaluate_vs_ocsn(weights_fast, weights_ocsn, top_k)
            records.append({
                'method': 'FastCSN', 'cells': N, 'rep': rep,
                'time': time_fast,
                'pearson': metrics['pearson'], 'spearman': metrics['spearman'],
                'l1': metrics['l1'], 'recall': metrics['recall']
            })

            # ---------- locCSN ----------
            start = time.time()
            nets_loc = build_loccsn_full(X_sub, gene_pairs, q=Q, target_size=20, k_neighbors=100)
            time_loc = time.time() - start
            weights_loc = get_network_weights(nets_loc, gene_pairs, N)
            metrics = evaluate_vs_ocsn(weights_loc, weights_ocsn, top_k)
            records.append({
                'method': 'locCSN', 'cells': N, 'rep': rep,
                'time': time_loc,
                'pearson': metrics['pearson'], 'spearman': metrics['spearman'],
                'l1': metrics['l1'], 'recall': metrics['recall']
            })

            # ---------- PearsonCSN ----------
            start = time.time()
            nets_pearson = build_pearson(X_sub, gene_pairs, q=Q, thresh=0)
            time_pearson = time.time() - start
            weights_pearson = get_network_weights(nets_pearson, gene_pairs, N)
            metrics = evaluate_vs_ocsn(weights_pearson, weights_ocsn, top_k)
            records.append({
                'method': 'PearsonCSN', 'cells': N, 'rep': rep,
                'time': time_pearson,
                'pearson': metrics['pearson'], 'spearman': metrics['spearman'],
                'l1': metrics['l1'], 'recall': metrics['recall']
            })

            # ---------- SpearmanCSN ----------
            start = time.time()
            nets_spearman = build_spearman(X_sub, gene_pairs, q=Q, thresh=0)
            time_spearman = time.time() - start
            weights_spearman = get_network_weights(nets_spearman, gene_pairs, N)
            metrics = evaluate_vs_ocsn(weights_spearman, weights_ocsn, top_k)
            records.append({
                'method': 'SpearmanCSN', 'cells': N, 'rep': rep,
                'time': time_spearman,
                'pearson': metrics['pearson'], 'spearman': metrics['spearman'],
                'l1': metrics['l1'], 'recall': metrics['recall']
            })

            # ---------- NMICSN ----------
            start = time.time()
            nets_nmi = build_nmi(X_sub, gene_pairs, q=Q, thresh=0)
            time_nmi = time.time() - start
            weights_nmi = get_network_weights(nets_nmi, gene_pairs, N)
            metrics = evaluate_vs_ocsn(weights_nmi, weights_ocsn, top_k)
            records.append({
                'method': 'NMICSN', 'cells': N, 'rep': rep,
                'time': time_nmi,
                'pearson': metrics['pearson'], 'spearman': metrics['spearman'],
                'l1': metrics['l1'], 'recall': metrics['recall']
            })

            # Print intermediate results
            print(f"    FastCSN    : time={time_fast:.2f}s, pearson={records[-5]['pearson']:.3f}, recall={records[-5]['recall']:.3f}")
            print(f"    locCSN     : time={time_loc:.2f}s, pearson={records[-4]['pearson']:.3f}, recall={records[-4]['recall']:.3f}")
            print(f"    PearsonCSN : time={time_pearson:.2f}s, pearson={records[-3]['pearson']:.3f}, recall={records[-3]['recall']:.3f}")
            print(f"    SpearmanCSN: time={time_spearman:.2f}s, spearman={records[-2]['spearman']:.3f}, recall={records[-2]['recall']:.3f}")
            print(f"    NMICSN     : time={time_nmi:.2f}s, pearson={records[-1]['pearson']:.3f}, recall={records[-1]['recall']:.3f}")

    # Save results
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # ==================== Visualization (publication‑quality style) ====================
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })
    sns.set_style("ticks")

    # Color scheme: FastCSN purple, locCSN yellow, others gray
    colors = {
        'FastCSN': '#AA4499',   # purple
        'locCSN': '#F0E442',    # yellow
        'PearsonCSN': '#8f9092',# gray
        'SpearmanCSN': '#8f9092',
        'NMICSN': '#8f9092',
    }
    methods_order = ['FastCSN', 'locCSN', 'PearsonCSN', 'SpearmanCSN', 'NMICSN']
    metrics = ['time', 'pearson', 'spearman', 'l1', 'recall']
    ylabels = ['Time (seconds)', "Pearson's r", "Spearman's ρ", 'L1 distance', 'Recall (top 1%)']

    for metric, ylabel in zip(metrics, ylabels):
        fig, axes = plt.subplots(1, len(CELL_SIZES), figsize=(8, 4),
                                 constrained_layout=True, sharey=False)
        if len(CELL_SIZES) == 1:
            axes = [axes]
        for ax, N in zip(axes, CELL_SIZES):
            sub = df[df['cells'] == N]
            # Boxplot: no outliers, custom colors
            sns.boxplot(data=sub, x='method', y=metric, order=methods_order,
                        palette=colors, ax=ax, showfliers=False, width=0.6,
                        boxprops=dict(alpha=0.7))
            # Overlay jittered points
            sns.stripplot(data=sub, x='method', y=metric, order=methods_order,
                          color='black', size=2, alpha=0.5, ax=ax)
            ax.set_title(f'N = {N} cells', fontweight='bold', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, axis='y')
        # Overall figure title
        fig.suptitle(f'Comparison with oCSN – {ylabel}', fontweight='bold', y=1.02, fontsize=11)
        plt.savefig(f"{OUTPUT_FIG_PREFIX}_{metric}.png", bbox_inches='tight')
        plt.close()
        print(f"Saved figure: {OUTPUT_FIG_PREFIX}_{metric}.png")

    print("\nExperiment 1 finished successfully.")

if __name__ == "__main__":
    main()