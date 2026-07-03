"""
FastCSN: Precomputed expression grid for fast cell‑specific network queries.

Core idea:
1. Partition each gene's expression range into R intervals.
2. For each gene pair (i,j), build an R×R grid.
3. For each cell c, compute its oCSN Z‑value, then aggregate it into the grid
   bin (p,q) corresponding to its expression levels, summing and counting.
4. Each grid cell stores the average Z‑value of all cells that fall into that bin.
5. At query time, for any cell, look up the precomputed Z‑value directly.

Advantage:
- Precomputation is done once (requiring one run of oCSN); subsequent queries are extremely fast.
- As R increases, the approximation converges to the exact oCSN.
"""

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from ocsn_official import build_ocsn   # ensure ocsn_official.py is in the path

class FastCSN:
    def __init__(self, resolution=100, q=0.1):
        """
        Parameters
        ----------
        resolution : int
            Number of intervals per gene, i.e., grid side length R.
        q : float
            oCSN window parameter (boxsize), passed to build_ocsn.
        """
        self.R = resolution
        self.q = q
        self.grid_cache = {}          # {(i,j): ndarray of shape (R,R)}
        self.full_data = None
        self.gene_pairs = None
        self.N_full = None
        self.gene_min = None
        self.gene_max = None
        self.gene_step = None

    def precompute_on_full_data(self, X_full, gene_pairs):
        """
        Precompute the grids for the entire dataset.

        Parameters
        ----------
        X_full : np.ndarray, shape (n_genes, n_cells)
            Expression matrix (genes × cells).
        gene_pairs : list of tuple (int, int)
            List of gene pairs to precompute.
        """
        print(f"Precomputing FastCSN on full data: {X_full.shape[0]} genes × {X_full.shape[1]} cells")
        self.full_data = X_full
        self.gene_pairs = gene_pairs
        self.N_full = X_full.shape[1]
        n_genes, N = X_full.shape

        # ---------- 1. Expression range binning ----------
        self.gene_min = np.min(X_full, axis=1)          # min expression per gene
        self.gene_max = np.max(X_full, axis=1)          # max expression per gene
        self.gene_step = (self.gene_max - self.gene_min) / self.R   # bin width
        # Handle constant genes (min == max -> step == 0) by setting step = 1.0
        self.gene_step[self.gene_step == 0] = 1.0

        # ---------- 2. Run oCSN once on the full dataset to get Z‑values ----------
        print("  Running oCSN on full data (one-time cost)...")
        nets_ocsn = build_ocsn(X_full, gene_pairs, q=self.q)   # list of csr_matrix, length = N

        # ---------- 3. Aggregate Z‑values by expression bins ----------
        for i, j in tqdm(gene_pairs, desc='FastCSN precompute'):
            # Skip gene pairs that are all zero (oCSN would return empty, but handle safely)
            if np.all(X_full[i] == 0) or np.all(X_full[j] == 0):
                self.grid_cache[(i, j)] = np.zeros((self.R, self.R))
                continue

            grid_sum = np.zeros((self.R, self.R))
            grid_cnt = np.zeros((self.R, self.R), dtype=int)

            # Extract expression vectors for this pair (avoid repeated indexing)
            exp_i = X_full[i]
            exp_j = X_full[j]

            for c in range(N):
                # Get the Z‑value for this cell and edge (0 if absent)
                z = nets_ocsn[c][i, j] if nets_ocsn[c][i, j] != 0 else 0.0

                # Compute bin indices (constant genes with step=1 always yield index 0)
                p = int((exp_i[c] - self.gene_min[i]) / self.gene_step[i])
                p = max(0, min(self.R - 1, p))
                q = int((exp_j[c] - self.gene_min[j]) / self.gene_step[j])
                q = max(0, min(self.R - 1, q))

                grid_sum[p, q] += z
                grid_cnt[p, q] += 1

            # Compute the average (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                grid = np.divide(grid_sum, grid_cnt,
                                 out=np.zeros_like(grid_sum),
                                 where=grid_cnt > 0)
            self.grid_cache[(i, j)] = grid

        return self

    def build_networks_for_subset(self, cell_indices):
        """
        Quickly build networks for a subset of cells (lookup from precomputed grids).

        Parameters
        ----------
        cell_indices : list of int
            Column indices (0‑based) of the cells to build networks for.

        Returns
        -------
        networks : list of csr_matrix
            Network (genes × genes) for each cell, as a sparse matrix.
        """
        n_genes = self.full_data.shape[0]
        R = self.R
        networks = []

        for c in cell_indices:
            rows, cols, vals = [], [], []
            for i, j in self.gene_pairs:
                # Retrieve expression values
                x = self.full_data[i, c]
                y = self.full_data[j, c]

                # Compute bin indices (constant genes step=1 prevents NaN)
                p = int((x - self.gene_min[i]) / self.gene_step[i])
                p = max(0, min(R - 1, p))
                q = int((y - self.gene_min[j]) / self.gene_step[j])
                q = max(0, min(R - 1, q))

                z = self.grid_cache[(i, j)][p, q]
                if z != 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(z)

            # Build sparse matrix
            mat = csr_matrix((vals, (rows, cols)), shape=(n_genes, n_genes))
            # Symmetrize (network is undirected)
            mat = mat + mat.T
            networks.append(mat)

        return networks