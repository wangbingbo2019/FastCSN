import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ocsn_official import build_ocsn


def generate_metacells(X, target_size=20, k_neighbors=100):
    """
    Generate metacells according to the Metacell algorithm (Baran et al. 2019).

    Parameters:
        X: numpy array (genes, cells) - raw UMI counts
        target_size: target number of cells per metacell (default 20)
        k_neighbors: number of nearest neighbors for balanced KNN graph

    Returns:
        meta_expr: numpy array (genes, n_metacells) - mean expression of each metacell
        cell_to_meta: numpy array (n_cells,) - metacell assignment for each cell
    """
    n_genes, n_cells = X.shape

    # Step 1: Transpose to (cells, genes) for PCA
    X_cells = X.T.copy()

    # Step 2: Filter constant genes (zero variance)
    gene_vars = X_cells.var(axis=0)
    if gene_vars.sum() == 0:
        # All genes constant, return each cell as its own metacell
        return X.copy(), np.arange(n_cells)

    X_cells = X_cells[:, gene_vars > 1e-12]

    # Step 3: Standardize (z-score normalization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cells)

    # Step 4: PCA dimensionality reduction
    n_pcs = min(30, n_cells - 1, X_scaled.shape[1] - 1)
    n_pcs = max(2, n_pcs)
    pca = PCA(n_components=n_pcs, random_state=0)
    emb = pca.fit_transform(X_scaled)

    # Step 5: Balanced KNN graph (each cell connects to k nearest neighbors)
    k = min(k_neighbors, n_cells - 1)
    if k < 1:
        k = 1
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(emb)
    adj = nn.kneighbors_graph(mode="connectivity")
    adj = (adj + adj.T) > 0  # Symmetrize
    adj = adj.astype(np.float32)

    # Step 6: Graph cover partition (greedy algorithm)
    clusters = graph_cover_partition(adj, target_size=target_size)

    # Step 7: Assign cells to metacells
    cell_to_meta = np.full(n_cells, -1, dtype=int)
    for i, cl in enumerate(clusters):
        cell_to_meta[cl] = i

    # Handle unassigned cells
    unassigned = np.where(cell_to_meta == -1)[0]
    for u in unassigned:
        cell_to_meta[u] = len(clusters)
        clusters.append(np.array([u]))

    n_meta = len(clusters)

    # Step 8: Compute metacell expression as MEAN of all cells in cluster
    # (Paper: "Expression of a metacell is defined as the mean of the cells")
    meta_expr = np.zeros((n_genes, n_meta), dtype=np.float32)
    for m in range(n_meta):
        cells_in_mc = clusters[m]
        # FULL MEAN (including zeros) - exactly as stated in the paper
        meta_expr[:, m] = X[:, cells_in_mc].mean(axis=1)

    # Step 9: Filter metacells that are too small (< 3 cells)
    cluster_sizes = np.array([len(c) for c in clusters])
    small_mask = cluster_sizes < 3
    if small_mask.any():
        # Merge small metacells into nearest large metacell
        large_centers = meta_expr[:, ~small_mask].T
        for small_idx in np.where(small_mask)[0]:
            small_center = meta_expr[:, small_idx]
            dists = np.linalg.norm(large_centers - small_center, axis=1)
            nearest = np.argmin(dists)
            # Reassign cells from small metacell
            cell_to_meta[clusters[small_idx]] = np.where(~small_mask)[0][nearest]

        # Recompute metacell expression after merging
        n_meta = (~small_mask).sum()
        new_clusters = [clusters[i] for i in np.where(~small_mask)[0]]
        meta_expr = np.zeros((n_genes, n_meta), dtype=np.float32)
        for i, cells_in_mc in enumerate(new_clusters):
            meta_expr[:, i] = X[:, cells_in_mc].mean(axis=1)

        # Remap cell_to_meta indices
        old_to_new = {old: new for new, old in enumerate(np.where(~small_mask)[0])}
        cell_to_meta = np.array([old_to_new.get(c, -1) for c in cell_to_meta])

    print(f"[MetaCell] {n_cells} cells -> {meta_expr.shape[1]} metacells (k={k_neighbors})")
    return meta_expr, cell_to_meta


def graph_cover_partition(adj, target_size=20):
    """
    Greedy graph cover partition algorithm.
    Starting from the highest-degree node, iteratively cover the graph
    with disjoint clusters of approximately target_size cells.
    """
    n_cells = adj.shape[0]
    visited = np.zeros(n_cells, dtype=bool)
    clusters = []
    remaining = np.arange(n_cells)

    while len(remaining) > 0:
        # Pick the cell with highest degree among remaining
        degrees = np.array(adj[remaining, :].sum(axis=1)).ravel()
        seed_idx = remaining[np.argmax(degrees)]

        # Get its neighbors
        neigh = np.where(adj[seed_idx].toarray().ravel() > 0)[0]
        cand = np.unique(np.concatenate([[seed_idx], neigh]))
        cand = cand[~visited[cand]]

        # Limit to target size
        if len(cand) > target_size:
            cand = cand[:target_size]

        clusters.append(cand)
        visited[cand] = True
        remaining = remaining[~visited[remaining]]

    return clusters


def build_networks(data, gene_pairs=None, q=0.1, target_size=20, k_neighbors=100):
    """
    locCSN: Metacell aggregation + oCSN network construction.

    Steps:
    1. Generate metacells (disjoint homogeneous clusters)
    2. Build oCSN networks on metacells
    3. Map metacell networks back to individual cells

    Parameters:
        data: numpy array (genes, cells) - raw UMI counts
        gene_pairs: list of (i,j) tuples
        q: quantile window size
        target_size: target cells per metacell
        k_neighbors: KNN neighbors

    Returns:
        list of csr_matrix, one network per cell
    """
    t_start = time.time()

    # Step 1: Generate metacells
    meta_expr, cell_to_meta = generate_metacells(
        data, target_size=target_size, k_neighbors=k_neighbors
    )

    # Step 2: Build networks on metacells using oCSN
    # (Paper: "For each metacell, we use locCSN to compute gene networks")
    meta_nets = build_ocsn(meta_expr, gene_pairs=gene_pairs, q=q)

    # Step 3: Map back to single cells
    # Each original cell inherits the network of its metacell
    single_cell_nets = [meta_nets[cell_to_meta[c]] for c in range(data.shape[1])]

    elapsed = time.time() - t_start
    print(f"  locCSN total time (clustering + network): {elapsed:.2f}s")
    return single_cell_nets