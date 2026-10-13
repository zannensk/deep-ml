import numpy as np

def isolation_forest(X: np.ndarray, n_trees: int, sample_size: int, random_state: int = 42) -> np.ndarray:
    """
    Implement Isolation Forest for anomaly detection.
    
    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features)
    - n_trees: Number of isolation trees to build
    - sample_size: Number of samples to use for building each tree
    - random_state: Random seed for reproducibility
    
    Returns:
    - scores: 1D numpy array of anomaly scores (higher = more anomalous)
    """
    # Your code here
    X = np.asarray(X, dtype=float)
    n_samples = len(X)
    if n_samples == 0:
        return np.array([])
    
    psi = min(n_samples, sample_size)
    if psi <= 1:
        return np.full(n_samples, 0.5)
    rng = np.random.RandomState(random_state)
    max_depth = int(np.ceil(np.log2(psi)))

    c = np.zeros(psi + 1)
    n = np.arange(2, psi + 1, dtype=float)
    c[2:] = (
        2 * (np.log(n - 1) + np.euler_gamma)
        - 2 * (n - 1) / n
    )

    def build_tree(data, depth=0):
        n = len(data)
        if n <= 1 or depth >= max_depth:
            return n
        feature = rng.choice(data.shape[1])
        values = data[:, feature]

        low, high = values.min(), values.max()
        if low == high:
            return n
        split = rng.uniform(low, high)
        mask = values < split

        if not np.any(mask) or np.all(mask):
            return n
        return (
            feature,
            split,
            build_tree(data[mask], depth + 1),
            build_tree(data[~mask], depth + 1)
        )

    total_paths = np.zeros(n_samples)
    def accumulate(tree, rows, depth=0):
        if len(rows) == 0:
            return
        if isinstance(tree, (int, np.integer)):
            total_paths[rows] += depth + c[tree]
            return
        feature, split, left, right = tree
        mask = X[rows, feature] < split
        accumulate(left, rows[mask], depth + 1)
        accumulate(right, rows[~mask], depth + 1)

    all_rows = np.arange(n_samples)
    for _ in range(n_trees):
        idx = rng.choice(
            n_samples, size=psi, replace=False
        )
        tree = build_tree(X[idx])
        accumulate(tree, all_rows)
    
    avg_paths = total_paths / n_trees
    scores = 2 ** (-avg_paths / c[psi])
    return np.round(scores, 4)

