import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    # Your code here
    X_minority = np.asarray(X_minority, dtype=float)
    n_samples, n_features = X_minority.shape
    k_actual = min(k, n_samples - 1)
    if k_actual == 0 or n_synthetic == 0:
        return np.empty((0, n_features))
    synthetic = []

    for _ in range(n_synthetic):
        i = np.random.randint(0, n_samples)
        x_i = X_minority[i]

        distances = np.linalg.norm(X_minority - x_i, axis=1)

        candidate_indices = np.arange(n_samples) != i
        other_indices = np.arange(n_samples)[candidate_indices]
        other_distances = distances[candidate_indices]

        nearest_order = np.argsort(other_distances)
        neighbor_indices = other_indices[nearest_order[:k_actual]]

        j = np.random.randint(0, k_actual)
        x_nn = X_minority[neighbor_indices[j]]

        gap = np.random.random()

        synthetic_i = x_i + gap * (x_nn - x_i)

        synthetic.append(synthetic_i)
    
    return np.array(synthetic)