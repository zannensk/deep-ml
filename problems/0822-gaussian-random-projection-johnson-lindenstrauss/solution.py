import numpy as np

def gaussian_random_projection(X: np.ndarray, n_components: int, seed: int = 0) -> np.ndarray:
    """
    Project X into a lower-dimensional space using a Gaussian random projection.

    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Target dimensionality
        seed: Random seed for reproducibility

    Returns:
        Projected matrix of shape (n_samples, n_components)
    """
    # Your code here
    n_features = X.shape[1]

    rng = np.random.RandomState(seed)
    R = rng.standard_normal(
        (n_features, n_components)
    ) / np.sqrt(n_components)

    return X @ R
