import numpy as np
from collections import deque

def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Implement DBSCAN clustering algorithm.
    
    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features)
    - eps: Maximum distance between two samples to be considered neighbors
    - min_samples: Minimum number of samples in a neighborhood for a core point
    
    Returns:
    - labels: 1D numpy array of cluster labels (-1 for noise points)
    """
    # Your code here
    n = len(X)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)

    cluster_id = 0

    def get_neighbors(i):
        distances = np.linalg.norm(X - X[i], axis=1)
        return np.flatnonzero(distances <= eps)
    
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = get_neighbors(i)
        if len(neighbors) < min_samples:
            continue
        labels[i] = cluster_id
        queue = deque(neighbors.tolist())
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                new_neighbors = get_neighbors(j)
                if len(new_neighbors) >= min_samples:
                    queue.extend(new_neighbors.tolist())
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels

