import numpy as np

def gcn_layer(A: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Perform a single GCN layer forward pass.
    
    Args:
        A: Adjacency matrix of shape (N, N)
        X: Node feature matrix of shape (N, F_in)
        W: Weight matrix of shape (F_in, F_out)
        
    Returns:
        Output feature matrix of shape (N, F_out)
    """
    A_hat = A + np.eye(A.shape[0])
    degree = np.sum(A_hat, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))

    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    Z = A_norm @ X @ W

    return np.maximum(0, Z)