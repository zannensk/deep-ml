import numpy as np

def xgboost_objective(gradients: np.ndarray, hessians: np.ndarray,
                      left_indices: np.ndarray, right_indices: np.ndarray,
                      lambda_reg: float = 1.0, gamma: float = 0.0) -> dict:
    """
    Calculate XGBoost objective function components for a potential split.
    
    Args:
        gradients: First-order gradients for each sample
        hessians: Second-order hessians for each sample
        left_indices: Indices of samples going to left child
        right_indices: Indices of samples going to right child
        lambda_reg: L2 regularization parameter
        gamma: Tree complexity penalty
        
    Returns:
        Dictionary with 'left_weight', 'right_weight', and 'gain'
    """
    G_L = np.sum(gradients[left_indices])
    G_R = np.sum(gradients[right_indices])

    H_L = np.sum(hessians[left_indices])
    H_R = np.sum(hessians[right_indices])

    G = G_L + G_R
    H = H_L + H_R

    left_weight = -G_L / (H_L + lambda_reg)
    right_weight = -G_R / (H_R + lambda_reg)

    gain = 0.5 * (
        G_L**2 / (H_L + lambda_reg)
        + G_R**2 / (H_R + lambda_reg)
        - G**2 / (H + lambda_reg)
    ) - gamma

    return {
        "left_weight": round(float(left_weight), 4),
        "right_weight": round(float(right_weight), 4),
        "gain": round(float(gain), 4)
    }