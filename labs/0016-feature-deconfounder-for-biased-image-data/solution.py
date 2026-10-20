import numpy as np
import torch
import torch.nn as nn

class FeatureDeconfounder(nn.Module):
    """
    Removes the linear influence of metadata/confounding variables from features.
    
    The key insight is that we can model features as:
        f = X @ beta + residual
    where X is the metadata and residual is the part orthogonal to metadata.
    
    For batch processing with precomputed statistics:
        - fit(): compute Sigma_inv = (X^T X + reg*I)^{-1} on full training metadata
        - transform(): compute beta = Sigma_inv @ X^T @ f, return residual = f - X @ beta
    """
    
    def __init__(self):
        super().__init__()
        self.Sigma_inv = None
        self.is_fitted = False
    
    def fit(self, metadata):
        """
        Precompute inverse covariance from training metadata.
        
        Args:
            metadata: Tensor of shape (N, K) - all training metadata
        
        Hints:
            - Compute Sigma = X^T @ X
            - Add small regularization (1e-5) for numerical stability
            - Store Sigma_inv for use in transform()
        """
        X = torch.as_tensor(
            metadata, dtype=torch.float32
        )
        K = X.shape[1]
        Sigma = X.T @ X
        I = torch.eye(
            K, dtype=X.dtype, device=X.device
        )
        Sigma = Sigma + 1e-5 * I
        with torch.no_grad():
            self.Sigma_inv = torch.linalg.inv(Sigma)
        self.is_fitted = True
        return self
    
    def transform(self, features, metadata):
        """
        Remove metadata influence from features.
        
        Args:
            features: Tensor of shape (M, D) - batch of features
            metadata: Tensor of shape (M, K) - corresponding metadata
        
        Returns:
            Residualized features of shape (M, D)
        
        Hints:
            - Compute: beta = Sigma_inv @ (X^T @ f)
            - Return residual: f - X @ beta
        """
        if not self.is_fitted:
            raise RuntimeError
        if isinstance(features, np.ndarray):
            f = torch.as_tensor(
                features, dtype=torch.float32
            )
        else:
            f = features
        X = torch.as_tensor(
            metadata, dtype=f.dtype, device=f.device
        )
        Sigma_inv = self.Sigma_inv
        beta = Sigma_inv @ (X.T @ f)
        residual = f - X @ beta
        return residual
