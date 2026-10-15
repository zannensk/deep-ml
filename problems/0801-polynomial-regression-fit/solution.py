import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = np.vander(X, N=degree+1, increasing=True)
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta.tolist()
