import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    """
    Perform one step of gradient boosting regression using a decision stump.
    
    Args:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target values (list), shape (n_samples,)
        current_predictions: Current ensemble predictions (list), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)
    
    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    current_predictions = np.asarray(current_predictions, dtype=float)

    residuals = y - current_predictions

    n_samples, n_features = X.shape

    best_mse = float("inf")
    best_predictions = None

    for feature in range(n_features):
        values = np.unique(X[:, feature])

        if len(values) < 2:
            continue
        thresholds = (values[1:] + values[:-1]) / 2

        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if not np.any(left_mask) or not np.any(right_mask):
                continue
            left_value = np.mean(residuals[left_mask])
            right_value = np.mean(residuals[right_mask])

            stump_prediction = np.where(
                left_mask,
                left_value,
                right_value
            )

            mse = np.mean(
                (residuals - stump_prediction) ** 2
            )

            if mse < best_mse:
                best_mse = mse
                best_predictions = stump_prediction
    
    if best_predictions is None:
        best_predictions = np.full(
            n_samples,
            np.mean(residuals)
        )
    
    updated = (
        current_predictions
        + learning_rate * best_predictions
    )

    return np.round(updated, 4).tolist()
