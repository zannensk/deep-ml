import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: List or array of true binary labels (0 or 1)
        y_prob: List or array of predicted probabilities for the positive class
        n_bins: Number of bins for grouping predictions (default: 10)
    
    Returns:
        float: ECE value rounded to 3 decimal places
    """
    # Your code here
    ece = 0.0
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == 0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob > lower) & (y_prob <= upper)
        
        count = np.sum(mask)
        if count == 0:
            continue
        
        weight = count / n
        confidence = np.mean(y_prob[mask])
        accuracy = np.mean(y_true[mask])

        ece += weight * abs(confidence - accuracy)
    
    return round(float(ece), 3)