import numpy as np

def monitor_prediction_distribution(reference_preds: list, current_preds: list, n_bins: int = 10) -> dict:
    """
    Monitor prediction distribution changes between reference and current predictions.
    
    Args:
        reference_preds: List of reference prediction scores (floats between 0 and 1)
        current_preds: List of current prediction scores (floats between 0 and 1)
        n_bins: Number of bins for histogram comparison
    
    Returns:
        Dictionary with keys: 'mean_shift', 'std_ratio', 'js_divergence', 'drift_detected'
    """
    # Your code here
    if not reference_preds or not current_preds:
        return {}
    
    ref = np.asarray(reference_preds, dtype=float)
    cur = np.asarray(current_preds, dtype=float)

    mean_shift = float(np.mean(cur) - np.mean(ref))
    ref_std = np.std(ref)
    cur_std = np.std(cur)

    if ref_std == 0:
        std_ratio = 1.0 if cur_std == 0 else float('inf')
    else:
        std_ratio = float(cur_std / ref_std)

    edges = np.linspace(0, 1, n_bins + 1)
    ref_count, _ = np.histogram(ref, bins=edges)
    cur_count, _ = np.histogram(cur, bins=edges)

    p = (ref_count + 1) / (len(ref) + n_bins)
    q = (cur_count + 1) / (len(cur) + n_bins)
    m = (p + q) / 2

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js = float(0.5 * (kl_pm + kl_qm))

    return {
        "mean_shift": mean_shift,
        "std_ratio": std_ratio,
        "js_divergence": round(js, 4),
        "drift_detected": bool(js > 0.1)
    }







