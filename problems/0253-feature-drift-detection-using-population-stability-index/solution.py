import numpy as np

def detect_feature_drift(reference_data: list, production_data: list, num_bins: int = 10) -> dict:
    """
    Detect feature drift using Population Stability Index (PSI).
    
    Args:
        reference_data: List of feature values from reference distribution (e.g., training)
        production_data: List of feature values from production distribution
        num_bins: Number of bins for histogram comparison
    
    Returns:
        dict with 'psi', 'drift_detected', and 'drift_level'
    """
    if not reference_data or not production_data:
        return {}
    ref = np.asarray(reference_data, dtype=float)
    prd = np.asarray(production_data, dtype=float)

    edges = np.histogram_bin_edges(
        np.concatenate([ref, prd]),
        bins=num_bins
    )
    ref_count, _ = np.histogram(ref, bins=edges)
    prd_count, _ = np.histogram(prd, bins=edges)

    q = ref_count / len(ref)
    p = prd_count / len(prd)
    eps = 0.0001
    q = np.where(q==0, eps, q)
    p = np.where(p==0, eps, p)

    psi = float(np.sum((q - p) * np.log(q / p)))

    if psi < 0.1:
        level = "none"
    elif psi < 0.25:
        level = "moderate"
    else:
        level = "significant"
    
    return {
        "psi": round(psi, 4),
        "drift_detected": bool(psi >= 0.1),
        "drift_level": level
    }


