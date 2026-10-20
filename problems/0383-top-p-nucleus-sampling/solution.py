import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    if not logits:
        return []

    x = np.asarray(logits, dtype=float)
    exp_x = np.exp(x - np.max(x))
    probs = exp_x / np.sum(exp_x)

    order = np.lexsort((np.arange(len(probs)), -probs))
    cum_probs = np.cumsum(probs[order])

    cutoff = np.searchsorted(cum_probs, p, side="left") + 1
    kept_indices = order[:cutoff]
    filtered = np.zeros_like(probs)
    filtered[kept_indices] = probs[kept_indices]

    filtered /= np.sum(filtered)
    return np.round(filtered, 4).tolist()


