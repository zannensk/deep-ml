import numpy as np

def contrastive_loss(embeddings: np.ndarray, temperature: float) -> float:
    """
    Compute the NT-Xent (SimCLR-style) contrastive loss.
    
    Args:
        embeddings: Array of shape (2N, d) where consecutive pairs
                    (2i, 2i+1) are positive pairs.
        temperature: Temperature scaling parameter (tau > 0).
    
    Returns:
        The mean contrastive loss as a float.
    """
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norm, 1e-12, None)
    similarities = embeddings @ embeddings.T

    losses = []

    n = embeddings.shape[0]
    logits = similarities / temperature

    for i in range(n):
        if i % 2 == 0:
            positive_idx = i + 1
        else:
            positive_idx = i - 1
        
        positive_logit = logits[i, positive_idx]
        mask = np.arange(n) != i
        valid_logits = logits[i, mask]

        m = np.max(valid_logits)
        log_sum_exp = m + np.log(
            np.sum(np.exp(valid_logits - m))
        )
        loss_i = -positive_logit + log_sum_exp
        losses.append(loss_i)
    
    return float(np.mean(losses))