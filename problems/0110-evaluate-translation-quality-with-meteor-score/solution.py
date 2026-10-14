import numpy as np
from collections import Counter

def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    """
    Calculate METEOR score for machine translation evaluation.
    
    Args:
        reference: Reference translation string
        candidate: Candidate translation string
        alpha: Weight for precision vs recall in F-mean (default 0.9)
        beta: Exponent for fragmentation penalty (default 3)
        gamma: Maximum penalty coefficient (default 0.5)
    
    Returns:
        METEOR score between 0 and 1
    """
    # Your code here
    ref = reference.lower().split()
    can = candidate.lower().split()

    if not ref or not can:
        return 0.0
    
    matches = []
    used = set()

    for i, word in enumerate(can):
        for j, token in enumerate(ref):
            if word == token and j not in used:
                matches.append((i, j))
                used.add(j)
                break
    
    m = len(matches)
    if m == 0:
        return 0.0
    
    precision = m / len(can)
    recall = m / len(ref)

    f_mean = (
        precision * recall
        / (alpha * precision + (1 - alpha) * recall)
    )
    chunks = 0
    for k, (i, j) in enumerate(matches):
        if (
            k == 0
            or i != matches[k-1][0] + 1
            or j != matches[k-1][1] + 1
        ):
            chunks += 1
    
    penalty = gamma * (chunks / m) ** beta

    return float(f_mean * (1 - penalty))
