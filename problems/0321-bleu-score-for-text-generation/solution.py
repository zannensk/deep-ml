import numpy as np
from collections import Counter
import math

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences.
    
    Args:
        candidate: List of tokens in the candidate sentence
        references: List of reference sentences, each as a list of tokens
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        BLEU score between 0 and 1
    """
    # Your code here
    if not candidate or not references or max_n < 1:
        return 0.0
    c = len(candidate)
    r = min(
        (len(ref) for ref in references),
        key=lambda x: (abs(x - c), x)
    )

    bp = min(1.0, np.exp(1 - r / c))
    log_precision = 0.0

    for n in range(1, max_n + 1):
        cand_count = Counter(
            tuple(candidate[i: i+n])
            for i in range(c - n + 1)
        )
        total = sum(cand_count.values())
        if total == 0:
            return 0.0
        max_ref_count = Counter()
        for ref in references:
            ref_cand = Counter(
                tuple(ref[i: i+n])
                for i in range(len(ref) - n + 1)
            )

            for gram, count in ref_cand.items():
                max_ref_count[gram] = max(
                    max_ref_count[gram], count
                )
        clipped = sum(
            min(count, max_ref_count[gram])
            for gram, count in cand_count.items()
        )
        if clipped == 0:
            return 0.0

        precision = clipped / total
        log_precision += math.log(precision)
    return float(bp * math.exp(log_precision / max_n))
