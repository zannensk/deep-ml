import math


def dcg_at_k(rels, k):
    # TODO: sum of rels[i] / log2(i + 2) over the first k entries
    return sum(
        (
            rels[i] / math.log2(i + 2)
            for i in range(min(k, len(rels)))
        ),
        0.0
    )

def ndcg_at_k(rels, k):
    # TODO: dcg / ideal dcg (rels sorted descending), 0.0 if ideal is 0, rounded to 4 decimals
    dcg = dcg_at_k(rels, k)
    ideal_rels = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)

def mrr(ranked_lists):
    # TODO: mean over queries of 1 / (rank of first relevant), 0 if none, rounded to 4 decimals
    if not ranked_lists:
        return 0.0
    total = 0.0
    for rels in ranked_lists:
        rr = 0.0
        for i, rel in enumerate(rels):
            if rel > 0:
                rr = 1 / (i + 1)
                break
        total += rr
    return round(total / len(ranked_lists), 4)

def map_at_k(ranked_lists, k):
    # TODO: mean over queries of average precision at k (denominator min(k, R)), rounded to 4 decimals
    if not ranked_lists:
        return 0.0
    
    total_ap = 0.0
    for rels in ranked_lists:
        R = sum(rel > 0 for rel in rels)
        if R == 0:
            continue
        precision_sum = 0.0
        rels_so_far = 0
        for i, rel in enumerate(rels[:k]):
            if rel > 0:
                rels_so_far += 1
                precision_at_i = rels_so_far / (i + 1)
                precision_sum += precision_at_i
        ap = precision_sum / min(k, R)
        total_ap += ap
    return round(total_ap / len(ranked_lists), 4)
