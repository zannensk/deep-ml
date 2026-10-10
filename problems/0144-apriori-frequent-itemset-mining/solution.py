import itertools
from collections import defaultdict

def apriori(transactions, min_support=0.5, max_length=None):
    """
    Returns: dict mapping frozenset(itemset) -> support (float)
    """
    # TODO: Implement the Apriori algorithm
    if not transactions:
        raise ValueError("transactions must not be empty")
    
    if max_length is not None and max_length < 1:
        return {}
    
    transactions = [set(t) for t in transactions]
    n = len(transactions)

    item_counts = defaultdict(int)
    result = {}

    current_frequent = set()
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    
    for item, count in item_counts.items():
        support = count / n
        if support >= min_support:
            itemset = frozenset([item])
            current_frequent.add(itemset)
            result[itemset] = support

    k = 2

    while current_frequent and (
        max_length is None or k <= max_length
    ):
        candidates = set()
        current_list = list(current_frequent)

        for i in range(len(current_list)):
            for j in range(i+1, len(current_list)):
                candidate = current_list[i] | current_list[j]
                if len(candidate) != k:
                    continue
                
                all_subset_frequent = all(
                    frozenset(subset) in current_frequent
                    for subset in itertools.combinations(candidate, k-1)
                )
                if all_subset_frequent:
                    candidates.add(candidate)
        if not candidates:
            break
        
        candidate_count = defaultdict(int)
        next_frequent = set()
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_count[candidate] += 1
        
        for candidate in candidates:
            support = candidate_count[candidate] / n
            if support >= min_support:
                next_frequent.add(candidate)
                result[candidate] = support
        
        current_frequent = next_frequent
        k += 1
    
    return result
