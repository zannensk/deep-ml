def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    if not data:
        return []

    count = {}
    for sample, label in data:
        count[label] = count.get(label, 0) + 1
    
    min_count = min(count.values())
    kept = {}
    result = []

    for sample, label in data:
        current = kept.get(label, 0)
        if current < min_count:
            kept[label] = current + 1
            result.append((sample, label))
    return result