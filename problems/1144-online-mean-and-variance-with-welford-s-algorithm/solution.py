def welford_stats(data, ddof=1):
    # data: iterable of numbers
    # return (count, mean, variance)
    count = 0
    mean = 0.0
    M2 = 0.0

    for x in data:
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2

    if count == 0:
        return (0, 0.0, 0.0)
    
    if count - ddof <= 0:
        variance = 0.0
    
    else:
        variance = M2 / (count - ddof)
    
    return (count, mean, variance)