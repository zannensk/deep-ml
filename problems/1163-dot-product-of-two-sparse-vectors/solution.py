def sparse_dot_product(vec1, vec2):
    # vec1 and vec2 are lists of numbers of equal length
    # return their dot product
    return sum(
        a * b
        for a, b in zip(vec1, vec2)
        if a != 0 and b != 0
    )