import hashlib

def feature_hash(features, n_features):
    """
    Apply the (signed) feature hashing trick.

    Args:
        features: dict mapping feature name (str) -> value (float).
        n_features: int, size of the output vector.

    Returns:
        list of length n_features.
    """
    result = [0.0] * n_features

    for name, value in features.items():
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % n_features
        sign_hash = int(digest[8:16], 16)
        sign = 1 if sign_hash % 2 == 0 else -1
        result[index] += sign * value
    
    return result