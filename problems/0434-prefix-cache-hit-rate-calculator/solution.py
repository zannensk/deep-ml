def prefix_cache_hit_rate(prompts: list, cache: list) -> dict:
    """
    Calculate the prefix cache hit rate for a batch of tokenized prompts.

    Args:
        prompts: List of tokenized prompts (list of list of ints).
        cache: List of cached prefixes (list of list of ints).

    Returns:
        Dictionary with 'hit_rate', 'cached_tokens', and 'total_tokens'.
    """
    # Your code here
    root = {}
    END = object()

    for prefix in cache:
        node = root
        for token in prefix:
            node = node.setdefault(token, {})
        node[END] = True
    
    total_tokens = sum(len(prompt) for prompt in prompts)
    cached_tokens = 0
    for prompt in prompts:
        node = root
        longest = 0
        for i, token in enumerate(prompt, 1):
            if token not in node:
                break
            node = node[token]
            if END in node:
                longest = i
        cached_tokens += longest
    
    hit_rate = (
        cached_tokens / total_tokens
        if total_tokens > 0 else 0.0
    )

    return {
        "hit_rate": round(hit_rate, 4),
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens
    }

