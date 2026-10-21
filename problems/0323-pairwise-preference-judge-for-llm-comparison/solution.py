def pairwise_preference_judge(comparisons: list, criteria_weights: dict, tie_threshold: float) -> dict:
    """
    Analyze pairwise comparisons between LLM responses.
    
    Args:
        comparisons: List of comparison dicts with 'id', 'scores_a', 'scores_b'
        criteria_weights: Dict mapping criterion names to importance weights
        tie_threshold: Maximum difference to declare a tie
    
    Returns:
        Dict with 'results', 'win_rate_a', 'win_rate_b', 'tie_rate', 'avg_margin'
    """
    # Your code here
    if not comparisons:
        return {
            "results": [],
            "win_rate_a": 0.0,
            "win_rate_b": 0.0,
            "tie_rate": 0.0,
            "avg_margin": 0.0
        }
    
    total_weight = sum(criteria_weights.values())

    weights = {
        k: v / total_weight
        for k, v in criteria_weights.items()
    }
    results = []
    wins_a = wins_b = ties = 0
    total_margin = 0

    for item in comparisons:
        score_a = sum(
            item["scores_a"][k] * w
            for k, w in weights.items()
        )
        score_b = sum(
            item["scores_b"][k] * w
            for k, w in weights.items()
        )
        margin = abs(score_a - score_b)
        total_margin += margin

        if margin <= tie_threshold:
            winner = "tie"
            ties += 1
        elif score_a > score_b:
            winner = "A"
            wins_a += 1
        else:
            winner = "B"
            wins_b += 1
        results.append({
            "id": item["id"],
            "winner": winner,
            "margin": round(margin,4)
        })
    n = len(comparisons)

    return {
        "results": results,
        "win_rate_a": round(wins_a / n, 4),
        "win_rate_b": round(wins_b / n, 4),
        "tie_rate": round(ties / n, 4),
        "avg_margin": round(total_margin / n, 4)
    }
        










