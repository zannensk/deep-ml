import numpy as np

def rubric_llm_judge_evaluation(
    judge_scores: list[list[float]],
    criteria_weights: list[float],
    passing_threshold: float = 0.6,
    max_score: float = 5.0
) -> dict:
    """
    Evaluate LLM response using rubric-based multi-judge scoring.
    
    Args:
        judge_scores: 2D list where judge_scores[i][j] is judge i's score for criterion j
        criteria_weights: Weights for each criterion (should sum to 1)
        passing_threshold: Minimum normalized score to pass (0 to 1)
        max_score: Maximum possible score for each criterion
    
    Returns:
        Dictionary with evaluation results
    """
    scores = np.asarray(judge_scores, dtype=float)
    weight = np.asarray(criteria_weights, dtype=float)

    means = np.mean(scores, axis=0)
    weighted = float(np.sum(means * weight))
    normalized = weighted / max_score
    passed = bool(normalized >= passing_threshold)

    if scores.shape[0] == 1:
        agreement = 1.0
    else:
        std = np.std(scores, axis=0, ddof=0)
        avg_std = float(np.mean(std))
        agreement = max(
            0.0, 1 - avg_std / (max_score / 2)
        )

    return {
        "criterion_scores": [
            round(float(v), 4) for v in means
        ],
        "weighted_score": round(weighted, 4),
        "normalized_score": round(normalized, 4),
        "pass_status": passed,
        "judge_agreement": round(float(agreement), 4)
    }






