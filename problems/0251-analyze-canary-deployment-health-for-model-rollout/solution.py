def analyze_canary_deployment(canary_results: list, baseline_results: list, accuracy_tolerance: float = 0.05, latency_tolerance: float = 0.10) -> dict:
    """
    Analyze canary deployment health metrics for model rollout decision.
    
    Args:
        canary_results: list of prediction results from canary (new) model
                       Each dict has 'latency_ms', 'prediction', 'ground_truth'
        baseline_results: list of prediction results from baseline (existing) model
                         Each dict has 'latency_ms', 'prediction', 'ground_truth'
        accuracy_tolerance: max acceptable relative accuracy degradation (0.05 = 5%)
        latency_tolerance: max acceptable relative latency increase (0.10 = 10%)
    
    Returns:
        dict with canary/baseline metrics and promotion recommendation
    """
    if not canary_results or not baseline_results:
        return {}

    def get_metrics(result):
        n = len(result)
        accuracy = sum(
            r["prediction"] == r["ground_truth"]
            for r in result
        ) / n
        latency = sum(
            r["latency_ms"] for r in result
        ) / n
        return round(accuracy, 4), round(latency, 4)
    
    can_acc, can_lat = get_metrics(canary_results)
    bas_acc, bas_lat = get_metrics(baseline_results)

    def pct_change(cur, bas):
        if bas == 0:
            return 0.0 if cur == 0 else float('inf')
        else:
            return round((cur - bas) / bas * 100, 2)
    
    acc_change = pct_change(can_acc, bas_acc)
    lat_change = pct_change(can_lat, bas_lat)

    accuracy_ok = can_acc >= bas_acc * (1 - accuracy_tolerance)
    latency_ok = lat_change <= bas_lat * (1 + latency_tolerance)
    promote = accuracy_ok and latency_ok

    return {
        "canary_accuracy": can_acc,
        "baseline_accuracy": bas_acc,
        "accuracy_change_pct": acc_change,
        "canary_avg_latency": can_lat,
        "baseline_avg_latency": bas_lat,
        "latency_change_pct": lat_change,
        "promote_recommended": bool(promote)
    }


