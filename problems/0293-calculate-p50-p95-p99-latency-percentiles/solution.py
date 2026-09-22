import numpy as np

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    # Your code here
    if not latencies:
        return {
            "P50": 0.0,
            "P95": 0.0,
            "P99": 0.0
        }
    
    return {
        "P50": round(float(np.percentile(latencies, 50)), 4),
        "P95": round(float(np.percentile(latencies, 95)), 4),
        "P99": round(float(np.percentile(latencies, 99)), 4),
    }