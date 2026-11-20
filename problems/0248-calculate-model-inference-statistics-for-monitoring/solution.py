def calculate_inference_stats(latencies_ms: list) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: list of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """
    if not latencies_ms:
        return {}

    n = len(latencies_ms)
    avg_latency = sum(latencies_ms) / n

    throughput = 1000 / avg_latency

    sorted_data = sorted(latencies_ms)

    def percentile(p):
        pos = (n - 1) * p
        lower = int(pos)
        upper = min(lower + 1, n - 1)
        weight = pos - lower
        return (
            sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
        )
        
    return {
        
        "throughput_per_sec": round(throughput, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_ms": round(percentile(0.50), 2),
        "p95_ms": round(percentile(0.95), 2),
        "p99_ms": round(percentile(0.99), 2)
    }
    



