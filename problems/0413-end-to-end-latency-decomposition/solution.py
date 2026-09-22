import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    stages = list(stage_latencies.keys())
    if not stages:
        return {
            "e2e_mean": 0.0,
            "e2e_percentiles": {},
            "stage_stats": {},
            "bottleneck": None,
            "stage_pct": {}
        }
    
    arrays = [np.asarray(stage_latencies[s], dtype=float) for s in stages]

    e2e = np.sum(np.vstack(arrays), axis=0)

    e2e_mean = round(float(np.mean(e2e)), 2)

    e2e_percentiles = {
        p: round(float(np.percentile(e2e, p)), 2)
        for p in percentiles
    }

    stage_stats = {}
    stage_means = {}

    for stage, array in zip(stages, arrays):
        mean = float(np.mean(array))
        std = float(np.std(array))

        stage_means[stage] = mean
        stage_stats[stage] = {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "percentiles": {
                p: round(float(np.percentile(array, p)), 2)
                for p in percentiles
            }
        }
    
    bottleneck = max(stage_means, key=stage_means.get)

    if e2e_mean == 0:
        stage_pct = {stage: 0.0 for stage in stages}
    else:
        stage_pct = {
            stage: round(mean / e2e_mean * 100, 2)
            for stage, mean in stage_means.items()
        }
    
    return {
        "e2e_mean": e2e_mean,
        "e2e_percentiles": e2e_percentiles,
        "stage_stats": stage_stats,
        "bottleneck": bottleneck,
        "stage_pct": stage_pct
    }

