def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    ai = flops / bytes_accessed
    ridge = peak_performance / peak_bandwidth
    bottleneck = (
        "memory-bound" if ai < ridge
        else "compute-bound"
    )
    achieved = min(
        peak_performance,
        ai * peak_bandwidth
    )

    utilization = achieved / peak_performance * 100
    
    return {
        "arithmetic_intensity": round(ai, 4),
        "ridge_point": round(ridge, 4),
        "bottleneck": bottleneck,
        "achieved_performance": round(achieved, 4),
        "utilization_percent": round(utilization, 2)
    }


