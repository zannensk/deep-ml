def autoscale_simulate(rps_series: list, capacity_per_replica: int, min_replicas: int, max_replicas: int, scale_up_threshold: float, scale_down_threshold: float, cooldown_steps: int) -> dict:
    """
    Simulate autoscaling of model-serving replicas with SLA tracking.

    Args:
        rps_series: List of requests per second at each time step
        capacity_per_replica: Max RPS each replica can handle
        min_replicas: Minimum number of replicas
        max_replicas: Maximum number of replicas
        scale_up_threshold: Utilization threshold to trigger scale-up
        scale_down_threshold: Utilization threshold to trigger scale-down
        cooldown_steps: Steps to wait after a scaling action before scaling again

    Returns:
        Dict with simulation results
    """
    replicas = min_replicas
    cooldown = 0

    total_sla_violations = 0
    total_dropped_requests = 0
    total_utilization = 0
    max_replicas_used = replicas

    for rps in rps_series:
        capacity = replicas * capacity_per_replica
        utilization = rps / capacity

        total_utilization += utilization
        if rps > capacity:
            total_sla_violations += 1
            total_dropped_requests += rps - capacity
        if cooldown > 0:
            cooldown -= 1

        else:
            if utilization > scale_up_threshold and replicas < max_replicas:
                replicas += 1
                cooldown = cooldown_steps
            elif utilization < scale_down_threshold and replicas > min_replicas:
                replicas -= 1
                cooldown = cooldown_steps
        
        max_replicas_used = max(max_replicas_used, replicas)
    
    average_utilization = (
        round(total_utilization / len(rps_series), 2)
        if rps_series else 0.0
    )

    return {
        "final_replicas": replicas,
        "total_sla_violations": total_sla_violations,
        "average_utilization": average_utilization,
        "max_replicas_used": max_replicas_used,
        "total_dropped_requests": total_dropped_requests
    }










