def batch_requests(requests: list, max_batch_size: int, max_wait_time: float) -> list:
    """
    Group inference requests into batches based on size and time constraints.
    
    Args:
        requests: List of dicts with 'id', 'timestamp', 'features'
        max_batch_size: Maximum number of requests per batch
        max_wait_time: Maximum time to wait before processing a batch
    
    Returns:
        List of tuples: (request_ids, batched_features, process_time)
    """
    # Your code here
    if not requests:
        return []
    
    batches = []
    batch = []

    ordered = sorted(
        requests,
        key=lambda r: r["timestamp"]
    )

    def finalize(current):
        ids = [r["id"] for r in current]
        features = [r["features"] for r in current]
        process_time = current[-1]["timestamp"]

        return (ids, features, process_time)
    
    for request in ordered:
        if batch:
            size_exceeded = len(batch) >= max_batch_size
            time_exceeded = (
                request["timestamp"] - batch[0]["timestamp"]
                > max_wait_time
            )
            if size_exceeded or time_exceeded:
                batches.append(finalize(batch))
                batch = []
        batch.append(request)
    
    if batch:
        batches.append(finalize(batch))
    
    return batches
    





