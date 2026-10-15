def asr_parallel_rtf(audio_duration: float, chunk_times: list, num_workers: int) -> dict:
    """
    Calculate ASR Real-Time Factor metrics for parallel chunk transcription.
    
    Args:
        audio_duration: Total audio duration in seconds
        chunk_times: List of processing times (seconds) for each audio chunk
        num_workers: Number of parallel transcription workers
    
    Returns:
        Dictionary with 'sequential_rtf', 'parallel_rtf', 'speedup', and 'is_realtime'
    """
    sequential_time = sum(chunk_times)
    parallel_time = 0

    for i in range(0, len(chunk_times), num_workers):
        batch = chunk_times[i: i + num_workers]
        parallel_time += max(batch)
    
    sequential_rtf = sequential_time / audio_duration
    parallel_rtf = parallel_time / audio_duration

    speedup = sequential_time / parallel_time if parallel_time > 0 else 0.0

    is_realtime = parallel_rtf < 1.0
    return {
        "sequential_rtf": round(sequential_rtf, 4),
        "parallel_rtf": round(parallel_rtf, 4),
        "speedup": round(speedup, 4),
        "is_realtime": is_realtime
    }
