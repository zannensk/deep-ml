import math


def tts_stream_capacity(rtf: float, num_gpus: int, mem_per_stream_mb: float, total_gpu_mem_mb: float, audio_bitrate_kbps: float, total_bandwidth_mbps: float) -> dict:
    """
    Calculate the maximum number of concurrent real-time TTS streams.

    Args:
        rtf: Real-Time Factor (compute time / audio duration) per stream per GPU
        num_gpus: Number of GPUs available
        mem_per_stream_mb: GPU memory required per stream in MB
        total_gpu_mem_mb: Total GPU memory available in MB
        audio_bitrate_kbps: Audio bitrate per stream in kbps
        total_bandwidth_mbps: Total network bandwidth in Mbps

    Returns:
        Dictionary with capacity breakdown, max concurrent streams, and bottleneck
    """
    # Your code here
    compute_capacity = math.floor(num_gpus / rtf)
    memory_capacity = math.floor(total_gpu_mem_mb / mem_per_stream_mb)
    bandwidth_capacity = math.floor(total_bandwidth_mbps * 1000 / audio_bitrate_kbps)

    capacities = {
        "compute": compute_capacity,
        "memory": memory_capacity,
        "bandwidth": bandwidth_capacity
    }
    bottleneck = min(capacities, key=capacities.get)
    max_streams = capacities[bottleneck]

    return {
        "compute_capacity": compute_capacity,
        "memory_capacity": memory_capacity,
        "bandwidth_capacity": bandwidth_capacity,
        "max_concurrent_streams": max_streams,
        "bottleneck": bottleneck
    }

