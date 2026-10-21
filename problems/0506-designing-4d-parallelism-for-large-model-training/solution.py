import numpy as np

def design_4d_parallelism(model_config: dict, parallel_config: dict, hardware_config: dict) -> dict:
    """
    Analyze a 4D parallelism configuration for large model training.
    
    Args:
        model_config: dict with num_layers, hidden_size, vocab_size, seq_len,
                      micro_batch_size, num_micro_batches
        parallel_config: dict with dp, tp, pp, sp
        hardware_config: dict with num_gpus, memory_per_gpu_gb
    
    Returns:
        Dictionary with parallelism analysis metrics
    """
    # Your code here
    L = model_config["num_layers"]
    H = model_config["hidden_size"]
    V = model_config["vocab_size"]
    S = model_config["seq_len"]
    B = model_config["micro_batch_size"]
    m = model_config["num_micro_batches"]

    # 2. Parallelism degrees
    dp = parallel_config["dp"]
    tp = parallel_config["tp"]
    pp = parallel_config["pp"]
    sp = parallel_config["sp"]

    num_gpus = hardware_config["num_gpus"]
    memory_limit = hardware_config["memory_per_gpu_gb"]

    GB =1e9

    valid = (dp * tp * pp * sp == num_gpus)
    P = 2 * V * H + 12 * L * H**2
    Ls = L // pp
    A = 2 * S * B * H

    param_memory = 16 * P / (tp * pp)
    activation_memory = (
        10 * A * Ls / (tp * sp)
    )
    total_memory = param_memory + activation_memory
    fits = (total_memory / GB <= memory_limit)

    dp_comm = (
        2 * (dp - 1) / dp
        * 2 * P / (tp * pp)
        if dp > 1 else 0.0
    )
    tp_comm = (
        4 * Ls * A * 2 * (tp - 1) / tp
        if tp > 1 else 0.0
    )
    pp_comm = (
        2 * A * (pp - 1)
        if pp > 1 else 0.0
    )
    sp_comm = (
        2 * Ls * A * (sp - 1) / sp
        if sp > 1 else 0.0
    )
    bubble_ratio = (
        (pp - 1) / (m + pp - 1)
        if pp > 1 else 0.0
    )

    return {
        "valid": bool(valid),
        "total_params": int(P),
        "param_memory_per_gpu_gb": round(param_memory / GB, 4),
        "activation_memory_per_gpu_gb": round(
            activation_memory / GB, 4
        ),
        "total_memory_per_gpu_gb": round(
            total_memory / GB, 4
        ),
        "fits_in_memory": bool(fits),
        "dp_comm_gb": round(dp_comm / GB, 4),
        "tp_comm_gb": round(tp_comm / GB, 4),
        "pp_comm_gb": round(pp_comm / GB, 4),
        "sp_comm_gb": round(sp_comm / GB, 4),
        "bubble_ratio": round(bubble_ratio, 4)
    }
