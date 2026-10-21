import numpy as np

def all_reduce_mean(worker_grads):
    """
    Simulate DDP all-reduce mean across workers.

    Args:
        worker_grads: list of length N. Each element is a list of numpy arrays
                      (gradients for each parameter on that worker).

    Returns:
        list of nested Python lists - the averaged gradients per parameter,
        rounded to 6 decimal places.
    """
    # Your code here
    if not worker_grads:
        return []
    
    n_params = len(worker_grads[0])
    result = []
    for worker in worker_grads:
        if len(worker) != n_params:
            raise ValueError
    
    for j in range(n_params):
        grads = [
            np.asarray(worker[j])
            for worker in worker_grads
        ]
        if any(g.shape != grads[0].shape for g in grads):
            raise ValueError
        stacked = np.stack(grads, axis=0)
        avg = np.mean(stacked, axis=0)

        result.append(np.round(avg, 4).tolist())
    
    return result
