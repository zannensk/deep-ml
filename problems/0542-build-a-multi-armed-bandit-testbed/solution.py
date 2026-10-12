import numpy as np

def create_bandit_testbed(k: int, num_pulls: int, seed: int = 42) -> tuple:
    """
    Build a k-armed bandit testbed and simulate pulling each arm.
    
    Args:
        k: Number of arms
        num_pulls: Number of times to pull each arm
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (true_values, sample_means, optimal_arm)
    """
    np.random.seed(seed)
    true_values = np.random.normal(0, 1, size=k)
    sample_means = []
    for q in true_values:
        rewards = np.random.normal(q, 1, size=num_pulls)
        sample_means.append(round(float(np.mean(rewards)), 4))
    
    optimal_action = int(np.argmax(true_values))
    return (
        [round(float(q), 4) for q in true_values],
        sample_means,
        optimal_action
    )