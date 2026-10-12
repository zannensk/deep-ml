import numpy as np

def ucb_action(counts, values, t, c):
    """
    Choose an action using the UCB1 formula.
    Args:
      counts (np.ndarray): Number of times each action has been chosen
      values (np.ndarray): Average reward of each action
      t (int): Current timestep (starts from 1)
      c (float): Exploration coefficient
    Returns:
      int: Index of action to select
    """
    # TODO: Implement the UCB action selection
    untried = np.flatnonzero(counts == 0)
    if untried.size > 0:
      return int(untried[0])

    ucb = values + c * np.sqrt(np.log(t) / counts)

    return int(np.argmax(ucb))