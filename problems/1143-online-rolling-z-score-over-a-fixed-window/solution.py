from collections import deque
import math

def rolling_zscore(stream, window):
    # stream: list of numbers
    # window: positive int, fixed window size
    # return: list of floats, same length as stream
    q = deque()
    rolling_sum = 0.0
    rolling_sq_sum = 0.0

    result = []

    for x in stream:
        q.append(x)
        rolling_sum += x
        rolling_sq_sum += x * x

        if len(q) > window:
            old = q.popleft()
            rolling_sum -= old
            rolling_sq_sum -= old * old
        
        if len(q) < window:
            result.append(0.0)
            continue
        
        mean = rolling_sum / window
        variance = rolling_sq_sum / window - mean * mean

        std = math.sqrt(variance)

        if std == 0:
            z = 0.0
        else:
            z = (x - mean) / std
        
        result.append(z)
    
    return result