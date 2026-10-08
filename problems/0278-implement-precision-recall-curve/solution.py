import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    # Your code here
    y_scores = np.asarray(y_scores)
    y_true = np.asarray(y_true)

    thresholds = sorted(np.unique(y_scores), reverse=True)

    precisions = []
    recalls = []
    actual_positives = np.sum(y_true == 1)

    for threshold in thresholds:
        y_pred = y_scores >= threshold
        tp = np.sum((y_true == 1) & y_pred)
        predicted_positives = np.sum(y_pred)
        if predicted_positives == 0:
            precision = 1.0
        else:
            precision = tp / predicted_positives
        
        if actual_positives == 0:
            recall = 0.0
        else:
            recall = tp / actual_positives
        
        precisions.append(float(precision))
        recalls.append(float(recall))
    
    thresholds = [float(x) for x in thresholds]
    
    return precisions, recalls, list(thresholds)