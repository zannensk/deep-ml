import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    # Your code here
    y_scores = np.asarray(y_scores)
    y_true = np.asarray(y_true)

    actual_positives = np.sum(y_true == 1)
    actual_negatives = np.sum(y_true == 0)
    tprs = [0.0]
    fprs = [0.0]
    thresholds = sorted(np.unique(y_scores), reverse=True)

    for threshold in thresholds:
        y_pred = y_scores >= threshold
        tp = np.sum((y_true == 1) & y_pred)
        fp = np.sum((y_true == 0) & y_pred)

        if actual_positives == 0:
            tpr = 0.0
        else:
            tpr = tp / actual_positives
        
        if actual_negatives == 0:
            fpr = 0.0
        else:
            fpr = fp / actual_negatives
        
        tprs.append(float(tpr))
        fprs.append(float(fpr))
    
    return fprs, tprs