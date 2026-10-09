import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
	"""
	Compute Focal Loss for multi-class classification.
	
	Args:
		y_true: Ground truth labels as class indices (list or 1D array)
		y_pred: Predicted probabilities (2D array, shape: [n_samples, n_classes])
		gamma: Focusing parameter (default: 2.0)
		alpha: Class weights (optional, list or 1D array of length n_classes)
	
	Returns:
		float: Average focal loss
	"""
	# Your code here
	y_pred = np.asarray(y_pred)
	y_true = np.asarray(y_true)

	n = len(y_true)

	p_t = y_pred[np.arange(n), y_true]

	losses = - ((1 - p_t) ** gamma) * np.log(p_t)

	if alpha is not None:
		alpha = np.asarray(alpha)
		losses = alpha[y_true] * losses
	
	return float(np.mean(losses))