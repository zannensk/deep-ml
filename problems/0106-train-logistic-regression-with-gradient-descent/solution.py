import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))
	y = y.reshape(-1, 1)
	X = np.hstack((np.ones((X.shape[0], 1)), X))
	B = np.zeros((X.shape[1], 1))

	losses = []
	for _ in range(iterations):
		z = X @ B
		y_pred = sigmoid(z)
		loss = - np.sum(
			y * np.log(y_pred)
			+ (1 - y) * np.log(1 - y_pred)
		)
		losses.append(round(float(loss), 4))

		gradient = X.T @ (y_pred - y)

		B -= learning_rate * gradient
	return B.flatten().round(4).tolist(), losses