import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes, counts = np.unique(
            y, return_counts = True
        )
        self.log_prior = np.log(counts / len(y))
        probs = []
        for cls, count in zip(self.classes, counts):
            X_cls = X[y==cls]
            p = (
                X_cls.sum(axis=0) + self.smoothing
            ) / (count + 2 * self.smoothing)
            probs.append(p)
        self.probs = probs
        p = np.clip(self.probs, 1e-12, 1 - 1e-12)
        self.log_p = np.log(p)
        self.log_1_minus_p = np.log1p(-p)

    def predict(self, X):
        # Predict class labels for test set X
        X = np.asarray(X)
        scores = (
            X @ self.log_p.T
            + (1 - X) @ self.log_1_minus_p.T
            + self.log_prior
        )
        return self.classes[np.argmax(scores, axis=1 )]