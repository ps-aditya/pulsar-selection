"""Logistic regression from scratch (NumPy only): sigmoid + BCE + batch gradient descent."""
import numpy as np


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class ScratchLogReg:
    def __init__(self, lr=1.0, epochs=20000, l2=1e-4, seed=42):
        self.lr, self.epochs, self.l2, self.seed = lr, epochs, l2, seed

    def fit(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w_ = rng.normal(0, 0.01, d)
        self.b_ = 0.0
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight) * n / np.sum(sample_weight)
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w_ + self.b_)
            err = (p - y) * sw
            self.w_ -= self.lr * (X.T @ err / n + self.l2 * self.w_)
            self.b_ -= self.lr * err.sum() / n
        return self

    def predict_proba(self, X):
        p = sigmoid(X @ self.w_ + self.b_)
        return np.c_[1 - p, p]
