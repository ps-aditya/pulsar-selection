"""MLP from scratch (NumPy only): 8 -> H (ReLU) -> 1 (sigmoid), BCE + L2,
hand-derived backprop, SGD/Adam. Gradient-checkable via `gradients()`."""
import numpy as np


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class ScratchMLP:
    def __init__(self, hidden=100, lr=0.001, epochs=500, alpha=1e-4,
                 batch=200, optim="adam", patience=10, seed=42):
        self.hidden, self.lr, self.epochs, self.alpha = hidden, lr, epochs, alpha
        self.batch, self.optim, self.patience, self.seed = batch, optim, patience, seed

    def _init(self, d):
        rng = np.random.default_rng(self.seed)
        self.W1_ = rng.normal(0, np.sqrt(2.0 / d), (d, self.hidden))
        self.b1_ = np.zeros(self.hidden)
        self.W2_ = rng.normal(0, np.sqrt(2.0 / self.hidden), (self.hidden, 1))
        self.b2_ = np.zeros(1)

    def _forward(self, X):
        Z1 = X @ self.W1_ + self.b1_
        A1 = np.maximum(Z1, 0)
        Z2 = A1 @ self.W2_ + self.b2_
        return Z1, A1, _sigmoid(Z2).ravel()

    def _loss(self, p, y):
        eps = 1e-12
        pc = np.clip(p, eps, 1 - eps)
        bce = -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()
        reg = 0.5 * self.alpha * (np.sum(self.W1_ ** 2) + np.sum(self.W2_ ** 2))
        return bce + reg

    def gradients(self, X, y):
        n = len(y)
        Z1, A1, p = self._forward(X)
        dZ2 = ((p - y) / n).reshape(-1, 1)
        dW2 = A1.T @ dZ2 + self.alpha * self.W2_
        db2 = dZ2.sum(0)
        dA1 = dZ2 @ self.W2_.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1 + self.alpha * self.W1_
        db1 = dZ1.sum(0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self._init(X.shape[1])
        params = ["W1", "b1", "W2", "b2"]
        P = {k: getattr(self, k + "_") for k in params}
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(v) for k, v in P.items()}
        rng = np.random.default_rng(self.seed)
        n = len(y)
        mon = (X_val, y_val) if X_val is not None else (X, y)
        best, wait, snap = 1e18, 0, None
        t = 0
        for ep in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch):
                t += 1
                b = idx[s:s + self.batch]
                g = self.gradients(X[b], y[b])
                for k in params:
                    if self.optim == "adam":
                        m[k] = 0.9 * m[k] + 0.1 * g[k]
                        v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                        mh = m[k] / (1 - 0.9 ** t)  # bias-corrected moments
                        vh = v[k] / (1 - 0.999 ** t)
                        step = self.lr * mh / (np.sqrt(vh) + 1e-8)
                    else:
                        step = self.lr * g[k]
                    P[k] -= step
            for k in params:
                setattr(self, k + "_", P[k])
            vl = self._loss(self._forward(mon[0])[2], np.asarray(mon[1], float))
            if vl < best - 1e-6:
                best, wait = vl, 0
                snap = {k: getattr(self, k + "_").copy() for k in params}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        for k in params:
            setattr(self, k + "_", snap[k])
        self.n_epochs_ = ep + 1
        return self

    def predict_proba(self, X):
        p = self._forward(np.asarray(X, float))[2]
        return np.c_[1 - p, p]
