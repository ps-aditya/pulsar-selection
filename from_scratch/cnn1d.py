"""1D-CNN from scratch (NumPy only): Conv(1->16, k=2, pad=1) + ReLU + flatten +
Linear(144 -> 2), softmax cross-entropy. Hand-derived forward/backward for the
v2 winning config. SGD with momentum, class-weighted loss, early stopping on
validation PR-AUC (NumPy implementation, same selection rule as cnn_grid.py)."""
import numpy as np


def _average_precision(y, p):
    order = np.argsort(-p, kind="mergesort")
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec[y == 1]).sum() / max(y.sum(), 1))


class ScratchCNN1d:
    def __init__(self, ch=16, k=2, lr=0.01, momentum=0.9, epochs=40,
                 batch=256, patience=5, seed=42):
        self.ch, self.k = ch, k
        self.lr, self.momentum, self.epochs = lr, momentum, epochs
        self.batch, self.patience, self.seed = batch, patience, seed

    def _init(self):
        rng = np.random.default_rng(self.seed)
        s = np.sqrt(2.0 / self.k)
        self.Wc_ = rng.normal(0, s, (self.ch, self.k))
        self.bc_ = np.zeros(self.ch)
        self.cache_ = {}

    def _conv_forward2(self, X):
        N, L = X.shape
        pad = self.k // 2
        xp = np.pad(X[:, None, :], ((0, 0), (0, 0), (pad, pad)))
        Lo = xp.shape[2] - self.k + 1
        out = np.einsum("cj,nij->nci", self.Wc_,
                        np.stack([xp[:, 0, j:j + Lo] for j in range(self.k)], axis=2))
        out = out + self.bc_[None, :, None]
        self.cache_["xp"], self.cache_["Lo"] = xp, Lo
        return np.maximum(out, 0), out > 0

    def forward(self, X):
        A, mask = self._conv_forward2(X)
        self.cache_["mask"] = mask
        self.cache_["Ashape"] = A.shape
        Z = A.reshape(len(X), -1) @ self.Wf_ + self.bf_
        e = np.exp(Z - Z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), A

    def _fit_init_head(self, Lo):
        rng = np.random.default_rng(self.seed + 1)
        self.Wf_ = rng.normal(0, np.sqrt(2.0 / (self.ch * Lo)), (self.ch * Lo, 2))
        self.bf_ = np.zeros(2)

    def gradients(self, X, y, w):
        # w: per-class weights; returns dict + probs
        N = len(y)
        P, A = self.forward(X)
        Y = np.zeros_like(P)
        Y[np.arange(N), y] = 1.0
        cw = np.array([w[0], w[1]])[y]
        dZ = (P - Y) * (cw / cw.sum() * N)[:, None] / N
        Af = A.reshape(N, -1)
        dWf = Af.T @ dZ
        dbf = dZ.sum(0)
        dA = (dZ @ self.Wf_.T).reshape(A.shape) * self.cache_["mask"]
        xp, Lo = self.cache_["xp"], self.cache_["Lo"]
        wins = np.stack([xp[:, 0, j:j + Lo] for j in range(self.k)], axis=2)  # (N,k,Lo)
        dWc = np.einsum("nci,nik->ck", dA, wins)
        dbc = dA.sum((0, 2))
        return {"Wc": dWc, "bc": dbc, "Wf": dWf, "bf": dbf}, P

    def fit(self, X, y, X_val, y_val, w):
        X = np.asarray(X, float)
        self._init()
        pad = self.k // 2
        Lo = X.shape[1] + 2 * pad - self.k + 1
        self._fit_init_head(Lo)
        y, yv = np.asarray(y, int), np.asarray(y_val, int)
        Xv = np.asarray(X_val, float)
        rng = np.random.default_rng(self.seed)
        n = len(y)
        V = {k: np.zeros_like(v) for k, v in
             [("Wc", self.Wc_), ("bc", self.bc_), ("Wf", self.Wf_), ("bf", self.bf_)]}
        P_ = {"Wc": self.Wc_, "bc": self.bc_, "Wf": self.Wf_, "bf": self.bf_}
        best, wait, snap = -1.0, 0, None
        for ep in range(self.epochs):
            for s in np.array_split(rng.permutation(n), max(1, n // self.batch)):
                g, _ = self.gradients(X[s], y[s], w)
                for k in V:
                    V[k] = self.momentum * V[k] + g[k]
                    P_[k] -= self.lr * V[k]
            self.Wc_, self.bc_, self.Wf_, self.bf_ = P_["Wc"], P_["bc"], P_["Wf"], P_["bf"]
            s = _average_precision(yv, self._val_proba(Xv)[:, 1])
            if s > best + 1e-6:
                best, wait = s, 0
                snap = {k: P_[k].copy() for k in P_}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        for k in P_:
            setattr(self, k + "_", snap[k])
            P_[k] = snap[k]
        self.n_epochs_ = ep + 1
        return self

    def _val_proba(self, X):
        P, _ = self.forward(np.asarray(X, float))
        return P

    def predict_proba(self, X):
        return self._val_proba(X)
