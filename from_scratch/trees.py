"""Decision tree (Gini, recursive binary splits) + Random Forest (bagging +
feature subsampling) from scratch — NumPy only, no sklearn inside."""
import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "prob")

    def __init__(self, feature=None, threshold=None, left=None, right=None, prob=0.0):
        self.feature, self.threshold, self.left, self.right, self.prob = \
            feature, threshold, left, right, prob


def _gini(wpos, wtot):
    p = np.divide(wpos, wtot, out=np.zeros_like(wpos, dtype=float), where=wtot > 0)
    return 2.0 * p * (1.0 - p)


class ScratchTree:
    def __init__(self, max_depth=20, min_samples_leaf=2, max_features=None, seed=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.sw_ = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.rng_ = np.random.default_rng(self.seed)
        self.n_features_ = X.shape[1]
        self.root_ = self._build(X, y, self.sw_, 0)
        return self

    def _feat_subset(self):
        if self.max_features is None:
            return np.arange(self.n_features_)
        k = self.max_features if isinstance(self.max_features, int) else \
            max(1, int(np.sqrt(self.n_features_)))
        return np.sort(self.rng_.choice(self.n_features_, k, replace=False))

    def _best_split(self, X, y, sw):
        n = len(y)
        if n < 2 * self.min_samples_leaf:
            return None
        tot_w, pos_w = sw.sum(), sw[y == 1].sum()
        parent = _gini(np.array([pos_w]), np.array([tot_w]))[0]
        best = None
        for f in self._feat_subset():
            col = X[:, f]
            order = np.argsort(col, kind="mergesort")
            v = col[order]
            is_pos = (y[order] == 1)
            w = sw[order]
            cum_w = np.cumsum(w)
            cum_p = np.cumsum(np.where(is_pos, w, 0.0))
            left_n = np.arange(1, n)
            ok = (v[1:] != v[:-1]) & (left_n >= self.min_samples_leaf) & \
                 ((n - left_n) >= self.min_samples_leaf)
            if not np.any(ok):
                continue
            lw, lpw = cum_w[:-1][ok], cum_p[:-1][ok]
            rw, rpw = tot_w - lw, pos_w - lpw
            gain = parent - (lw * _gini(lpw, lw) + rw * _gini(rpw, rw)) / tot_w
            j = int(np.argmax(gain))
            if best is None or gain[j] > best[0]:
                cut = np.flatnonzero(ok)[j]
                best = (float(gain[j]), f, float((v[cut] + v[cut + 1]) / 2))
        return best[1:] if best and best[0] > 0 else None

    def _build(self, X, y, sw, depth):
        p1 = sw[y == 1].sum() / sw.sum()
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or p1 in (0.0, 1.0):
            return _Node(prob=float(p1))
        s = self._best_split(X, y, sw)
        if s is None:
            return _Node(prob=float(p1))
        f, t = s
        L = X[:, f] <= t
        return _Node(f, t, self._build(X[L], y[L], sw[L], depth + 1),
                     self._build(X[~L], y[~L], sw[~L], depth + 1))

    def _walk(self, node, x):
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prob

    def predict_proba(self, X):
        p = np.array([self._walk(self.root_, x) for x in np.asarray(X, float)])
        return np.c_[1 - p, p]


class ScratchForest:
    def __init__(self, n_estimators=25, max_depth=20, min_samples_leaf=2,
                 max_features=3, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, float), np.asarray(y, int)
        n = len(y)
        base = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        self.trees_ = []
        for t in range(self.n_estimators):
            rng = np.random.default_rng(self.seed + t)
            idx = rng.integers(0, n, n)
            tree = ScratchTree(self.max_depth, self.min_samples_leaf,
                               self.max_features, seed=self.seed + 1000 + t)
            tree.fit(X[idx], y[idx], base[idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        p = np.mean([t.predict_proba(X)[:, 1] for t in self.trees_], axis=0)
        return np.c_[1 - p, p]
