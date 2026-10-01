"""v2 compute step: 32-config 1D-CNN grid (Report6 search space), SGD, val PR-AUC selection.
Saves cnn_grid_results.json + cnn_best.pt (repo-root artifacts, deterministic SEED=42).
Run: python cnn_grid.py
"""
import itertools, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)

df = pd.read_csv("data/HTRU_2.csv", header=None)
X, y = df.iloc[:, :8].values, df.iloc[:, 8].values
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
sc = StandardScaler().fit(X_tr)
Z_tr, Z_va = sc.transform(X_tr), sc.transform(X_va)
w = len(y_tr) / (2 * np.bincount(y_tr))

class CNN1d(nn.Module):
    def __init__(self, blocks=1, ch=8, k=3, pool=True, act="relu"):
        super().__init__()
        f = nn.ReLU if act == "relu" else nn.Tanh
        layers, c_in = [], 1
        for _ in range(blocks):
            layers += [nn.Conv1d(c_in, ch, k, padding=k // 2), f()]
            c_in = ch
            if pool:
                layers += [nn.MaxPool1d(2)]
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 1, 8)).numel()
        self.fc = nn.Linear(n_flat, 2)
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))
    def cfg(self):
        return dict(blocks=len([m for m in self.conv if isinstance(m, nn.Conv1d)]))

def loader(Z, y, shuffle=False):
    ds = TensorDataset(torch.tensor(Z, dtype=torch.float32).unsqueeze(1), torch.tensor(y))
    g = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(ds, batch_size=256, shuffle=shuffle, num_workers=0, generator=g)

def fit_eval(cfg, max_ep=40, patience=5):
    torch.manual_seed(SEED)
    m = CNN1d(**cfg)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    tr_dl, va_dl = loader(Z_tr, y_tr, shuffle=True), loader(Z_va, y_va)
    best, wait, hist = -1, 0, {"loss": [], "val_prauc": []}
    for ep in range(max_ep):
        m.train(); tl, n = 0.0, 0
        for xb, yb in tr_dl:
            opt.zero_grad(); loss = ce(m(xb), yb); loss.backward(); opt.step()
            tl += loss.item() * len(xb); n += len(xb)
        m.eval()
        with torch.no_grad():
            pv = torch.cat([m(xb) for xb, _ in va_dl]).softmax(1)[:, 1].numpy()
        s = average_precision_score(y_va, pv)
        hist["loss"].append(tl / n); hist["val_prauc"].append(s)
        if s > best + 1e-6:
            best, wait = s, 0
            best_state = {k: v.cpu() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(best_state)
    return m, best, hist, ep + 1

grid = [dict(blocks=b, ch=c, k=k, pool=p, act=a)
        for b, c, k, p, a in itertools.product([1, 2], [8, 16], [2, 3], [True, False], ["relu", "tanh"])]
t0 = time.time()
rows = []
for i, cfg in enumerate(grid):
    _, s, _, eps = fit_eval(cfg)
    rows.append({"cfg": cfg, "val_prauc": s, "epochs": eps})
    print(f"[{i+1:>2}/{len(grid)}] val-PRAUC={s:.4f} eps={eps} {cfg}", flush=True)
rows.sort(key=lambda r: -r["val_prauc"])
winner = rows[0]
m, _, hist, _ = fit_eval(winner["cfg"])
torch.save({"cfg": winner["cfg"], "state": m.state_dict()},
           "cnn_best.pt")
json.dump({"seed": SEED, "grid": len(grid), "rows": rows,
           "winner": winner, "hist": hist}, open("cnn_grid_results.json", "w"), indent=1)
print(f"WON {winner} in {time.time()-t0:.0f}s")
