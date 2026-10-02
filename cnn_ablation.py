"""Ablation: winning CNN config on profile-only vs DM-only vs full-8 inputs.
Same splits/seed/SGD/early-stop protocol as cnn_grid.py. Saves cnn_ablation.json.
Run: python cnn_ablation.py
"""
import json
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, precision_score, recall_score)

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
CFG = dict(blocks=1, ch=16, k=2, pool=False, act="relu")

df = pd.read_csv("data/HTRU_2.csv", header=None)
X8, y = df.iloc[:, :8].values, df.iloc[:, 8].values
SETS = {"profile-only": [0, 1, 2, 3], "DM-only": [4, 5, 6, 7], "full-8": list(range(8))}

class CNN1d(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, CFG["ch"], CFG["k"], padding=CFG["k"] // 2), nn.ReLU())
        with torch.no_grad():
            self.fc = nn.Linear(self.conv(torch.zeros(1, 1, L)).numel(), 2)
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))

def run(cols):
    Xa = X8[:, cols]
    Xtr, Xtmp, ytr, ytmp = train_test_split(Xa, y, test_size=0.40, stratify=y, random_state=SEED)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    w = len(ytr) / (2 * np.bincount(ytr))
    def loader(Z, yy, shuffle=False):
        ds = TensorDataset(torch.tensor(Z, dtype=torch.float32).unsqueeze(1), torch.tensor(yy))
        g = torch.Generator().manual_seed(SEED) if shuffle else None
        return DataLoader(ds, batch_size=256, shuffle=shuffle, num_workers=0, generator=g)
    torch.manual_seed(SEED)
    m = CNN1d(len(cols))
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    tr_dl, va_dl = loader(Ztr, ytr, True), loader(Zva, yva)
    best, wait, bs = -1, 0, None
    for ep in range(40):
        m.train()
        for xb, yb in tr_dl:
            opt.zero_grad(); loss = ce(m(xb), yb); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = torch.cat([m(xb) for xb, _ in va_dl]).softmax(1)[:, 1].numpy()
        s = average_precision_score(yva, pv)
        if s > best + 1e-6:
            best, wait = s, 0
            bs = {k: v.cpu() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 5:
                break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        p = torch.cat([m(xb) for xb, _ in loader(Zte, yte)]).softmax(1)[:, 1].numpy()
    yh = (p >= 0.5).astype(int)
    return dict(roc=roc_auc_score(yte, p), pr=average_precision_score(yte, p),
                brier=brier_score_loss(yte, p), acc=accuracy_score(yte, yh),
                posprec=precision_score(yte, yh), posrec=recall_score(yte, yh),
                val_prauc=best, epochs=ep + 1)

out = {}
for name, cols in SETS.items():
    r = run(cols)
    out[name] = {k: round(float(v), 4) for k, v in r.items()}
    print(f"{name:<12} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
json.dump({"seed": SEED, "cfg": CFG, "results": out}, open("cnn_ablation.json", "w"), indent=1)
print("saved cnn_ablation.json")
