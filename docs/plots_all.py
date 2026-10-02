"""Regenerate ALL docs/assets figures with polished, non-overlapping layout.
Deterministic SEED=42. Run: python docs/plots_all.py
"""
from pathlib import Path
import urllib.request, zipfile, io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
A = Path(__file__).parent / "assets"
A.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10.5, "axes.titlesize": 12,
                     "figure.constrained_layout.use": True})

# ---- data + splits (identical everywhere) ----
DATA = Path(__file__).parent.parent / "data" / "HTRU_2.csv"
if not DATA.exists():
    DATA.parent.mkdir(exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(
        "https://archive.ics.uci.edu/static/public/372/htru2.zip", timeout=120).read()))
    DATA.write_bytes(z.read("HTRU_2.csv"))
COLS = ["profile_mean", "profile_stdev", "profile_kurtosis", "profile_skewness",
        "dm_mean", "dm_stdev", "dm_kurtosis", "dm_skewness", "target"]
df = pd.read_csv(DATA, header=None, names=COLS)
feats = COLS[:8]
X, y = df[feats].values, df["target"].values
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
sc = StandardScaler().fit(X_tr)
Z_tr, Z_va, Z_te = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)

# ---- models ----
lr = make_pipeline(StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_leaf=2,
    class_weight="balanced_subsample", random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
cal = CalibratedClassifierCV(estimator=FrozenEstimator(rf), method="sigmoid").fit(X_va, y_va)

class CNN1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 16, 2, padding=1), nn.ReLU())
        with torch.no_grad():
            self.fc = nn.Linear(self.conv(torch.zeros(1, 1, 8)).numel(), 2)
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))

cnn = CNN1d()
cnn.load_state_dict(torch.load(Path(__file__).parent.parent / "cnn_best.pt",
                               weights_only=True)["state"])
cnn.eval()
def cnn_logits(Z):
    with torch.no_grad():
        dl = DataLoader(TensorDataset(torch.tensor(Z, dtype=torch.float32).unsqueeze(1)),
                        batch_size=1024)
        return torch.cat([cnn(xb) for xb, in dl]).numpy()
z_va, z_te = cnn_logits(Z_va), cnn_logits(Z_te)
zv = torch.tensor(z_va)
logT = torch.zeros(1, requires_grad=True)
opt = torch.optim.LBFGS([logT], max_iter=100)
def _cl():
    opt.zero_grad()
    l = nn.CrossEntropyLoss()(zv / logT.exp(), torch.tensor(y_va))
    l.backward()
    return l
opt.step(_cl)
T = float(logT.exp().detach())
p_rf = rf.predict_proba(X_te)[:, 1]
p_rf_cal = cal.predict_proba(X_te)[:, 1]
p_cnn = torch.softmax(torch.tensor(z_te), 1)[:, 1].numpy()
p_cnnT = torch.softmax(torch.tensor(z_te) / T, 1)[:, 1].numpy()
print(f"T={T:.3f}")

# ---- 1. balance ----
bal = df["target"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5.2, 3.6))
bars = ax.bar(["non-pulsar (0)", "pulsar (1)"], [bal[0], bal[1]], color=["#3b82f6", "#f59e0b"])
ax.set_title("HTRU2 class balance (n = 17,898)")
ax.set_ylabel("count")
ax.set_ylim(0, max(bal[0], bal[1]) * 1.22)
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
for b, v in zip(bars, [bal[0], bal[1]]):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/len(df):.1%})",
            ha="center", va="bottom", fontsize=10)
fig.savefig(A / "balance.png", bbox_inches="tight"); plt.close(fig)

# ---- 2. reliability (RF) ----
fig, ax = plt.subplots(figsize=(5.4, 4.4))
ax.plot([0, 1], [0, 1], ls="--", color="gray", label="ideal")
for name, p, c in [("rf", p_rf, "#f59e0b"), ("rf + calibrated", p_rf_cal, "#16a34a")]:
    ft, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax.plot(mp, ft, marker="o", ms=5, label=name, color=c)
ax.set_title("Reliability diagram — Random Forest (test)")
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("fraction positive")
ax.legend(loc="lower right", fontsize=9); ax.set_aspect("equal", adjustable="box")
fig.savefig(A / "reliability.png", bbox_inches="tight"); plt.close(fig)

# ---- 3. learning curve ----
lc = RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_leaf=2,
    class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
tr_n, tr_s, va_s = learning_curve(lc, X_tr, y_tr,
    cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
    train_sizes=np.linspace(0.1, 1.0, 5), scoring="average_precision", n_jobs=-1)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(tr_n, tr_s.mean(1), marker="o", label="train PR-AUC")
ax.plot(tr_n, va_s.mean(1), marker="o", label="cv PR-AUC")
ax.fill_between(tr_n, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=0.15)
ax.set_title("Learning curve — RF (PR-AUC)")
ax.set_xlabel("training examples"); ax.set_ylabel("PR-AUC")
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
ax.legend(fontsize=9)
fig.savefig(A / "learning_curve.png", bbox_inches="tight"); plt.close(fig)

# ---- 4. permutation importance ----
pi = permutation_importance(cal, X_te, y_te, n_repeats=10, random_state=SEED,
                            n_jobs=-1, scoring="average_precision")
order = np.argsort(pi.importances_mean)[::-1]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
yy = np.arange(len(feats))
ax.barh(yy, pi.importances_mean[order], xerr=pi.importances_std[order],
        color="#3b82f6", ecolor="black", capsize=3)
ax.set_yticks(yy); ax.set_yticklabels([feats[i] for i in order])
ax.invert_yaxis()
ax.set_title("Permutation importance — test PR-AUC drop")
ax.set_xlabel("mean PR-AUC drop (10 repeats)")
for j, i in enumerate(order):
    ax.text(pi.importances_mean[i] + pi.importances_std[i] + 0.008, j,
            f"{pi.importances_mean[i]:.3f}", va="center", fontsize=9)
fig.savefig(A / "importance.png", bbox_inches="tight"); plt.close(fig)

# ---- 5/6. CNN train curves + CNN reliability (from stored grid hist) ----
import json as _json
G = _json.load(open(Path(__file__).parent.parent / "cnn_grid_results.json"))
hist = G["hist"]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax[0].plot(hist["loss"], color="#3b82f6"); ax[0].set_title("CNN training loss (winner)")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy")
ax[1].plot(hist["val_prauc"], color="#16a34a"); ax[1].set_title("CNN validation PR-AUC (winner)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("PR-AUC")
for a in ax: a.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.savefig(A / "cnn_train.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 4.4))
ax.plot([0, 1], [0, 1], ls="--", color="gray", label="ideal")
ft, mp = calibration_curve(y_te, p_cnn, n_bins=10, strategy="quantile")
ax.plot(mp, ft, marker="o", ms=5, color="#f59e0b", label="cnn (uncalibrated)")
ax.set_title("CNN reliability (test)")
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("fraction positive")
ax.legend(loc="lower right", fontsize=9); ax.set_aspect("equal", adjustable="box")
fig.savefig(A / "cnn_reliability.png", bbox_inches="tight"); plt.close(fig)

# ---- 7. temperature scaling effect ----
fig, ax = plt.subplots(figsize=(5.4, 4.4))
ax.plot([0, 1], [0, 1], ls="--", color="gray", label="ideal")
for name, p, c in [("cnn raw", p_cnn, "#f59e0b"), (f"cnn T-scaled (T={T:.2f})", p_cnnT, "#16a34a")]:
    ft, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax.plot(mp, ft, marker="o", ms=5, label=name, color=c)
ax.set_title("Temperature scaling effect (test)")
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("fraction positive")
ax.legend(loc="lower right", fontsize=9); ax.set_aspect("equal", adjustable="box")
fig.savefig(A / "cnn_temp.png", bbox_inches="tight"); plt.close(fig)

# ---- 8. delta CIs (redesigned: annotated, uncrowded) ----
B, n = 2000, len(y_te)
rng = np.random.default_rng(42)
idx = rng.integers(0, n, size=(B, n))
d_roc = np.array([roc_auc_score(y_te[i], p_cnnT[i]) - roc_auc_score(y_te[i], p_rf[i]) for i in idx])
d_pr = np.array([average_precision_score(y_te[i], p_cnnT[i]) - average_precision_score(y_te[i], p_rf[i]) for i in idx])
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for j, (lab, d) in enumerate([("ΔROC-AUC (CNN − RF)", d_roc), ("ΔPR-AUC (CNN − RF)", d_pr)]):
    lo, hi, mu = np.percentile(d, 2.5), np.percentile(d, 97.5), d.mean()
    ax.errorbar(mu, j, xerr=[[mu - lo], [hi - mu]], fmt="o", ms=7, capsize=6, capthick=1.5,
                color="#3b82f6", ecolor="#1e40af")
    ax.text(hi + 0.0006, j, f"{mu:+.4f}  [{lo:+.4f}, {hi:+.4f}]", va="center", fontsize=9.5)
ax.axvline(0, ls="--", color="gray")
ax.set_yticks([0, 1]); ax.set_yticklabels(["ΔROC-AUC (CNN − RF)", "ΔPR-AUC (CNN − RF)"])
ax.set_xlabel("paired difference on test")
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.set_title("CNN − RF deltas with 95% bootstrap CIs (B = 2000)")
ax.set_xlim(-0.006, 0.026)
fig.savefig(A / "delta_ci.png", bbox_inches="tight"); plt.close(fig)
print(f"dROC={d_roc.mean():+.4f} dPR={d_pr.mean():+.4f}")

# ---- 9. error-group profiles (legend outside, readable labels) ----
te = pd.DataFrame(StandardScaler().fit(X_tr).transform(X_te), columns=feats)
yr, yc = (p_rf >= 0.5).astype(int), (p_cnnT >= 0.5).astype(int)
groups = {"both-right": (yr == y_te) & (yc == y_te), "RF-only-right": (yr == y_te) & (yc != y_te),
          "CNN-only-right": (yr != y_te) & (yc == y_te), "both-wrong": (yr != y_te) & (yc != y_te)}
prof = pd.DataFrame({g: te[m][feats].mean() for g, m in groups.items()}).T
pop = te[feats].mean()
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(len(feats))
colors = ["#3b82f6", "#f59e0b", "#16a34a", "#ef4444"]
for j, g in enumerate(prof.index):
    ax.bar(x + (j - 1.5) * 0.19, prof.loc[g].values - pop.values, 0.19, label=g, color=colors[j])
ax.set_xticks(x); ax.set_xticklabels(feats, rotation=20, ha="right")
ax.axhline(0, color="gray", lw=0.8)
ax.set_title("Error-group deviation from population mean (std units)")
ax.set_ylabel("deviation (std units)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
fig.savefig(A / "error_profile.png", bbox_inches="tight"); plt.close(fig)

# ---- 10. missed-pulsar scores (headroom, annotated) ----
pos = y_te == 1
fig, ax = plt.subplots(figsize=(7.2, 4.0))
n1, _, _ = ax.hist(p_rf[pos], bins=30, alpha=0.6, label="RF scores (pulsars)", color="#3b82f6")
n2, _, _ = ax.hist(p_cnnT[pos], bins=30, alpha=0.6, label="CNN scores (pulsars)", color="#f59e0b")
ax.axvline(0.5, color="red", ls="--", label="threshold (0.5)")
ymax = max(n1.max(), n2.max()) * 1.15
ax.set_ylim(0, ymax)
ax.text(0.99, 0.96, f"near 1.0: RF≈{int(n1[-1])}, CNN≈{int(n2[-1])} of {pos.sum()} pulsars",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", boxstyle="round,pad=0.4"))
ax.set_title("Test-pulsar score distributions")
ax.set_xlabel("p(pulsar)"); ax.set_ylabel("count")
ax.legend(fontsize=9)
fig.savefig(A / "missed_scores.png", bbox_inches="tight"); plt.close(fig)

print("wrote", sorted(p.name for p in A.glob("*.png")))
