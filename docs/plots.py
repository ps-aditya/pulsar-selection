"""Regenerate docs/assets figures deterministically (SEED=42, best-params RF)."""
from pathlib import Path
import urllib.request, zipfile, io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance

SEED = 42
rng = np.random.seed(SEED)
A = Path(__file__).parent / "assets"
A.mkdir(parents=True, exist_ok=True)
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
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=SEED)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=SEED)
P = dict(n_estimators=400, max_depth=20, min_samples_leaf=2,
         class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
base = RandomForestClassifier(**P).fit(Xtr, ytr)
cal = CalibratedClassifierCV(estimator=FrozenEstimator(base), method="sigmoid").fit(Xva, yva)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

bal = df["target"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.bar(["non-pulsar (0)", "pulsar (1)"], [bal[0], bal[1]])
ax.set_title("HTRU2 class balance (n=17,898)")
ax.set_ylabel("count")
for i, v in enumerate([bal[0], bal[1]]):
    ax.text(i, v, str(v), ha="center", va="bottom")
fig.tight_layout(); fig.savefig(A / "balance.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 4))
ax.plot([0, 1], [0, 1], ls="--", label="ideal")
for name, p in [("rf", base.predict_proba(Xte)[:, 1]), ("rf+cal", cal.predict_proba(Xte)[:, 1])]:
    ft, mp = calibration_curve(yte, p, n_bins=10, strategy="quantile")
    ax.plot(mp, ft, marker="o", ms=4, label=name)
ax.set_title("Reliability diagram (test)")
ax.set_xlabel("mean predicted"); ax.set_ylabel("fraction positive")
ax.legend(); fig.tight_layout(); fig.savefig(A / "reliability.png", bbox_inches="tight"); plt.close(fig)

lc = RandomForestClassifier(**P)
tr_n, tr_s, va_s = learning_curve(lc, Xtr, ytr,
    cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
    train_sizes=np.linspace(0.1, 1.0, 5), scoring="average_precision", n_jobs=-1)
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(tr_n, tr_s.mean(1), marker="o", label="train PR-AUC")
ax.plot(tr_n, va_s.mean(1), marker="o", label="cv PR-AUC")
ax.set_title("Learning curve (RF, PR-AUC)")
ax.set_xlabel("train size"); ax.legend()
fig.tight_layout(); fig.savefig(A / "learning_curve.png", bbox_inches="tight"); plt.close(fig)

pi = permutation_importance(cal, Xte, yte, n_repeats=10, random_state=SEED,
                            n_jobs=-1, scoring="average_precision")
order = np.argsort(pi.importances_mean)[::-1]
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.barh([feats[i] for i in order], pi.importances_mean[order], xerr=pi.importances_std[order])
ax.set_title("Permutation importance (test PR-AUC drop)")
fig.tight_layout(); fig.savefig(A / "importance.png", bbox_inches="tight"); plt.close(fig)
print("wrote", sorted(p.name for p in A.glob("*.png")))
