# Pulsar Candidate Selection (HTRU2) — v1 + v2 CNN replication

Reproducible notebooks: Random Forest triage of pulsar candidates (v1) plus a 1D-CNN replication of the published CNN-on-HTRU2 experiment (v2), with data cards for HTRU2's provenance, class balance, and requested citation. Dataset-only project — no hardware.

## Contents

| File | Purpose |
|---|---|
| `pulsar_selection.ipynb` | v1 source notebook — run top-to-bottom (§1–10) |
| `pulsar_selection.executed.ipynb` | v1 executed copy with outputs and figures |
| `pulsar_selection_cnn.ipynb` | v2 CNN replication notebook (replicates UCSD ECE228 Report6 setup) |
| `pulsar_selection_cnn.executed.ipynb` | v2 executed copy |
| `cnn_grid.py` / `cnn_grid_results.json` / `cnn_best.pt` | v2 compute step: 32-config grid, selection log, winning weights |
| `data/HTRU_2.csv` | Cached HTRU2 data (re-downloaded automatically if missing) |
| `requirements.txt` | Pinned environment (torch CPU for v2) |
| `docs/` | GitHub Pages progress site (served from `/docs`) |

## Quickstart

```bash
pip install -r requirements.txt
# option A: interactive
jupyter notebook pulsar_selection.ipynb
# option B: headless re-execution
python -m jupyter nbconvert --to notebook --execute pulsar_selection.ipynb \
  --output pulsar_selection.executed.ipynb --ExecutePreprocessor.timeout=1200
```

Fixed `SEED=42` throughout. Split is stratified 60/20/20, created *before* any class weighting.

## Data (summary — full card in notebook §1–2)

- Source: HTRU-South survey via Parkes Observatory; donor R. J. Lyon (Univ. of Manchester); UCI ML Repository id 372, CC-BY 4.0.
- 17,898 candidates = **1,639 pulsars (9.16%)** + 16,259 non-pulsars. 8 continuous features (profile + DM–SNR statistics), no sky position.
- Metrics are ROC-AUC / PR-AUC / recall@cost-threshold — never accuracy-first (91% is achievable by predicting all-zero).

## Verified results (SEED=42, sklearn 1.9.1)

| Model | Test ROC-AUC | Test PR-AUC | Precision / Recall @ T=0.10 |
|---|---|---|---|
| Logistic regression (baseline) | 0.977 | 0.934 | — |
| Random Forest, calibrated (400 trees, depth 20) | 0.976 | 0.933 | 0.867 / 0.896 |

Honest misses: joint target (recall ≥ 0.90 at precision ≥ 0.90) not met on this split (best: 0.884); calibration improved val Brier but test Brier flat. Top permutation feature: `profile_kurtosis`. See notebook §10.

## v2 — 1D-CNN replication (torch 2.14.1 CPU, identical splits)

32-config grid over the published space (blocks × channels × kernel × pool × relu/tanh, SGD, val PR-AUC selection). Winner: 1 block, 16ch, k=2, no pool, ReLU. Test results, all rows measured on identical splits:

| Model | Acc | ROC-AUC | PR-AUC | Pos-prec / Pos-rec @0.5 |
|---|---|---|---|---|
| RF (v1, retrained) | 0.980 | 0.976 | 0.933 | 0.913 / 0.863 |
| MLP baseline | 0.974 | 0.968 | 0.914 | 0.940 / 0.768 |
| **1D-CNN (v2)** | 0.970 | **0.982** | **0.938** | 0.786 / **0.927** |
| Literature, quoted: Lyon 2016 GH-VFDT 0.978/0.899/0.829; Report6 CNN 0.984/0.993/0.975 |

Verdict: CNN wins ranking quality (ROC/PR-AUC) and recall, RF wins accuracy and calibration (Brier 0.0167 vs CNN 0.0284, uncalibrated). Our CNN does not reach Report6's numbers — gap documented in the notebook (their split/balancing unspecified; ours fixed and stricter). Temperature scaling is the named next step. Image-based systems (MICNN/PICS) are out of scope: HTRU2 has no plot pixels.

## Citation

- Dataset: Lyon, R. (2015). HTRU2 [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5DK6R
- Paper: R. J. Lyon et al., "Fifty Years of Pulsar Candidate Selection…", MNRAS 459(1), 2016.

## Limitations

Reproduces survey expert-vetting on one labelled dataset; discovers no new pulsars. Threshold depends on the stated 10:1 miss-vs-false-alarm cost — change the cost, re-pick T.

