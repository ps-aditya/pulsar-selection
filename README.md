# Pulsar Candidate Selection (HTRU2) — v1

Reproducible notebook: Random Forest triage of pulsar candidates (pulsar vs RFI/noise), with data cards for HTRU2's provenance, class balance, and requested citation. Dataset-only project — no hardware.

## Contents

| File | Purpose |
|---|---|
| `pulsar_selection.ipynb` | Source notebook — run top-to-bottom (§1–10) |
| `pulsar_selection.executed.ipynb` | Executed copy with outputs and figures |
| `data/HTRU_2.csv` | Cached HTRU2 data (re-downloaded automatically if missing) |
| `requirements.txt` | Pinned environment |

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

## Citation

- Dataset: Lyon, R. (2015). HTRU2 [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5DK6R
- Paper: R. J. Lyon et al., "Fifty Years of Pulsar Candidate Selection…", MNRAS 459(1), 2016.

## Limitations

Reproduces survey expert-vetting on one labelled dataset; discovers no new pulsars. Threshold depends on the stated 10:1 miss-vs-false-alarm cost — change the cost, re-pick T.

