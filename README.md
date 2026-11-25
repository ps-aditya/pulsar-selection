# Pulsar Candidate Selection (HTRU2) — Project 7

Classical machine-learning mainline: leakage-audited split, CV-tuned Random Forest, four-way calibration study selected by CV Brier, cost-model thresholding, single final test evaluation with bootstrap CIs and ECE. Supplementary tracks retained in the repo: 1D-CNN replication (v2), temperature scaling + significance (v3), error analysis (v4), ground-up NumPy reimplementations. Dataset-only — no hardware.

## Contents

| File | Purpose |
|---|---|
| `pulsar_selection.ipynb` | v1 source notebook — run top-to-bottom (§1–11): leakage audit, single final test evaluation |
| `pulsar_selection.executed.ipynb` | v1 executed copy with outputs and figures |
| `pulsar_selection_cnn.ipynb` | v2 CNN replication notebook (replicates UCSD ECE228 Report6 setup) |
| `pulsar_selection_cnn.executed.ipynb` | v2 executed copy |
| `cnn_grid.py` / `cnn_grid_results.json` / `cnn_best.pt` | v2 compute step: 32-config grid, selection log, winning weights |
| `pulsar_selection_v3.ipynb` | v3: temperature scaling + bootstrap CIs + McNemar |
| `pulsar_selection_v3.executed.ipynb` | v3 executed copy |
| `pulsar_selection_v4_errors.ipynb` | v4: disagreement + error analysis (discordant and jointly-missed pulsars, profiled by feature) |
| `pulsar_selection_v4_errors.executed.ipynb` | v4 executed copy |
| `from_scratch/` | Ground-up NumPy reimplementations: `logreg.py`, `trees.py`, `mlp.py`, `cnn1d.py` (~350 lines, no libraries inside the models) |
| `pulsar_selection_groundup.ipynb` (+ executed) | Side-by-side twin comparison: logreg ROC exact, tree 99.16% agreement, MLP gradcheck 2.5e-08, CNN gradcheck 6.6e-10, all metrics within band |
| `paper.md` / `paper.pdf` | Project 7 final report (v2.0, 13 pages A4 PDF): classical mainline with calibrated results, threshold reconciliation, error analysis + supplementary-tracks appendix |
| `VIVA_PREP.md` | Viva Q&A (Q1–Q14): split strategy, metric choice, calibration study, cost model, 0.54 vs 0.091 threshold reconciliation — each answer tied to a repo artifact |
| `DATA_CARD.md` | Data card: provenance, release, licence, acknowledgement, splits, 8 known biases |
| `METRIC_JUSTIFICATION.md` | Gate 1–5 milestone record, rejection of 90.84% accuracy trap, metric & 10:1 cost-model justification |
| `docs/poster.html` / `docs/poster.pdf` | A2 landscape poster (HTML + ready-to-print A2 PDF) |
| `docs/VIDEO_SCRIPT.md` / `docs/Pulsar_Project_Video.mp4` | 3-minute (180 s) shot-by-shot script and rendered 1280×720 HD MP4 video |
| `LICENSE` | MIT (code); data keep CC-BY 4.0 |
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
- **Data release / catalogue version:** UCI ML Repository id 372 (donated 13 Feb 2017); figshare mirror DOI `10.6084/m9.figshare.3080389.v1`.
- **Download date:** cached copy (`data/HTRU_2.csv`) committed to this repository on **2026-09-25**; the notebook re-downloads it automatically if missing.
- **Required acknowledgement text:** raw data acquired by the HTRU collaboration with the Parkes radio telescope, part of the Australia Telescope National Facility, operated by CSIRO; survey funded by UK EPSRC grant EP/I028099/1. Citation (both): Lyon, R. (2015). *HTRU2* [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5DK6R · Lyon et al., MNRAS 459(1), 2016.
- 17,898 candidates = **1,639 pulsars (9.16%)** + 16,259 non-pulsars. 8 continuous features (profile + DM–SNR statistics), no sky position.
- Metrics are ROC-AUC / PR-AUC / precision-recall at the cost-chosen threshold / Brier / ECE — never accuracy-first (91% is achievable by predicting all-zero).

## Verified results (SEED=42, sklearn 1.9.1)

| Model | Test ROC-AUC | Test PR-AUC | Precision / Recall @ T=0.10 |
|---|---|---|---|
| Logistic regression (baseline) | 0.977 | 0.934 | — |
| Random Forest (400 trees, depth 20), calibration selected by CV Brier | 0.9777 | 0.9332 | 0.9183 / 0.8567 @ T=0.54 |

Honest reporting: ROC-AUC ≥0.97 is met. The joint recall ≥0.90 at precision ≥0.90 target is not met at the cost-minimising threshold (recall 0.857). Calibration comparison on train+val selected **no calibration** (raw RF had the lowest cross-fitted CV Brier, 0.01721); sigmoid/isotonic did not improve out-of-sample Brier. ECE on test: 0.0099. Top permutation feature: `profile_kurtosis`. Test predictions, metrics, and persisted model hashes are written to `results/` during the single final evaluation (§10). See notebook §12.

## v2 — 1D-CNN replication (torch 2.14.1 CPU, identical splits)

32-config grid over the published space (blocks × channels × kernel × pool × relu/tanh, SGD, val PR-AUC selection). Winner: 1 block, 16ch, k=2, no pool, ReLU. Test results, all rows measured on identical splits:

| Model | Acc | ROC-AUC | PR-AUC | Pos-prec / Pos-rec @0.5 |
|---|---|---|---|---|
| RF (v1, retrained) | 0.980 | 0.976 | 0.933 | 0.913 / 0.863 |
| MLP baseline | 0.974 | 0.968 | 0.914 | 0.940 / 0.768 |
| **1D-CNN (v2)** | 0.970 | **0.982** | **0.938** | 0.786 / **0.927** |
| Literature, quoted: Lyon 2016 GH-VFDT 0.978/0.899/0.829; Report6 CNN 0.984/0.993/0.975 |

Verdict: CNN wins ranking quality (ROC/PR-AUC) and recall, RF wins accuracy and calibration (Brier 0.0167 vs CNN 0.0284, uncalibrated). Our CNN does not reach Report6's numbers — gap documented in the notebook (their split/balancing unspecified; ours fixed and stricter). Temperature scaling is the named next step. Image-based systems (MICNN/PICS) are out of scope: HTRU2 has no plot pixels.

## v3 — temperature scaling + paired significance

- Temperature T=0.718 (val NLL fit): test NLL 0.1228→0.1119, Brier 0.0284→0.0258. Gap to RF (0.0167) narrowed, not closed.
- Bootstrap 95% CIs (B=2000, paired) include 0 for both deltas: ΔROC +0.0058 [−0.0003, +0.0128], ΔPR +0.0051 [−0.0030, +0.0138]. Ranking lead suggestive, not significant.
- McNemar exact p=0.0001 at 0.5 threshold — decisions differ significantly, **in RF's favour** (RF-only-right 56 vs CNN-only-right 21).
- Paper claim, exactly: *CNN ranks marginally better, RF decides significantly better.* Anything stronger is unsupported.

## v4 — error analysis + paper draft

Correctness partition of the test set: both-right 3,452; RF-only-right 56 (**all noise** — CNN false alarms); CNN-only-right 21 (**all pulsars** — RF's lost discoveries); both-wrong 51 (24 pulsars + 27 noise). Every CNN pulsar-miss is also an RF miss; CNN recovers 21 RF loses, loses none RF keeps. Both-missed pulsars are feature-atypical (near population mean while found pulsars hit profile kurtosis +2.9σ); missed-pulsar median scores ~0.1 — confidently wrong, unrecoverable by threshold moves.

`paper.md` (report v1.0): the classical mainline (leakage audit, calibration study, cost-model thresholding, single test evaluation with CIs + ECE) with an appendix retaining the CNN replication, temperature scaling, error analysis, and ground-up rebuild as supplementary tracks.

**Ablation (winning CNN, same protocol):** profile-only ROC 0.977 / PR 0.925; DM-only ROC 0.896 / PR 0.373; full-8 0.982 / 0.938. Profile carries the signal, DM refines it. (`cnn_ablation.py` → `cnn_ablation.json`)

## Licence

- Code (notebooks, `from_scratch/`, scripts, docs): MIT, see `LICENSE`.
- Data (`data/HTRU_2.csv`): not ours. HTRU2 is CC-BY 4.0 (UCI ML Repository id 372). Keep the acknowledgement and citation above when reusing it.

## Citation

- Dataset: Lyon, R. (2015). HTRU2 [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5DK6R
- Paper: R. J. Lyon et al., "Fifty Years of Pulsar Candidate Selection…", MNRAS 459(1), 2016.

## Limitations

Reproduces survey expert-vetting on one labelled dataset; discovers no new pulsars. Threshold depends on the stated 10:1 miss-vs-false-alarm cost — change the cost, re-pick T.

