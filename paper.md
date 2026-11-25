# Calibrated, cost-sensitive pulsar candidate selection on HTRU2: a classical machine learning study (report v1.0)

> **Status.** Project 7 report. Every number below is measured in the accompanying notebooks on fixed SEED=42 stratified splits; literature values are quoted, not recomputed. `results/final_metrics.json` is the machine-readable source of the final test numbers.

## Abstract

We build a complete classical machine-learning pipeline for pulsar candidate selection on the HTRU2 benchmark (17,898 human-verified candidates from the High Time Resolution Universe Survey South, 9.16% pulsars). The pipeline is leakage-controlled end to end: a stratified 60/20/20 train/validation/test split is created before any weighting, and a code-printed audit verifies zero duplicate rows overall and across splits, pairwise-disjoint index sets, a scaler fitted on train only via a `Pipeline`, and a calibration fold disjoint from train and test. Against a standardised logistic-regression baseline (validation ROC-AUC 0.9744), a Random Forest (400 trees, depth 20, min-samples-leaf 2) is selected by 3-fold stratified CV on PR-AUC. A four-way calibration study — no calibration, sigmoid, isotonic, and 5-fold cross-fitted sigmoid — is run with repeated stratified CV (5×2) on train+val only and selects **no calibration** (cross-fitted CV Brier 0.01721 vs 0.01721/0.01733/0.01723; isotonic additionally degrades ranking). The operating threshold is chosen from a written 10:1 miss-vs-false-alarm cost model on train+val (T = 0.54, where the model achieves zero missed pulsars at 42 false alarms on the selection set); a sensitivity analysis across 5:1/10:1/20:1 cost ratios selects the same threshold. The test set is evaluated exactly once: **ROC-AUC 0.9777** (95% CI 0.9671–0.9872), **PR-AUC 0.9332** (0.9117–0.9542), **precision 0.9183** (0.8889–0.9461), **recall 0.8567** (0.8201–0.8933), **Brier 0.0166** (0.0135–0.0199), **ECE 0.0099** (B = 2000 paired stratified bootstrap, SEED = 42). The recall ≥ 0.90 at precision ≥ 0.90 target is honestly not met at the cost-minimising threshold. Permutation importance is dominated by `profile_kurtosis` (PR-AUC drop 0.633), consistent with pulse-shape physics; DM statistics refine rather than carry the signal.

## 1. Introduction

Large radio surveys produce far more pulsar candidates than humans can vet, and nearly all are radio-frequency interference (RFI) or noise. Automated triage — reproducing the expert's pulsar-vs-noise decision — is the bottleneck of modern searches (Lyon et al. 2016). Pulsars matter beyond surveys: their interiors probe otherwise unreachable states of matter, millisecond pulsars anchor pulsar-timing-array gravitational-wave searches, and historic pulsar timing gave the first indirect detection of gravitational waves.

Prior work spans statistical filters (Lyon et al. 2016, GH-VFDT: accuracy 0.978, precision 0.899, recall 0.829), ANNs on survey features (Eatough et al. 2010; Bates et al. 2012; SPINN, Morello et al. 2014 — built for maximum recall), 1D-CNNs on HTRU2 feature vectors (UCSD ECE228 Report6, 2019), and image-based deep systems consuming raw diagnostic plots (MICNN, Lin et al. 2020; the PICS line). The image-based line is not replicable from HTRU2, which ships eight statistics per candidate and no pixels; we cite it as a boundary, not a baseline.

**This report's contribution** is the classical-ML track of the project, executed to the brief's letter:

1. a profiled, documented dataset with the full 8×8 feature-correlation structure;
2. a code-printed **leakage audit** and a protocol in which the test set is evaluated **exactly once**;
3. a fair four-way **calibration study** selected by cross-validated Brier on train+val only;
4. a written **cost model** with a plotted expected-cost curve, a marked operating point, and a cost-ratio sensitivity analysis;
5. a single final test evaluation with **paired bootstrap confidence intervals** (B = 2000) and **ECE**;
6. an astrophysical reading of feature importances on the validation set.

The project's supplementary deep-learning track (1D-CNN replication, temperature scaling, error analysis) and the ground-up NumPy reimplementations of all model families are summarised in Appendix A; the main line of this report is deliberately classical.

## 2. Data

**HTRU2** (Lyon, UCI ML Repository id 372; CC-BY 4.0; DOI `10.24432/C5DK6R`): 17,898 human-verified candidates — **1,639 pulsars (9.16%)** and 16,259 RFI/noise. Eight continuous features: mean, standard deviation, excess kurtosis, and skewness of the folded integrated profile, plus the same four statistics of the DM–SNR curve (`PulsarFeatureLab` extraction; no sky position retained). No missing values. Required citation: Lyon, R. (2015), UCI ML Repository; and Lyon et al., "Fifty Years of Pulsar Candidate Selection…", MNRAS 459(1), 2016. Raw data: HTRU collaboration, Parkes Observatory (CSIRO); UK EPSRC grant EP/I028099/1.

**Class balance.** Imbalance ≈ 9.9:1. Plain accuracy is worthless (91% is free by predicting all-zero); the protocol reports ROC-AUC, PR-AUC, precision/recall at a cost-chosen threshold, Brier, and ECE — never accuracy-first.

**Feature correlation structure (8×8, code-printed).** The strongest pairwise correlations are `profile_mean`–`profile_kurtosis` −0.874, `profile_skewness`–`profile_kurtosis` +0.946, `dm_mean`–`dm_stdev` +0.797, `dm_stdev`–`dm_kurtosis` −0.810, and `dm_skewness`–`dm_kurtosis` +0.924. Median absolute correlation with the target is 0.446. Profile statistics form one tightly coupled block and DM statistics another — consistent with their physical origin (folded-profile shape vs dispersion sweep) and with the expectation that no single feature suffices.

**Leakage audit (all numbers code-printed, §6b of the notebook).**

| Check | Result |
|---|---|
| Exact duplicate rows overall | **0** |
| Duplicate rows train↔val / train↔test / val↔test | **0 / 0 / 0** |
| Index-set disjointness | asserted OK |
| Class ratios (pos rate) train / val / test | 0.091544 / 0.091620 / 0.091620 |
| Scaler fitted on train only (`Pipeline`, `n_samples_seen_` = 10,738 = train size) | asserted OK |
| Calibration fold disjoint from train and test | asserted OK |

## 3. Method

**Split.** Single stratified 60/20/20 split (SEED = 42), created before any weighting; test n = 3,580 at the dataset's native 9.16% positive rate, retained as-is under all circumstances. Class imbalance is handled inside training only (`balanced`/`balanced_subsample` weights).

**Baseline.** Standardised logistic regression (`balanced`), trained on train.

**Random Forest (main model).** Hyperparameters selected by 3-fold stratified grid CV on train with PR-AUC scoring: **400 trees, max_depth 20, min_samples_leaf 2**, `class_weight="balanced_subsample"`, n_jobs = −1. Validation diagnostics: ROC-AUC 0.9721, PR-AUC 0.9240, Brier 0.0169 (vs baseline 0.9744 / 0.9230 / 0.0331).

**Calibration study (selection on train+val only, never test).** Four strategies compared under repeated stratified CV (5 splits × 2 repeats) on the combined train+val set:

1. **none** — raw RF probabilities;
2. **sigmoid** — Platt scaling, assessed with 3-fold cross-fitting on each CV validation fold;
3. **isotonic** — same 3-fold cross-fitting;
4. **cross** — 5-fold cross-fitted sigmoid (a more conservative calibrator).

Cross-fitting is used throughout so no calibrator is scored on the data it was fit to. Selection criterion: **lowest mean CV Brier**.

**Cost model (written).** Missing a real pulsar (FN) costs follow-up discovery; a false alarm (FP) costs wasted telescope time. Set `C_FN=10, C_FP=1` (10:1) — a modelling choice to defend, not a physical constant. The operating threshold minimises `10·FN + 1·FP` over a 91-point grid on 0.05–0.95 on train+val.

**Final evaluation.** The test set is touched exactly once: point estimates, ECE, and paired stratified bootstrap 95% CIs (B = 2000, SEED = 42) for ROC-AUC, PR-AUC, precision@T, recall@T, Brier (raw and calibrated) and ΔBrier. Predictions are written to `results/test_predictions.parquet`; all later analysis reads the saved file.

**Interpretation.** Learning curve and permutation importance on train/validation only.

## 4. Results

### 4.1 Calibration study — no calibration wins

| Method | Mean CV Brier (↓) | Mean CV ROC-AUC (↑) |
|---|---:|---:|
| **none** | **0.01721 ± 0.00214** | **0.97422** |
| sigmoid | 0.01721 ± 0.00235 | 0.97000 |
| isotonic | 0.01733 ± 0.00248 | 0.96885 |
| cross (5-fold) | 0.01723 ± 0.00236 | 0.96959 |

`none` is selected by CV Brier (and also has the best CV ROC-AUC — isotonic's piecewise-constant mapping collapses raw probabilities onto few unique values, costing ranking quality). The final model is therefore the **uncalibrated RF trained on full train+val** (n = 14,318, pos rate 0.0916); sigmoid/isotonic calibrators fitted to this RF do not improve out-of-sample Brier. The brief's "demonstrably improved Brier after calibration" success criterion is **not met**, and the reason is now measured rather than assumed: with `balanced_subsample` weighting the raw forest is already close to calibrated on this problem (test ECE 0.0099), and any calibrator fitted on 14,318 samples adds variance without removing bias.

### 4.2 Threshold economics

The expected-cost curve on train+val has its minimum at **T = 0.54**: at that threshold the selection set has **FN = 0, FP = 42** (cost 42), i.e. every labelled pulsar in train+val is retained at 42 false alarms. The cost-ratio sensitivity table (thresholds chosen on train+val; metrics computed on test):

| Cost ratio | Chosen T | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|
| 5:1 | 0.54 | 0.9183 | 0.8567 | 0.8864 |
| 10:1 | 0.54 | 0.9183 | 0.8567 | 0.8864 |
| 20:1 | 0.54 | 0.9183 | 0.8567 | 0.8864 |

All three ratios select the same threshold because at T = 0.54 the selection set has zero false negatives, so cost is driven only by false positives. The cost ranking, not the ratio, is what separates this from a default-0.5 policy: at T = 0.54 the test confusion matrix is **TN 3227 · FP 25 · FN 47 · TP 281**, versus far more false alarms at lower thresholds.

### 4.3 Final test evaluation (single pass)

| Metric | Point estimate | 95% CI (B = 2000, paired) |
|---|---:|---|
| ROC-AUC | **0.9777** | 0.9671 – 0.9872 |
| PR-AUC | **0.9332** | 0.9117 – 0.9542 |
| Precision @ T = 0.54 | **0.9183** | 0.8889 – 0.9461 |
| Recall @ T = 0.54 | **0.8567** | 0.8201 – 0.8933 |
| Brier | **0.0166** | 0.0135 – 0.0199 |
| ΔBrier (raw − calibrated) | 0.0000 | −0.0000 – 0.0000 |
| ECE | **0.0099** | — |

Success criteria against the brief: **ROC-AUC ≥ 0.97 — met** (point estimate and entire CI above 0.967). **Recall ≥ 0.90 at precision ≥ 0.90 — not met** at the cost-minimising operating point (precision comfortably above 0.90 at 0.9183; recall 0.8567, CI top 0.8933). Both failures and passes are reported as-is.

### 4.4 Interpretation — what the model listens to

Permutation importance (validation set, PR-AUC drop, 10 repeats):

`profile_kurtosis` **0.633** ≫ `profile_skewness` 0.037 > `dm_stdev` 0.020 > `profile_mean` 0.016 > `dm_mean` 0.008 ≈ `dm_skewness` 0.008 > `profile_stdev` 0.008 > `dm_kurtosis` 0.006.

The astrophysical reading, stated physically: excess kurtosis of the folded integrated profile measures how sharply the pulse profile peaks. A real pulsar is a rotating neutron star whose radio beam sweeps the telescope once per rotation — the folded profile is a sharp, peaked, repeating blink against a smooth noise background, so pulsar candidates sit far out on the kurtosis (and skewness) axis while RFI/noise folds to a flat, near-Gaussian profile. The DM–SNR statistics measure the sweep across dispersion measure: a genuine astrophysical signal is dispersed in a physically lawful way, which is why the DM block refines the decision (its permutation drops are positive) without carrying the headline — exactly the detect-then-confirm structure of real survey pipelines (profile-based detection, DM-based confirmation). The learning curve (3-fold CV inside train, up to 7,159 labels, i.e. 2/3 of train) rises to about 0.92 PR-AUC by roughly 5,500 labels and then flattens. Within the range tested, more labels give little gain, so the current ceiling looks like information rather than sample size; the curve does not show whether more labels would help beyond 7,159.

## 5. Discussion

**Why "no calibration" is a finding, not a failure.** The calibration study was designed to satisfy a brief that expects calibrators to improve Brier. The measured answer is that on this dataset, with class-weighted forests, Platt/isotonic calibrators fit on ~14k samples change the Brier by less than the CV noise band, and isotonic actively harms ranking by collapsing the score distribution. We report that honestly; we do not claim the threshold is a textbook posterior cut. For calibrated probabilities the cost-optimal rule would be C_FP/(C_FP+C_FN) = 1/11 ≈ 0.09, yet the measured expected-cost minimum on train+val is at 0.54. The gap is an open question, reported rather than explained away; the threshold is the empirical cost minimum, not the posterior rule.

**Protocol evolution, for the record.** The project's earlier dossier run (RF trained on train only, Platt fitted on the validation fold) had chosen T = 0.10 with test precision 0.867 / recall 0.896 and a flat test Brier (0.0167 → 0.0171). That earlier run computed test metrics, so the test set was not untouched at that stage; the numbers are retained for traceability, and the final model, calibration choice, and threshold are chosen on train+val only. The restructure to single-test-evaluation moved selection fully onto train+val and surfaced the cost curve above; T moved from 0.10 to 0.54 and precision rose about five points (0.867 → 0.918) at the price of about four points of recall (0.896 → 0.857) — a better operating point under the stated 10:1 cost, though not a strictly dominant one. This traceability is retained deliberately.

**Relation to literature.** Quoted for context, not recomputed: GH-VFDT (Lyon et al. 2016) accuracy 0.978 / precision 0.899 / recall 0.829; Report6 CNN (2019) accuracy 0.9837 / positive precision 0.9926 / positive recall 0.9746. Our RF reaches comparable ranking (ROC-AUC 0.9777) with a fully specified, leakage-audited split and cost-based operating point — the protocol is stricter than most published HTRU2 numbers, which typically do not describe their splits.

**What would change the picture.** The supplementary error analysis (Appendix A) shows a residue of feature-atypical pulsars that any feature-space model scores near 0.1 — confidently wrong and unrecoverable by threshold moves; raw diagnostic-plot inputs (the MICNN/PICS line) are the natural next modality, and HTRU2 does not ship them.

## 6. Limitations

- Reproduces survey expert-vetting on one labelled dataset; discovers no new pulsars.
- No position or flux information; the RFI environment is HTRU-South's only.
- The 10:1 cost model is a stated choice, not a physical constant — change the cost, re-pick T (the sensitivity table brackets 5:1–20:1).
- The success criterion on joint recall/precision is not met at the cost-optimal threshold; meeting it would require either a different cost ratio (accepting more false alarms) or more informative features.
- CPU-only compute with fixed seeds; re-runs are deterministic in the pinned environment.

## 7. Conclusion

A leakage-audited, cost-aware classical pipeline — stratified split, CV-tuned Random Forest, four-way cross-validated calibration study, expected-cost thresholding, single-pass test evaluation — reproduces expert pulsar triage on HTRU2 at ROC-AUC 0.9777 / PR-AUC 0.9332 with precision 0.918 at the cost-optimal threshold, and demonstrates that on this benchmark the honest calibration answer is "the raw class-weighted forest is already calibrated". Every claim is traceable to a notebook cell, a printed number, or a saved artifact with a recorded SHA-256 hash.

## References

- Lyon, R. (2015). *HTRU2* [Dataset]. UCI Machine Learning Repository. DOI 10.24432/C5DK6R.
- Lyon, R. J. et al. "Fifty Years of Pulsar Candidate Selection: From Simple Filters to Modern Machine Learning." MNRAS 459(1), 2016.
- Lyon, R. J. PhD thesis, University of Manchester, 2015 (feature definitions).
- Keith, M. J. et al. "The High Time Resolution Universe Pulsar Survey — I." MNRAS 409, 2010.
- Eatough, R. P. et al. MNRAS 407, 2010; Bates, S. D. et al. MNRAS 427, 2012.
- Morello, V. et al. (SPINN). MNRAS 443, 2014.
- UCSD ECE228 Report6, 2019 (CNN-on-HTRU2 replication target — Appendix A).
- Lin, H.-W. et al. MICNN, arXiv:2007.14843, 2020 (boundary: requires plot pixels).

---

## Appendix A — supplementary tracks (measured, demoted from the headline)

The repository retains four supplementary studies, each on the same SEED=42 splits; they are supporting evidence for this report's classical line, not its main results:

1. **1D-CNN replication (v2)** — 32-config grid over the published search space; winner 1 block/16ch/k=2/unpooled/ReLU. On the v1-era RF's splits the CNN led ranking metrics (ROC 0.982 / PR 0.938 vs RF 0.976 / 0.933) but the lead was not significant (paired bootstrap CIs included zero) and the RF won 0.5-threshold decisions significantly (McNemar exact p = 0.0001): *the CNN ranks marginally better, the RF decides significantly better.*
2. **Temperature scaling (v3)** — T = 0.718 halves the CNN's calibration gap (Brier 0.0284 → 0.0258) without closing it (RF 0.0167).
3. **Error analysis (v4)** — of the CNN/RF discordant cases, all 56 RF-only-correct were CNN false alarms and all 21 CNN-only-correct were RF's lost pulsars; 24/328 test pulsars were missed by both, feature-atypical with median scores ~0.1 — unrecoverable by threshold moves.
4. **Ground-up NumPy rebuild** — logistic regression, decision tree, random forest, MLP and 1D-CNN reimplemented from scratch (~350 lines, no library code inside the models) and verified against their library twins: weight cosine 0.999997, tree decision agreement 99.16%, gradient checks 2.5e-08 (MLP) and 6.6e-10 (CNN), all metrics within ~0.002 of the twins.

## Appendix B — reproducibility

- Environment: Python 3.14.6, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.2, matplotlib 3.11.2 (pinned in `requirements.txt`); joblib + pyarrow for persistence.
- One-command re-execution: `python -m jupyter nbconvert --to notebook --execute pulsar_selection.ipynb --output pulsar_selection.executed.ipynb --ExecutePreprocessor.timeout=1800`.
- Final artifacts (written once by §10): `results/test_predictions.parquet`, `results/final_metrics.json`, `results/rf_best.joblib`, `results/rf_calibrated.joblib`; SHA-256 hashes recorded in the executed notebook. The `results/` folder is listed in `.gitignore` and is not in the repository; it is regenerated by re-running `pulsar_selection.ipynb`.
- Site figures regenerate deterministically via `python docs/plots_all.py` (SEED=42).
- Data release record: UCI ML Repository id 372 (donated 13 Feb 2017); figshare mirror DOI `10.6084/m9.figshare.3080389.v1`; cached copy committed to this repository on 2026-09-25 and re-downloaded automatically if missing.
