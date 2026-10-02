# Replicating CNN-based pulsar candidate selection on HTRU2: a calibrated comparison against tree ensembles (draft v0.1)

> Working paper draft — every number below is measured in the accompanying notebooks (SEED=42, fixed stratified splits). Literature values are quoted, not recomputed. Status: draft for internal review, not submitted anywhere.

## Abstract

We replicate the published 1D-CNN approach to pulsar candidate selection on the HTRU2 benchmark (17,898 candidates, 9.16% pulsars) under a fixed, stratified protocol, and compare it against a tuned Random Forest (RF) and an MLP baseline on identical splits. A 32-configuration grid over the published search space (conv blocks × channels × kernel × pooling × ReLU/tanh, SGD, validation PR-AUC selection) selects a 1-block, 16-channel, kernel-2, unpooled ReLU network. The CNN leads on ranking metrics (test ROC-AUC 0.982, PR-AUC 0.938 vs RF 0.976/0.933), but paired bootstrap 95% CIs include zero for both deltas — the lead is suggestive, not significant. At the 0.5 decision threshold the models differ significantly (McNemar exact p=0.0001), in the RF's favour. Error analysis reveals a stark asymmetry: all 56 CNN-only errors are false alarms while all 21 RF-only errors are lost pulsars; 24 pulsars (7.3%) are missed by both, are feature-atypical, and score ~0.1 — confidently wrong, hence unrecoverable by threshold moves. Temperature scaling (T=0.718) halves the CNN's calibration gap without closing it. Verdict: the CNN ranks marginally better, the RF decides significantly better.

## 1. Introduction

Large radio surveys (HTRU, GBNCC, FAST) produce far more pulsar candidates than humans can vet, and nearly all are radio-frequency interference (RFI) or noise. Automated triage — reproduce the expert's pulsar-vs-noise decision — is the bottleneck (Lyon et al. 2016). Pulsars matter beyond surveys: their interiors probe otherwise unreachable states of matter, millisecond pulsars anchor pulsar-timing-array gravitational-wave searches, and historic pulsar timing gave the first indirect detection of gravitational waves.

Prior work spans statistical filters (Lyon et al. 2016, GH-VFDT: accuracy 0.978, precision 0.899, recall 0.829), ANNs on survey features (Eatough et al. 2010; Bates et al. 2012; SPINN, Morello et al. 2014 — built for maximum recall), 1D-CNNs on HTRU2 feature vectors (UCSD ECE228 Report6, 2019: CNN accuracy 0.9837, positive precision 0.9926, positive recall 0.9746), and image-based deep systems consuming raw diagnostic plots (MICNN, Lin et al. 2020: recall 0.962, precision 0.967; PICS and descendants). The image-based line is not replicable from HTRU2, which ships 8 statistics per candidate and no pixels — we cite it as a boundary, not a baseline.

Contribution: (i) a strict replication of the tabular 1D-CNN experiment on fixed stratified splits shared with all baselines; (ii) post-hoc calibration with honest reporting (gap narrowed, not closed); (iii) paired significance testing that overturns the naive "CNN wins" reading; (iv) a feature-level error analysis with an astrophysical read.

## 2. Data

HTRU2 (Lyon, UCI ML Repository id 372, CC-BY 4.0; DOI 10.24432/C5DK6R): 17,898 human-verified candidates — 1,639 pulsars (9.16%), 16,259 RFI/noise. Eight continuous features: mean, standard deviation, excess kurtosis, skewness of the folded integrated profile, and the same four of the DM–SNR curve (PulsarFeatureLab extraction; no sky position retained). No missing values. Requested citation: Lyon et al., "Fifty Years of Pulsar Candidate Selection…", MNRAS 459(1), 2016.

Protocol: single stratified 60/20/20 split (train/validation/test, SEED=42) created before any weighting; test n=3,580, positive rate 0.0916. Class imbalance is handled by explicit inverse-frequency class weights (neural nets) and `balanced_subsample` (RF). Accuracy is never the headline metric (91% is free by predicting all-zero).

## 3. Method

- **Logistic regression** (standardised, `balanced`): linear separability baseline.
- **Random Forest**: 400 trees, depth 20, min-samples-leaf 2, selected by 3-fold stratified grid on PR-AUC; Platt (sigmoid) calibration on the validation fold.
- **MLP**: sklearn (100,) hidden units, early stopping — literature-parity baseline only.
- **1D-CNN** (replication): 8-feature vector as a length-8 1-channel sequence; grid over blocks {1,2} × channels {8,16} × kernel {2,3} × max-pool {on,off} × {ReLU,tanh} = 32 configs, SGD lr=0.01 momentum 0.9, batch 256, early stopping on validation cross-entropy, selection by validation PR-AUC. Winner: 1 block, 16 channels, kernel 2, no pooling, ReLU (val PR-AUC 0.9325; ReLU beat tanh in every pairing).
- **Operating threshold**: from a written 10:1 miss-vs-false-alarm cost model on validation, never default 0.5 for reporting (0.5 retained only where McNemar requires a fixed decision rule).
- **Calibration**: single-parameter temperature scaling fit on validation NLL (test touched once).
- **Significance**: paired bootstrap (B=2000) 95% CIs on ΔROC-AUC/ΔPR-AUC; exact McNemar on discordant 0.5-threshold decisions.

## 4. Results

| Model | Acc | ROC-AUC | PR-AUC | Brier | Pos-prec / Pos-rec @0.5 |
|---|---|---|---|---|---|
| LogReg | — | 0.977 | 0.934 | 0.0318 | — |
| RF (calibrated) | 0.980 | 0.976 | 0.933 | 0.0167 | 0.913 / 0.863 |
| MLP | 0.974 | 0.968 | 0.914 | 0.0203 | 0.940 / 0.768 |
| 1D-CNN (raw) | 0.970 | 0.982 | 0.938 | 0.0284 | 0.786 / 0.927 |
| 1D-CNN (T=0.718) | 0.970 | 0.982 | 0.938 | 0.0258 | — |

(Ranking invariant to temperature scaling, as expected; NLL 0.1228→0.1119.)

**Significance.** ΔROC-AUC (CNN−RF) +0.0058, 95% CI [−0.0003, +0.0128]; ΔPR-AUC +0.0051, [−0.0030, +0.0138]. Both include zero: the ranking lead is not significant (P(Δ>0) = 0.964/0.889). McNemar exact p=0.0001: decisions differ significantly — with 56 RF-only-correct vs 21 CNN-only-correct, in the RF's favour.

**Error analysis.** Correctness partition of the test set: both-right 3,452; RF-only-right 56 (all noise — CNN false alarms); CNN-only-right 21 (all pulsars — RF's lost discoveries); both-wrong 51 (24 pulsars + 27 noise). Every CNN pulsar-miss is also an RF miss; the CNN recovers 21 pulsars the RF loses and loses none it keeps. Both-missed pulsars sit near the population mean on all 8 features while found pulsars are extreme (profile kurtosis +2.9σ, skewness +2.6σ) — weak, atypical profiles; CNN-recovered pulsars show elevated DM features (dm_stdev +1.3σ), implicating the DM-confirmation pathway. Missed-pulsar median scores are 0.09–0.10 under both models: confidently wrong, so no threshold move recovers them.

**Ablation (winning CNN, same protocol).** Profile-only inputs: ROC-AUC 0.9774, PR-AUC 0.9253. DM-only: ROC-AUC 0.8960, PR-AUC 0.3725 (early-stopped, 6 epochs). Full-8: 0.9818/0.9378. The profile carries the headline signal; DM features contribute the refinement (+0.004 ROC, +0.013 PR) — consistent with the DM-confirmation pathway in the error analysis.

## 5. Discussion

Three honest gaps. First, our CNN trails Report6's CNN (0.984/0.993/0.975): their split and balancing are undescribed, ours are fixed and strict — we report the gap rather than tune it away. Second, calibration improves but does not reach the tree ensemble. Third, the headline-correct paper claim is the split verdict (CNN ranks marginally better, RF decides significantly better), not a victory narrative.

What would change the picture: raw diagnostic-plot inputs (the MICNN/PICS line) for the 24 atypical missed pulsars, since feature-space information appears exhausted for them; temperature scaling variants (vector/matrix) and threshold-free decision analysis as immediate follow-ups.

## 6. Limitations

Single labelled survey slice; reproduces expert vetting, discovers no new pulsars; threshold and cost model are stated choices, not physical constants; compute is CPU-only with fixed seeds (re-runs are deterministic in this environment).

## References

Lyon et al., MNRAS 459(1), 2016 (introductory paper + dataset). Lyon, PhD thesis, Univ. of Manchester, 2015 (feature definitions). Keith et al., MNRAS 409, 2010 (HTRU survey). Eatough et al., MNRAS 407, 2010; Bates et al., MNRAS 427, 2012; Morello et al. (SPINN), MNRAS 443, 2014; UCSD ECE228 Report6, 2019 (replication target); Lin et al., arXiv:2007.14843, 2020 (boundary); Zhu et al./PICS line (boundary).

## Reproducibility

`pulsar_selection.ipynb` (v1) → `pulsar_selection_cnn.ipynb` + `cnn_grid.py` (v2) → `pulsar_selection_v3.ipynb` (v3) → `pulsar_selection_v4_errors.ipynb` (v4). SEED=42 everywhere; `requirements.txt` pinned (torch CPU); executed copies committed; figures regenerate via `docs/plots.py` (v1) and the notebooks themselves (v2–v4).
