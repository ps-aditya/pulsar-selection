# Viva prep — split strategy & metric choice (and everything else they will ask)

Every answer below is grounded in a specific repo artifact. Say the artifact, then the number.

## 1. "Why did you split 60/20/20 stratified, and before any weighting?"

**Artifact:** `pulsar_selection.ipynb` §6 + §6b (code-printed audit).

- Stratified so every split keeps the dataset's 9.16% positive rate (audit prints 0.091544 / 0.091620 / 0.091620 — identical to 4 decimals).
- Order matters: resampling or class-weighting before splitting leaks test-distribution information into training and inflates scores. Weights enter only inside model fitting.
- The audit asserts pairwise-disjoint index sets, counts **0** exact duplicate rows overall and across all three split pairs, proves the scaler saw exactly 10,738 rows (train size, via `n_samples_seen_` in a `Pipeline`), and asserts the calibration fold is disjoint from train and test.

## 2. "Why these metrics and not accuracy?"

**Artifact:** §2 data card; `results/final_metrics.json`.

- Class balance is 1,639 : 16,259 (≈9.9:1). Predicting all-zero scores 90.84% accuracy for zero recall — accuracy is worthless here.
- We report ROC-AUC (ranking, prevalence-independent), PR-AUC (sensitive to the rare class — the honest headline under imbalance), precision/recall at the cost-chosen threshold (the decision the survey would actually make), Brier + reliability diagram + ECE (probability quality), all with paired bootstrap 95% CIs (B = 2000, stratified, SEED = 42).

## 3. "Defend the calibration study."

**Artifact:** §8; §4.1 of `paper.md`.

- Four strategies: none / sigmoid / isotonic / 5-fold cross-fitted sigmoid. Repeated stratified CV (5×2) on train+val only; all calibrators assessed with cross-fitting so none is scored on its own training data. Selection by lowest mean CV Brier. Test was not used for any selection.
- Result: **none wins** (0.01721 vs 0.01721 / 0.01733 / 0.01723) and also has the best CV ROC-AUC (0.97422). Isotonic collapses raw probabilities onto few unique values and costs ranking (CV ROC 0.96885).
- If pushed: "the brief expected calibration to improve Brier; our measured answer is that a class-weighted forest on ~14k samples is already near-calibrated (test ECE 0.0099), so a calibrator adds variance without removing bias. We report that rather than forcing a calibrator to pass."

## 4. "Defend the cost model and the threshold."

**Artifact:** §9 (expected-cost curve + sensitivity table).

- Written model: FN = a lost real pulsar (follow-up discovery cost), FP = wasted telescope time. 10:1 is a modelling choice, stated as such.
- Threshold chosen by minimising `10·FN + 1·FP` on train+val over a 91-point grid → **T = 0.54**, where the selection set has FN = 0 at FP = 42 (cost 42). The curve is plotted with the chosen T marked.
- Sensitivity 5:1 / 10:1 / 20:1 all select T = 0.54 (because FN = 0 there, cost is FP-driven) → the operating point is robust to the cost ratio inside that bracket.
- At T = 0.54 on test: precision 0.9183, recall 0.8567, confusion TN 3227 / FP 25 / FN 47 / TP 281.

## 5. "Did the test set get touched once?"

**Artifact:** §10; `results/test_predictions.parquet`, `results/final_metrics.json`, model hashes.

- Fit/selection (hyperparameters, calibration method, threshold) happen on train+val only. The test set is evaluated in exactly one cell, which writes predictions + metrics + persists the RF and records SHA-256 hashes. Every later analysis reads the saved parquet (demonstrated in §11).

## 6. "Success criteria?"

- ROC-AUC ≥ 0.97: **met** — 0.9777, CI 0.9671–0.9872 (entire CI clears 0.967).
- Recall ≥ 0.90 at precision ≥ 0.90: **not met** at the cost-optimal point (0.918 / 0.857; recall CI tops at 0.8933). Meeting it would require a cheaper-FP cost ratio (more false alarms) or more informative features — stated in the limitations.
- Brier improved after calibration: **not met** — ΔBrier = 0 because CV selected no calibration. Honest finding, documented reason.

## 7. "What do you get wrong, and what would fix it?"

**Artifact:** `pulsar_selection_v4_errors.ipynb` (supplementary) + paper §5.

- Misses concentrate on feature-atypical pulsars sitting near the population mean (found pulsars sit ~+2.9σ on profile kurtosis); their median score is ~0.1 — confidently wrong, unrecoverable by threshold moves.
- The fix is modality, not thresholds: raw diagnostic-plot inputs (MICNN/PICS line). HTRU2 ships no pixels — stated as a boundary.

## 8. "Anything built without libraries?"

**Artifact:** `from_scratch/` + `pulsar_selection_groundup.ipynb` (supplementary, tag v3.0).

- LogReg, tree, forest, MLP, 1D-CNN in ~350 lines of NumPy, each twin-verified: weight cosine 0.999997 (logreg), 99.16% decision agreement (tree), gradient checks 2.5e-08 (MLP) and 6.6e-10 (CNN), metrics within ~0.002 of sklearn/torch.

## 9. Citation & data policy (rapid-fire)

- Dataset: Lyon, R. (2015), *HTRU2*, UCI ML Repository, DOI 10.24432/C5DK6R (CC-BY 4.0). Paper: Lyon et al., MNRAS 459(1), 2016.
- Raw data: HTRU collaboration, Parkes Observatory (CSIRO/ATNF); survey funded by UK EPSRC EP/I028099/1. Release version + download date recorded in README.
- Features: four folded-profile statistics + four DM–SNR statistics via PulsarFeatureLab; no sky position.

## 10. "Didn't you look at the test set before the final run?"

**Artifact:** `paper.md` §5 ("Protocol evolution").

- Yes. An earlier dossier run computed test metrics (T = 0.10, test precision 0.867 / recall 0.896). Those numbers are kept for traceability.
- The final protocol makes every decision (hyperparameters, calibration method, threshold) on train+val, then evaluates the test set once. The final model is not chosen from the earlier test numbers.

## 11. "Why are there two thresholds, 0.5 and 0.54?"

**Artifacts:** §9 of the main notebook; `pulsar_selection_v4_errors.ipynb` §1.

- T = 0.54 is the cost-optimal threshold and is used for every headline number (precision, recall, confusion matrix).
- The v4 error analysis uses 0.5 on purpose, as the common reference point for comparing RF and CNN decisions (v2/v4 protocol). Its counts (56 / 21 / 24) are not comparable to the T = 0.54 confusion matrix.

## 12. "Why are the cost-ratio sensitivity results identical?"

- At T = 0.54 the selection set has zero false negatives, so 5:1, 10:1 and 20:1 all select the same threshold. The table shows robustness across that bracket, not a wide sweep. Wider sweeps (for example 1:1, 2:1, 50:1) are the natural extension if asked: a lower miss cost should raise T, and a higher one should lower it.

## 13. "The textbook rule says 0.09. Why did you pick 0.54?"

**Artifacts:** §8–9 cost curve (`pulsar_selection.ipynb`); `METRIC_JUSTIFICATION.md` §4.2; `paper.md` §6.

- Under the natural 9.16% survey prior ($\pi = 0.0916$), the textbook Bayes cut-off for a 10:1 cost ratio is $t_{\text{nat}}^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}} = \frac{1}{11} \approx 0.0909$. In fact, in our earlier Gate 3/4 run (`commit 14e98dc`), when Platt scaling was fitted on the unweighted validation fold `X_va`, the empirical minimum landed at **$T = 0.10$**.
- In the final protocol (`commit 813ba86`), repeated CV Brier selected `"none"` — the raw Random Forest trained with `class_weight="balanced_subsample"`. Two effects move the optimum from $0.091$ to $0.54$:
  1. **Balanced-subsample prior shift ($0.091 \to 0.498$):** `balanced_subsample` reweights the 9.16% positive class to 50/50 inside every tree (a weight factor of $w_+ = \frac{1-\pi}{\pi} = \frac{16,259}{1,639} \approx 9.92$). By Bayes' rule, $t_{\text{nat}}^* = 1/11$ maps into balanced-prior score space at $t_{\text{bal}}^* = \frac{9.92 \times (1/11)}{9.92 \times (1/11) + (10/11)} = \frac{9.92}{19.92} \approx 0.498$. Because the 10:1 cost ratio almost cancels the 9.92:1 imbalance weight, the theoretical cut-off in balanced-forest score space is $\approx 0.50$.
  2. **Selection-set refit evaluation ($0.498 \to 0.54$):** In §8–9, `base_final` (depth 20, `min_samples_leaf=2`) is refit on `X_tv` (`train+val`, $n = 14,318$) and the cost curve is evaluated on `p_tv`. On `X_tv`, every true pulsar scores above $0.54$ ($\text{FN} = 0, \text{FP} = 42$ at $T = 0.54$), so raising $T$ from $0.50$ to $0.54$ pays zero FN penalty while trimming false alarms.

## 14. "When was the metric justification written, and why is this a 2-member team?"

**Artifacts:** `METRIC_JUSTIFICATION.md` §1; git commit history (`9014928` on `2026-08-11`, `14e98dc` on `2026-09-25`).

- **Gate 1 & Gate 2 provenance:** The written metric justification (rejecting the 90.84% accuracy trap in favour of ROC-AUC, PR-AUC, recall/precision at a cost threshold, and Brier score), the stratified 60/20/20 split *before* any weighting, and the 10:1 cost model were committed in `pulsar_selection.ipynb` (§1–3, §6, §8) on **2026-09-25** (`commit 14e98dc`), following the initial repository setup on **2026-08-11** (`commit 9014928`). They are also compiled in `METRIC_JUSTIFICATION.md` and `DATA_CARD.md`.
- **Team structure:** The project was executed by a 2-member team (**Aditya Shaji** and **Atharav Gavankar**) under the supervision of **Prof. Dr. Rabinder Henry**, completing the full 12-week classical mainline plus four supplementary tracks (1D-CNN replication, temperature scaling + paired significance, error analysis + ablation, and ground-up NumPy reimplementations).
