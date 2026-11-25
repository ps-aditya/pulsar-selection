# Metric justification and Gate 2 specification record

**Project 7, Section A: Astrophysics — Pulsar Candidate Selection (HTRU2)**  
**Team:** Aditya Shaji · Atharav Gavankar (2-member team, Second Year B.Tech CSE — AI & ML)  
**Guide:** Prof. Dr. Rabinder Henry

## 1. Milestone & Gate provenance (verified from git history)

| Milestone Gate | Date | Commit | Verified artifact in repository |
|---|---|---|---|
| **Gate 1:** Dataset loaded, profiled, citation requirements recorded | `2026-08-11` / `2026-09-25` | `9014928` / `14e98dc` | `README.md` + `pulsar_selection.ipynb` §1–2 (provenance, UCI id 372, CC-BY 4.0, Parkes/CSIRO/EPSRC acknowledgement, class balance 1,639 : 16,259) |
| **Gate 2:** Split strategy & written metric justification before modelling | `2026-09-25` | `14e98dc` | `pulsar_selection.ipynb` §2, §3, §6, §8 — written rejection of accuracy (90.84% all-zero trap), stratified 60/20/20 split *before* any weighting, and written 10:1 cost model (`C_FN = 10, C_FP = 1`) committed prior to multi-model extensions |
| **Gate 3 & 4:** Baseline vs tuned Random Forest, calibration & cost study | `2026-09-25` – `2026-11-15` | `14e98dc` / `813ba86` | `pulsar_selection.ipynb` §7–10 — Standardized Logistic Regression baseline alongside 3-fold CV-tuned Random Forest, 4-way repeated-CV calibration study, single final test evaluation |
| **Gate 5:** Error analysis, ablations, operating-point sensitivity | `2026-10-02` – `2026-11-22` | `73c45bc` / `5811298` / `01c6d1e` | `pulsar_selection_v4_errors.ipynb`, `cnn_ablation.py`, 5:1 / 10:1 / 20:1 cost-ratio sensitivity table |

*Note on standalone file extraction:* The metric justification and split strategy were originally written inside `pulsar_selection.ipynb` (§2–3, §6, §8) in commit `14e98dc` (`2026-09-25`) and extracted into this standalone `METRIC_JUSTIFICATION.md` on `2026-11-25` for immediate rubric inspection.

---

## 2. Why accuracy is rejected (the 90.84% majority-class trap)

In HTRU2, only **1,639 of 17,898** candidates are genuine pulsars (**9.157% positive**, **90.843% negative**, an imbalance ratio of **9.92 : 1**).

A trivial classifier that predicts `"not a pulsar"` ($\hat{y} = 0$) for every candidate achieves:
- **Accuracy:** $16,259 / 17,898 = 90.84\%$
- **Recall (Sensitivity):** $0 / 1,639 = 0.00\%$ (zero pulsars discovered)
- **PR-AUC:** $0.0916$ (no better than random guessing)

In radio-survey astronomy, the rare positive class is the entire scientific objective, and the operational question is what the model has learned to reject without discarding true pulsars. Therefore, headline accuracy figures are rejected across the notebook, report, poster, and presentation.

---

## 3. Selected metrics and their operational roles

| Metric | Mathematical / operational role | Project 7 target | Final test result (`SEED = 42`) |
|---|---|---|---|
| **ROC-AUC** | Threshold-independent ranking quality across all trade-offs ($P(\hat{p}_+ > \hat{p}_-)$). Insensitive to class prior. | $\ge 0.97$ | **0.9777** (95% CI: 0.9671–0.9872) — **Met** |
| **PR-AUC (Average Precision)** | Area under the precision–recall curve; sensitive to false-positive contamination of the rare positive class ($9.16\%$). Used as the primary hyperparameter tuning objective (`scoring="average_precision"`). | Primary CV objective | **0.9332** (95% CI: 0.9117–0.9542) |
| **Recall & Precision at cost threshold $T$** | Operational performance at the survey's actual decision boundary ($T = 0.54$ from the 10:1 cost model). | Recall $\ge 0.90$ at Precision $\ge 0.90$ | Precision **0.9183** (0.8889–0.9461), Recall **0.8567** (0.8201–0.8933) — **Not met** at $T=0.54$ |
| **Brier score** | Mean squared probability error $\frac{1}{N}\sum (\hat{p}_i - y_i)^2$, decomposing into calibration reliability, resolution, and uncertainty. | Demonstrably improved after calibration | **0.0166** (0.0135–0.0199); $\Delta\text{Brier} = 0.0000$ (`none` selected by CV) — **Not met** |
| **Expected Calibration Error (ECE)** | Weighted average gap between predicted confidence and empirical accuracy across probability bins. | Diagnostic | **0.0099** ($< 1\%$ calibration error on test) |
| **Paired Stratified Bootstrap 95% CIs** | Quantifies sampling uncertainty ($B = 2000$ resamples, `SEED = 42`) on the single held-out test pass. | Required by rubric | Reported on all test metrics (`paper.md` §5.4) |

---

## 4. Cost model and the reconciliation of $T = 0.54$ vs textbook $t^* \approx 0.09$

### 4.1 Explicit cost statement
Using a default threshold of $0.50$ without argument implicitly assumes that missing a real pulsar ($C_{\text{FN}}$) and following up a false alarm ($C_{\text{FP}}$) have equal cost. In pulsar surveys, missing a true pulsar loses a scientific discovery, whereas a false positive wastes a slot of follow-up telescope inspection. We define:
$$C_{\text{FN}} = 10, \qquad C_{\text{FP}} = 1 \qquad (\text{cost ratio } 10 : 1)$$
and select the threshold $T \in [0.05, 0.95]$ (91-point grid) that minimises the total cost $10 \cdot \text{FN}(T) + 1 \cdot \text{FP}(T)$ on the selection set.

### 4.2 Why the empirical optimum is $T = 0.54$ instead of $t_{\text{nat}}^* = 1/11 \approx 0.091$
Under Bayes decision theory, if probabilities $\hat{p}_{\text{nat}}(y=1 \mid x)$ are calibrated to the **natural survey prior** ($\pi = 0.0916$), the theoretical minimum-cost threshold is:
$$t_{\text{nat}}^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}} = \frac{1}{1 + 10} = \frac{1}{11} \approx 0.0909$$
Indeed, in our earlier Gate 3/4 dossier run (`commit 14e98dc`), when Platt (`sigmoid`) scaling was fit on the held-out validation fold (`X_va`, unweighted natural prior $\pi = 0.0916$), the empirical cost curve on `X_va` selected **$T = 0.10$** (the grid point closest to $0.0909$).

In the final single-test-evaluation protocol (`commit 813ba86`, `pulsar_selection.ipynb` §8–9), the empirical minimum on `train+val` moves to **$T = 0.54$** for two exact, verifiable reasons:

1. **Prior-odds shift from `class_weight="balanced_subsample"` (`0.091 → 0.498`):**  
   Repeated cross-validated Brier selection (§8) chose `"none"` (the raw Random Forest trained with `class_weight="balanced_subsample"`). Inside each bootstrap tree, `balanced_subsample` reweights the classes so positives and negatives carry equal total weight (a 50/50 balanced prior, multiplying positive odds by $w_+ = \frac{1-\pi}{\pi} = \frac{16,259}{1,639} \approx 9.92$). By Bayes' theorem, a natural-prior probability $t_{\text{nat}}^* = 1/11$ maps into class-balanced score space at:
   $$t_{\text{bal}}^* = \frac{w_+ \, t_{\text{nat}}^*}{w_+ \, t_{\text{nat}}^* + (1 - t_{\text{nat}}^*)} = \frac{9.92 \times (1/11)}{9.92 \times (1/11) + (10/11)} = \frac{9.92}{19.92} \approx 0.498$$
   Thus, a 10:1 cost ratio against a 9.92:1 class-imbalance weight places the theoretical Bayes cut-off right at $\approx 0.50$ in the forest's balanced-subsample score space.

2. **Selection-set refit sharpening (`0.498 → 0.54`):**  
   In §8–9, the final forest (`400` trees, `max_depth=20`, `min_samples_leaf=2`) is refit on the combined selection set `X_tv` (`train+val`, $n = 14,318$), and the cost curve $10 \cdot \text{FN}(T) + 1 \cdot \text{FP}(T)$ is evaluated on `p_tv = final_model.predict_proba(X_tv)[:, 1]`. Because every labelled pulsar in `X_tv` receives an ensemble score above $0.54$ ($\text{FN} = 0$ on `X_tv` for all $T \le 0.54$, with $\text{FP} = 42$ at $T = 0.54$), raising the threshold from $0.50$ to $0.54$ incurs zero false-negative penalty on `X_tv` while eliminating false positives. Above $0.54$, false negatives begin appearing with weight $10$, causing total cost to jump sharply. This also explains why cost ratios $5:1$, $10:1$, and $20:1$ all share the exact same minimum at $T = 0.54$ on `X_tv`.
