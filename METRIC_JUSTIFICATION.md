# Metric justification (Gate 2 note)

**Status:** written retrospectively, for this repository. The original Gate 2 sign-off date is not recorded here: _[fill in the date and instructor sign-off if a Gate 2 note was written earlier]_. Do not present this as a note written before modelling unless that is true.

## Why accuracy is not used

About 9.16% of HTRU2 candidates are pulsars. A model that always predicts "not a pulsar" scores 90.8% accuracy and finds no pulsars (recall 0%). Accuracy therefore cannot distinguish a useful classifier from a useless one, so it is not reported as a headline metric anywhere in this project.

## Metrics used, and why

| Metric | Why it is used | Where reported |
|---|---|---|
| ROC-AUC | Ranking quality, independent of the class prior. Target from the spec: ≥ 0.97. | `paper.md` §4.3 |
| PR-AUC | Sensitive to the rare class; the more honest headline under imbalance. | `paper.md` §4.3 |
| Precision and recall at the chosen threshold | The operating decision the survey makes. Target from the spec: recall ≥ 0.90 at precision ≥ 0.90. | `paper.md` §4.3 |
| Brier score | Quality of the predicted probabilities. Target from the spec: improved after calibration. | `paper.md` §4.1 |
| Expected calibration error (ECE) | A summary of calibration error on the test set. | `paper.md` §4.3 |
| Bootstrap 95% confidence intervals (B = 2000, paired, stratified) | Shows how much the test numbers could move from sampling variation. | `paper.md` §4.3 |

## Cost model (chosen before the threshold)

A missed pulsar (false negative) costs 10 units: lost discovery follow-up. A wasted follow-up (false positive) costs 1 unit. The 10:1 ratio is a modelling choice, not a measured cost. The threshold is the minimiser of expected cost on train+val. Sensitivity was checked at 5:1 and 20:1 (`paper.md` §4.2).

## Known limit

The threshold of 0.54 does not match the textbook value of C_FP / (C_FP + C_FN) = 1/11 ≈ 0.09 that holds for calibrated probabilities. The gap is reported as open in `paper.md` §5 and `VIVA_PREP.md` §13.
