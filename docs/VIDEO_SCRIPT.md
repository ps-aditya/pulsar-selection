# 3-minute video — shot-by-shot script (v0.2, corrected)

**Format:** screen recording or slides. **Total:** 180 s, ~330 words (about 115 wpm, leaving room for pauses).
**Recording plan:** record over `docs/poster.html` and the figures in `docs/assets/`, or the executed notebook with this script as the teleprompter. Do 2–3 dry runs and time each scene.

Every scene answers one rubric item from the brief: data, split and leakage, baselines, calibration, cost, learning curve, interpretation, and honest misses.

| # | Time | On screen | Narration (word-for-word) |
|---|---|---|---|
| 1 | 0:00–0:15 | Title: *Finding Pulsars in a Haystack of Noise* + "91% accuracy — found 0 pulsars" | "A model that says 'not a pulsar' for every candidate scores ninety-one percent accuracy and finds nothing. So on this project accuracy is the wrong score. The question is what the model rejects, and how sure it is." |
| 2 | 0:15–0:28 | `docs/assets/balance.png` | "The data is HTRU2: seventeen thousand eight hundred ninety-eight radio candidates. Nine point one six percent are real pulsars. Each candidate is eight numbers: four describe its pulse profile, four describe its dispersion curve." |
| 3 | 0:28–0:50 | Split bar (train / val / test) + leakage-audit output (§6b) | "We split first, stratified, before any reweighting, so near-copies cannot land on both sides. The audit prints zero duplicate rows and disjoint index sets. The test set is evaluated once, at the end." |
| 4 | 0:50–1:20 | `docs/assets/reliability.png` (right panel: CV Brier bars) | "Calibration: we tested four ways to make probabilities honest. None beat the raw forest on cross-validated Brier, so we kept it. Test calibration error is zero point zero one. The brief's target, a better Brier after calibration, is not met, and we report that." |
| 5 | 1:20–1:50 | `docs/assets/cost_curve.png` | "The threshold comes from a written cost model: a missed pulsar costs ten, a wasted follow-up costs one. Under the natural nine percent prior, the textbook rule gives zero point zero nine. Because our forest balances classes by nine point nine times inside each tree, that point shifts to zero point five zero, and the empirical cost minimum on train plus validation lands at zero point five four — zero missed pulsars at forty-two false alarms." |
| 6 | 1:50–2:10 | `docs/assets/learning_curve.png` | "How many labels are needed? Cross-validated PR-AUC rises to about zero point nine two by around five and a half thousand labels, then flattens. That is within the range we tested, which stopped at two-thirds of the training set." |
| 7 | 2:10–2:30 | `docs/assets/importance.png` | "What does the model listen to? Profile kurtosis, how sharply the pulse peaks, by a factor of seventeen over anything else. A real pulsar is a sharp repeating blink. Interference folds flat. The dispersion statistics refine the decision but do not carry it." |
| 8 | 2:30–2:50 | Results card: ROC 0.9777 ✓ · recall-at-precision ✗ · Brier ✗ | "Final test, one pass, with bootstrap intervals. ROC-AUC zero point nine seven seven, which meets the target. Recall is zero point eight six at precision zero point nine two, which misses the target. And the Brier score did not improve after calibration." |
| 9 | 2:50–3:00 | Close card: *The score was easy. The marks are in what it can't do.* | "The score was easy. The real work is calibration, the cost model, and the misses we cannot fix with a threshold." |

## Recording checklist

- [ ] Dry-run the script until each scene lands within ±3 s of its timing.
- [ ] Numbers to get exactly right: **17,898 · 9.16% · 10,738 · 0.54 · 0.09 · 0.9777 · 0.857 · 0.918 · 0.0099 · 47 · 0.633 · 17×**.
- [ ] Say "accuracy" only to call it worthless (scene 1).
- [ ] Say the threshold $T = 0.54$ reconciles with $1/11 \approx 0.091$ via the $9.92\times$ `balanced_subsample` prior weight ($0.498$) and the `train+val` zero-FN knee ($0.54$) (scene 5).
- [ ] Say that the Brier target and the recall-at-precision target are not met (scenes 4 and 8).
- [ ] Audio: quiet room, phone mic about 20 cm away. Re-take a single scene after a stumble, not the whole take.
