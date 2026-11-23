# 3-minute video — shot-by-shot script (prototype v0.1)

**Format:** screen recording. **Total:** 180 s, ~365 words (≈130 wpm — deliberate headroom for pauses).
**Recording plan:** record over `docs/video_prototype.html` (the animatic auto-times every scene), or record the live notebook with this script as your teleprompter. Do 2–3 dry runs against the animatic before recording.

| # | Time | On screen | Narration (word-for-word) |
|---|---|---|---|
| 1 | 0:00–0:15 | Title card: *"Pulsar Candidate Selection — Project 7"* + the line "91% accuracy — free" | "Here is a dataset where doing almost nothing scores ninety-one percent — just predict 'not a pulsar' for every candidate. And a random forest reaches ninety-eight percent discrimination in minutes. So on this project, the score was never the point." |
| 2 | 0:15–0:28 | `assets/balance.png` | "HTRU2: seventeen thousand eight hundred ninety-eight human-verified candidates from the High Time Resolution Universe survey. Nine point one six percent are genuine pulsars." |
| 3 | 0:28–0:40 | `assets/corr8.png` | "Each candidate is eight numbers — four statistics of its folded pulse profile, four of its dispersion curve. The correlation matrix shows two clean physics blocks: profile shape, and dispersion sweep." |
| 4 | 0:40–1:10 | Leakage-audit terminal output (§6b) | "Before any model, we made the numbers trustworthy. A code-printed leakage audit: zero duplicate rows overall and across every split pair; the three index sets pairwise disjoint; the scaler saw exactly the ten thousand seven hundred thirty-eight training rows; the calibration fold is disjoint from both. And the test set is evaluated exactly once — everything else is selected on train plus validation only." |
| 5 | 1:10–1:45 | `assets/reliability.png` (calibration study panel) | "The brief expects calibration to improve the Brier score. We tested it properly: four strategies — no calibration, sigmoid, isotonic, and cross-fitted sigmoid — under repeated cross-validated Brier on train plus validation. The winner: no calibration. The class-weighted forest is already near-calibrated — expected calibration error under one percent — and isotonic would collapse the ranking. We report the tested answer, not the expected one." |
| 6 | 1:45–2:15 | `assets/cost_curve.png` | "The operating threshold comes from a written cost model: a missed pulsar costs ten; a wasted follow-up costs one. The expected-cost curve bottoms at threshold zero point five four — zero missed pulsars at forty-two false alarms on the selection set — and all three cost ratios we tested, five-to-one, ten-to-one, twenty-to-one, pick the same threshold." |
| 7 | 2:15–2:35 | `assets/importance.png` | "So what does the model listen to? Profile kurtosis — by a factor of seventeen over anything else. That's the lighthouse: a real pulsar folds to a sharp, peaked blink; interference folds flat. The dispersion statistics confirm the detection." |
| 8 | 2:35–2:55 | Results card: ROC 0.9777 ✓ · misses ✗✗ | "Final test evaluation, one pass, with bootstrap confidence intervals: ROC-AUC zero point nine seven seven seven. And the honest misses first: the recall-at-precision target is not met at the cost-optimal point, and calibration did not improve the Brier — both documented with measured reasons." |
| 9 | 2:55–3:00 | Close card: *"The score was free. We earned marks on everything the score hides."* | "The score was free. We earned marks on everything the score hides." |

## Recording checklist

- [ ] Dry-run against the animatic (`docs/video_prototype.html`) until each scene lands within ±3 s.
- [ ] Numbers you must not fluff: **17,898 · 9.16% · 10,738 · T=0.54 · 42 · 0.633 · 0.9777 · ECE 0.0099**.
- [ ] Never say "accuracy" except to call it worthless (scene 1).
- [ ] If recording the notebook instead: pre-open the executed notebook at §6b, §8 figure, §9 figure — don't scroll-hunt on camera.
- [ ] Audio: quiet room, phone mic at ~20 cm is fine; re-take any scene with a stumble rather than the whole take.
