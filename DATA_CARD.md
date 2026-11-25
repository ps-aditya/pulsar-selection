# Data card — HTRU2 (pulsar candidate selection)

Project 7, Section A: Astrophysics. This card follows the Projects 5–7 specification: provenance, release version, licence, acknowledgement text, class balance, features, splits, and known biases.

## 1. Provenance

| Item | Value |
|---|---|
| Dataset | HTRU2 (High Time Resolution Universe survey, South) |
| Survey | HTRU-South, Parkes Observatory, operated by CSIRO (Australia Telescope National Facility) |
| Raw-data acquirer | HTRU collaboration |
| Funding | UK EPSRC grant EP/I028099/1 |
| Donor | R. J. Lyon, University of Manchester |
| Public repository | UCI Machine Learning Repository, dataset id 372 |
| Release | Donated 13 February 2017 (UCI release) |
| Mirror | figshare, DOI `10.6084/m9.figshare.3080389.v1` |
| Dataset DOI | `10.24432/C5DK6R` |
| Local copy | `data/HTRU_2.csv`, committed 2026-09-25 (download date of the cached copy) |

The notebook re-downloads the file automatically if `data/HTRU_2.csv` is missing.

## 2. Licence and acknowledgement

- **Licence:** CC-BY 4.0. Attribution is required when the data are reused.
- **Required acknowledgement:** raw data acquired by the HTRU collaboration with the Parkes radio telescope, part of the Australia Telescope National Facility, operated by CSIRO; survey funded by UK EPSRC grant EP/I028099/1.
- **Required citations:**
  - Lyon, R. (2015). *HTRU2* [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5DK6R
  - Lyon, R. J. et al. (2016). "Fifty Years of Pulsar Candidate Selection: From Simple Filters to Modern Machine Learning." *MNRAS* 459(1).

The code in this repository is MIT licensed (see `LICENSE`). The data are not owned by the repository authors and keep their CC-BY 4.0 terms.

## 3. Contents and class balance

- **Rows:** 17,898 candidates. No missing values.
- **Positive class (pulsar):** 1,639 (9.16%).
- **Negative class (RFI or noise):** 16,259 (90.84%).
- **Imbalance ratio:** about 9.9 : 1.
- **Exact duplicate rows:** 0 (checked in the leakage audit, notebook §6b).

## 4. Features

Eight continuous features, all computed from each candidate's folded integrated pulse profile and its dispersion-measure (DM) curve, using PulsarFeatureLab.

| Group | Feature | Meaning |
|---|---|---|
| Profile | `profile_mean` | mean of the folded integrated profile |
| Profile | `profile_stdev` | standard deviation of the folded profile |
| Profile | `profile_kurtosis` | excess kurtosis: how sharply the pulse peaks |
| Profile | `profile_skewness` | skewness: asymmetry of the profile |
| DM–SNR | `dm_mean` | mean of the DM–SNR curve |
| DM–SNR | `dm_stdev` | standard deviation of the DM–SNR curve |
| DM–SNR | `dm_kurtosis` | excess kurtosis of the DM–SNR curve |
| DM–SNR | `dm_skewness` | skewness of the DM–SNR curve |

No sky position, flux, period, or beam information is included.

## 5. Splits used in this project

Stratified 60 / 20 / 20, `SEED = 42`, created before any class weighting or resampling.

| Split | n | Pulsars | Non-pulsars | Pulsar rate |
|---|---|---|---|---|
| Train | 10,738 | 983 | 9,755 | 0.0915 |
| Validation | 3,580 | 328 | 3,252 | 0.0916 |
| Test | 3,580 | 328 | 3,252 | 0.0916 |

The test set keeps the native class balance and is evaluated once (`paper.md` §3).

## 6. Known biases and limitations

These are the limits a reader must know before using the data or the results.

1. **Single survey and telescope.** All candidates come from HTRU-South at Parkes. A model trained here is not calibrated for other surveys, telescopes, observing setups, or sky regions.
2. **Labels are expert judgements.** The "pulsar" and "non-pulsar" labels come from human vetting of candidates. Vetters can disagree and make mistakes, so some labels are uncertain. The exact vetting protocol is not recorded in this repository; see Lyon et al. (2016).
3. **Class prior reflects the candidate pipeline.** The 9.16% positive rate describes this candidate set, not the true base rate of pulsars in the sky. Any probability or cost-based threshold depends on this prior.
4. **Summary features lose information.** Eight summary statistics discard most of the profile and DM-curve shape. Some real pulsars look like interference in these statistics; the error analysis (`paper.md` Appendix A) finds such cases.
5. **Survey-specific interference.** The non-pulsar class reflects the RFI environment at HTRU-South. A model may learn that environment rather than the physics of pulsars.
6. **No independent validation set from another survey.** Performance is measured on a random held-out split from the same survey, so it estimates in-survey performance only.
7. **Duplicates checked only exactly.** The leakage audit checks exact duplicate rows. Near-duplicates are not checked.
8. **Dataset cannot discover pulsars.** It contains labelled candidates. A model trained on it reproduces expert triage; it finds no new pulsars.

## 7. Reproducibility

- Local copy: `data/HTRU_2.csv`.
- Re-download: automatic in `pulsar_selection.ipynb` §1 if the file is missing.
- Environment: pinned in `requirements.txt`.
