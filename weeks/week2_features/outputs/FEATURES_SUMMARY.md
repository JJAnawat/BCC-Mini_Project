# Week 2 — Feature extraction summary

## Files

One CSV per domain per subject: `features/<domain>/s{NN}_<domain>.csv`, 600 rows each
(40 trials × 15 non-overlapping 4s windows). Every file starts with the same 3 **ID columns**,
`subject_id` (1-32), `trial_idx` (0-39), `window_idx` (0-14). They are not features; they join
each row to its labels in `metadata/metadata_window_level.csv` (valence, arousal, dominance,
liking, quadrant). Labels are per trial, so all 15 windows of a trial share them.

| Domain | Feature columns | Column pattern (example) | Unit |
|---|---|---|---|
| `de` | 32 channels × 4 bands = 128 | `{channel}_{band}_de` (`Fp1_alpha_de`) | nats |
| `hjorth` | 32 channels × 3 = 96 | `{channel}_{activity\|mobility\|complexity}` (`F7_activity`) | Activity µV²; others unitless |
| `asymmetry` | 14 pairs × 4 bands × 2 = 112 | `{left}-{right}_{band}_{dasm\|rasm}` (`F3-F4_alpha_dasm`) | DASM nats; RASM ratio |
| `mse` | 32 channels × 10 scales = 320 | `{channel}_scale{1-10}_mse` (`Oz_scale3_mse`) | unitless |
| `microstates` | 4 maps × 3 + 12 transitions = 24 | `map{0-3}_{coverage\|duration_s\|occurrence_per_s}`, `trans_map{i}_to_map{j}` | fraction, s, per s, probability |

Microstate transitions: 4 maps give 16 from→to pairs, but map-to-itself is always 0 here
(repeats are ignored), so only the 12 cross-map transitions are kept.

## Methods

- **DE:** Welch PSD per window (Week 1 idiom, `nperseg=512`), band power by trapezoid
  integration, DE = 0.5·ln(2πe·power).
- **Hjorth:** Activity = variance, Mobility, Complexity; broadband, time domain.
- **DASM/RASM:** DE(left) − DE(right) and DE(left) / DE(right), computed from the saved DE files.
- **MSE:** coarse-graining + antropy `sample_entropy`, m=2, r = 0.15·std of the scale-1 window,
  kept fixed across scales. EntropyHub 2.0's `MSEn` is the reference cross-check (exact match).
- **Microstates:** K=4 maps fit once on pooled GFP peaks (1000 per subject, each subject scaled
  to unit median GFP, `ModKMeans` n_init=100, seed 42). Each whole trial is backfit with
  smoothing (factor 10, ±4 samples ≈ ±31 ms), then labels are split into windows. Map names
  are arbitrary k-means order, not canonical A/B/C/D.

Runtime from scratch: ~15 min (MSE ~12 min, microstates ~3 min). MSE and microstates are
cached per subject; delete `features/mse/`, `features/microstates/` or `models/` to recompute.

## Sanity checks (`sanity_checks/<domain>/`)

- **DE:** 17/32 subjects show a posterior alpha peak above the 1/f trend. A narrowband 22 Hz
  line (non-physiological) appears in 29/32 subjects and inflates beta power by a per-subject
  amount (`subject_checks.csv`, `de_spectrum_pooled.png`).
- **Hjorth:** no NaN, inf or zero values.
- **Asymmetry:** RASM is stable because DE is always positive (0.1-99.9th percentile 0.40-2.85).
- **MSE:** 538 `inf` values at scales 4-10 (≤0.024% of any scale): short coarse-grained windows
  with no template match. Left in place.
- **Microstates:** group GEV 0.38, and the maps are focal (single electrode such as F7, or
  temporal), not canonical. 18/32 subjects are flagged `flag_degenerate` in
  `subject_quality.csv`: one fixed scalp pattern (eye/muscle/bad-electrode artifact) explains
  >90% of their variance, or >50% of their windows sit in a single map.

The user chose to keep the locked preprocessing unchanged despite these findings (no notch
filter, no artifact removal).
