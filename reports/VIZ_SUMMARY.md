# Week 3 — Visualization + cross-domain correlation

Files named below are in `outputs/viz/`. Code: `pipeline/viz/`.

Inputs: Week 2 features, unchanged. MSE `inf` is set to the subject's column median (Week 3 only).
"z-scored" means z-scored per subject. Plots use trial means (1,280), correlations use windows (19,200).

## 1. Features cluster by subject, not by emotion (`embedding_<domain>.png`, `silhouette.csv`)

| Domain | Silhouette by subject, raw | by subject, z-scored | by quadrant, raw / z-scored |
|---|---|---|---|
| de | 0.61 | −0.05 | −0.01 / −0.03 |
| hjorth | 0.27 | −0.07 | −0.01 / −0.03 |
| asymmetry | 0.71 | −0.02 | −0.01 / −0.01 |
| mse | 0.24 | −0.03 | −0.03 / −0.01 |
| microstates | 0.28 | −0.17 | −0.02 / −0.01 |

- On raw data every domain forms one cluster per subject (t-SNE gives 32 islands for DE and asymmetry).
- Per-subject z-scoring removes the subject clusters entirely. So per-subject normalization is required for cross-subject work.
- No domain separates the quadrants, before or after normalization. Any class signal is weak and
  multivariate, not visible in 2-D.
- In the z-scored microstate t-SNE, the tight blobs are degenerate subjects: their features are
  constant within the subject, so they z-score to 0.

## 2. Class effects are small (`boxplots_by_quadrant.png`, `topomap_de.png`, `topomap_dasm.png`)

- For the pre-chosen features, Kruskal-Wallis η² is at most 0.012 (T7 gamma DE, Pz alpha DE); most are ≈ 0.
- The DE topomaps show a whole-scalp shift of about ±0.2 SD, not a localized pattern:
  - high valence is above the subject's mean in theta and alpha, low valence below it;
  - HVHA is highest and LVLA lowest in beta and gamma.
  - Beta carries the 22 Hz line caveat.
- The DASM topomaps have no consistent pattern (|mean z| ≤ 0.15). Frontal alpha asymmetry (F3-F4) does not
  separate the quadrants (η² 0.001).

## 3. Cross-domain redundancy (`correlation_all_features.png`, `correlation_domain_summary.csv/.png`)

Each cell below is mean best-match |Spearman r|: for each feature in the row domain, its strongest
|r| with any feature in the column domain, averaged. The table is asymmetric.

| from \ to | de | hjorth | asymmetry | mse | microstates |
|---|---|---|---|---|---|
| **de** | 0.75 | 0.54 | 0.33 | 0.35 | 0.18 |
| **hjorth** | 0.61 | 0.84 | 0.18 | 0.72 | 0.11 |
| **asymmetry** | 0.37 | 0.17 | 0.92 | 0.13 | 0.06 |
| **mse** | 0.26 | 0.43 | 0.10 | 0.54 | 0.08 |
| **microstates** | 0.12 | 0.12 | 0.07 | 0.07 | 0.66 |

- **DE and Hjorth overlap substantially** (0.54 / 0.61), because Activity is broadband power.
  Hjorth also overlaps strongly with MSE (0.72, sign mostly negative), because Mobility/Complexity
  and entropy both track frequency content. Hjorth sits between the power and entropy domains.
- **Asymmetry is largely *not* redundant with DE** (0.37), even though it is computed from DE.
  Within a subject, left and right DE correlate at ρ ≈ 0.69. Subtracting them cancels the shared
  power, which leaves |r| ≈ √((1−ρ)/2) ≈ 0.39. The observed value is 0.33, which fits.
- **Within-domain redundancy is high:**
  - asymmetry 0.92: DASM and RASM of the same pair carry the same information;
  - Hjorth 0.84;
  - DE 0.75: channels share a global power factor.
- **Microstates are nearly independent** of the other domains (≤ 0.18).

**For Week 5:** if DE, Hjorth and MSE score similarly, that is partly shared information, not
independent confirmation. Asymmetry and microstates are the genuinely distinct views.

Reproduce with `python pipeline/viz/run_all.py` (~2 min).
