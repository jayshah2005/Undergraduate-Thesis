# Do the leaf spectra separate by variety or by disease?

Each leaf is represented by one mean spectrum (all 204 bands, 998 leaves). Every analysis was run on the raw spectra and again after SNV, which removes overall brightness and keeps the spectral shape.

The numbers are in `results/tables.md` and the figures in `figures/`. Rerun with `../../.venv/bin/python -u explore_spectra.py` (about 10 s).

## Verdict: the spectra cluster by variety; disease barely moves them

| share of total spectral variance explained by | raw | SNV |
|---|---|---|
| variety | **0.559** | **0.510** |
| disease class within variety | 0.002 | 0.011 |
| row within variety (row = the models' CV fold) | 0.083 | 0.083 |

- **Brightness:** raw PC1 holds 96.6% of the variance and is overall brightness (r = +1.00 with mean reflectance). Brightness itself differs by variety (η² = 0.56): Baco is brightest and Vidal darkest (fig. 01, 04).
- **Shape alone:** after SNV, variety still dominates PC1–3 (η² 0.59 / 0.51 / 0.38). Disease class within variety reaches at most 0.02 on any of the first 10 PCs (fig. 06).
- **Predicting variety:** a linear model predicts the variety from the mean spectrum with 93% balanced accuracy (leave-one-row-out; chance 25%).
- **Effect sizes:** per-band effect sizes between varieties reach |d| = 2–4.8, against 0.16–0.38 for disease within a variety (fig. 03 puts both on the same y-axis).
- **Pooled disease colouring:** in the all-leaf PCA it mostly repeats variety. Cab Franc is 96% "RB only" and Vidal 63%.
- **Rows:** the row explains about 40× (raw) and 8× (SNV) more variance than disease does. Disease prevalence also differs by row: Baco R1 has 3 healthy leaves out of 50, R4 has 21.

## Inside Baco and Chardonnay

The disease p-values below come from shuffling labels within rows, so a row effect cannot pass for disease. "Chance R²" is the average R² the shuffled labels give. AUC is a leave-one-row-out logistic regression, computed within rows, as for the models.

| | R² (chance) raw | p raw | R² (chance) SNV | p SNV | linear AUC raw / SNV |
|---|---|---|---|---|---|
| **Baco** healthy vs any disease | 0.001 (0.004) | 0.66 | 0.012 (0.008) | 0.22 | 0.47 / 0.52 |
| Baco red blotch | 0.002 (0.004) | 0.51 | 0.012 (0.005) | 0.08 | 0.52 / 0.48 |
| Baco leafroll | 0.001 (0.004) | 0.71 | 0.002 (0.005) | 0.75 | 0.51 / 0.48 |
| Baco row (for scale) | 0.086 (0.016) | 0.001 | 0.246 (0.016) | 0.001 | |
| **Chardonnay** healthy vs any disease | 0.003 (0.004) | 0.44 | 0.031 (0.005) | **0.003** | 0.55 / 0.59 |
| Chardonnay red blotch | 0.001 (0.004) | 0.60 | 0.069 (0.007) | **0.001** | 0.58 / 0.57 |
| Chardonnay leafroll | 0.006 (0.004) | 0.26 | 0.009 (0.004) | 0.12 | 0.53 / 0.46 |
| Chardonnay row (for scale) | 0.056 (0.016) | 0.007 | 0.063 (0.016) | 0.001 | |

- **Baco:** healthy and diseased leaves do not separate. Every disease R² is at or below chance, and the linear AUC is 0.47–0.52. Rows separate Baco leaves far more than disease does: R² 0.25 after SNV. The PCA fitted inside Baco shows no class structure (fig. 05).
- **Chardonnay:** there is a small, real **shape** signal for red blotch. It is invisible in the raw spectra, where brightness swamps it.
  - After SNV, red blotch explains 6.9% of Chardonnay's spectral variance (chance 0.7%, p = 0.001). The effect sits at the red edge (peak |d| = 0.38 at 703 nm) and in the green, 520–620 nm (fig. 07).
  - That is still a weak separation: linear AUC about 0.57–0.59, comparable to Chardonnay's own row effect (R² 0.063).
  - Leafroll shows nothing.

## What this means for the models

This supports the concern: cultivar differences dominate the leaf-level representation. The disease signal in the mean spectra is tiny, and the only part that survives (red blotch at the red edge in Chardonnay) matches the models' results. Their within-variety AUC was 0.51–0.58, and red blotch was the only label with any signal. The high pooled scores come from recognising the variety.

## Caveats

- **"Variety" includes whatever was acquired with it,** such as imaging day, illumination and leaf handling, so it cannot be put down to cultivar biology alone. The same goes for "row".
- **Mean spectra discard spatial information.** Patchy symptoms, such as red blotch's interveinal reddening, are diluted in a whole-leaf mean. A pixel-level look could still find signal that this analysis cannot.
- **Specular glints:** on average 20.6% of a leaf's pixels have a band above 1.0 (up to 96%). They are included in the means, as requested, and may drive part of the brightness differences.

## What to try next

1. **Evaluate and normalise within variety.** Use SNV, or per-variety centring, and report within-variety metrics only. Pooled metrics mostly measure variety recognition.
2. **Find out what a row is** (imaging session? field position?). It explains more spectral variance than disease, and it is the CV fold. If rows are sessions, normalise per session.
3. **Look below the leaf mean:** within-leaf pixel distributions (percentiles, the share of pixels with a shifted red edge), with glints (> 1.0) masked first.
