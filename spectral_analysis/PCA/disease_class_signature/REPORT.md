# Verdict: not by the four disease classes, once variety is held fixed

Same 998 SNV leaf-mean spectra. Classes: healthy / RB only / LR only / both.

Pooled class colouring mostly repeats variety (Cab Franc is 96% RB only). Fit PCA inside one variety, and shuffle labels within vineyard rows.

| grouping | R² (SNV) |
|---|---|
| variety | 0.510 |
| class pooled over varieties | 0.051 |
| class within variety | 0.011 |
| row within variety (for scale) | 0.083 |

- Baco: no disease separation. Every disease R² is at or below chance. Rows separate Baco far more (R² 0.25).
- Cab Franc: almost every leaf is RB only (9 healthy). No class structure. Large per-band |d| here is that tiny healthy group, not a real 4-class split.
- Chardonnay: a small shape signal for red blotch (R² 0.069, chance 0.006, p = 0.001). Peak |d| = 0.81 at 703 nm (red edge). Leafroll shows nothing. Row R² is 0.063.
- Vidal: a similar small healthy-vs-disease effect (R² 0.023, p = 0.003), much smaller than row (R² 0.22).

Within-variety PCA (`figures/pca_within_variety.png`) does not form four class clouds. Mean spectra discard patchy symptoms; a pixel-level look could still find signal this analysis cannot.

Numbers: `results/tables.md`.
