# Results on the interim labels

Every disease result as it stood before the relabel to the FINAL 2026 virus tests (git tag `interim-labels`, commit
`4f0f0aa`), for comparison. Paths mirror the repo: `interim_labels/disease_classification/REPORT.md` is the old
`disease_classification/REPORT.md`, and so on. The files are exact copies, so the paths they mention resolve to
the other copies here. Only reports, metrics and figures are copied; the fold files and training logs are in the tag
(read one with `git show interim-labels:<path>`).

| here | current version |
|---|---|
| [`disease_classification/REPORT.md`](disease_classification/REPORT.md) | [`../disease_classification/REPORT.md`](../disease_classification/REPORT.md) |
| `disease_classification/<model>/results/<run>/{final_report,metrics}.md`, `figures/` | same paths under `../` |
| [`spectral_analysis/PCA/disease_class_signature/`](spectral_analysis/PCA/disease_class_signature/) | [`../spectral_analysis/PCA/disease_class_signature/`](../spectral_analysis/PCA/disease_class_signature/) |

Variety classification never uses the disease labels, so it is the same on both.

## What changed

Only Vidal: 7 leaves (red blotch 0→1 at R1/V36, R2/V21, R2/V45, R3/V22, R3/V31; 1→0 at R3/V8; leafroll 0→1 at
R3/V5). Baco, Chardonnay and Cab Franc are identical, and so are their within-variety results.

| Vidal (250 leaves) | Red blotch | Leafroll | Healthy |
|---|---|---|---|
| interim labels (here) | 169 | 25 | 68 |
| FINAL 2026 (current) | 173 | 26 | 64 |

Line-by-line differences: `git diff interim-labels HEAD -- disease_classification/REPORT.md`. To check the labels
themselves: `../labelling/README.md`.
