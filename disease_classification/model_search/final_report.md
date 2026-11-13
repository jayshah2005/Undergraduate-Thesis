# Disease model search: results

Can any leaf-level model detect leafroll or red blotch on held-out field rows? Over 500 candidate models (see README.md) were trained on the repo's row folds and chosen on validation rows only. Test balanced accuracy (0.5 = chance; a leaf is positive at P ≥ 0.5). The two diseases are scored separately.

- **Within variety**: one score per variety, then their mean. **Pick**: for each test row, the candidate with the best validation score on the other rows' leaves scores that row.
- **Pooled**: Baco + Chardonnay + Vidal leaves scored together, each model's output adjusted by its variety's infection rate in the training rows; *variety only* predicts from that rate alone.
- Every pool was scored on the same test rows, several times over the search, so these numbers are slightly optimistic. Secondary scores and diagnostics: `metrics.md`.

## Leafroll

Within variety:

| model | Baco | Chardonnay | Vidal | mean |
|---|---|---|---|---|
| logistic regression on the SNV leaf mean (repo baseline) | 0.524 | 0.537 | 0.544 | 0.535 |
| pick, untuned families (359) | 0.577 | 0.492 | 0.473 | 0.514 |
| pick, tuned (114) | 0.518 | 0.512 | 0.513 | 0.514 |
| pick, literature (42) | 0.453 | 0.479 | 0.620 | 0.517 |
| pick, every candidate (515) | 0.547 | 0.536 | 0.440 | 0.508 |
| TabPFN on the SNV leaf mean (fixed in advance) | 0.479 | 0.520 | 0.524 | 0.508 |

Baco + Chardonnay + Vidal pooled:

| model | bal. acc. |
|---|---|
| variety only | 0.679 |
| pick, untuned families (359) | 0.657 |
| pick, tuned (114) | 0.661 |
| pick, literature (42) | 0.654 |
| pick, every candidate (515) | 0.644 |

## Red blotch

Within variety:

| model | Baco | Chardonnay | Vidal | mean |
|---|---|---|---|---|
| logistic regression on the SNV leaf mean (repo baseline) | 0.506 | 0.605 | 0.552 | 0.554 |
| pick, untuned families (359) | 0.529 | 0.567 | 0.521 | 0.539 |
| pick, tuned (114) | 0.538 | 0.528 | 0.536 | 0.534 |
| pick, literature (42) | 0.486 | 0.612 | 0.615 | 0.571 |
| pick, every candidate (515) | 0.482 | 0.595 | 0.552 | 0.543 |
| TabPFN on the SNV leaf mean (fixed in advance) | 0.494 | 0.568 | 0.543 | 0.535 |

Baco + Chardonnay + Vidal pooled:

| model | bal. acc. |
|---|---|
| variety only | 0.615 |
| pick, untuned families (359) | 0.637 |
| pick, tuned (114) | 0.620 |
| pick, literature (42) | 0.612 |
| pick, every candidate (515) | 0.637 |
