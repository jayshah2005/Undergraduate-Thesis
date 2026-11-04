# Disease classification: report

**Question.** Can a whole-leaf CNN detect leafroll or red blotch better than logistic regression on the leaf's mean
spectrum? Does it help to train one model per variety or to pool varieties? Do all 204 bands beat 3 RGB bands?

**Set-up.** Fold k tests on row Rk (of every variety in a pool, since each variety has its own rows); every model
trains on the same 3 other rows. Every model uses SNV, as in variety classification, and every CNN sees a 212 × 212
canvas. CNNs pool over leaf pixels by a plain mean or by
attention and average 3 seeds. Score: balanced accuracy at P ≥ 0.5 (0.5 = chance); intervals are in each `metrics.md`.

**Macro average** (mean of the leafroll and red-blotch balanced accuracies; one number per model). Per variety =
one model per variety; pooled = one model over the pool, scored on all its leaves.

| Model for, input → | Log. regression | CNN mean | CNN attention |
|---|---|---|---|
| Baco, full spectrum | 0.515 | 0.525 | 0.517 |
| Chardonnay, full spectrum | 0.571 | 0.557 | 0.539 |
| Vidal, full spectrum | 0.548 | 0.459 | 0.485 |
| Baco + Chardonnay + Vidal, full spectrum | 0.587 | 0.606 | 0.612 |
| All 998, full spectrum | 0.607 | 0.643 | 0.653 |
| Baco, RGB | 0.481 | 0.447 | 0.494 |
| Chardonnay, RGB | 0.604 | 0.558 | 0.576 |
| Vidal, RGB | 0.477 | 0.447 | 0.442 |
| Baco + Chardonnay + Vidal, RGB | 0.574 | 0.583 | 0.594 |
| All 998, RGB | 0.592 | 0.607 | 0.624 |

**Per variety, full spectrum.** Within = one model per variety (`within_variety_disease_cnn/`). Pooled = the CNN
trained on all 998 leaves (`cnn/results/all`), scored on that variety's leaves.

| Variety, task | Log. regression | CNN mean | CNN attention | Pooled CNN mean | Pooled CNN attention |
|---|---|---|---|---|---|
| Baco, leafroll | 0.524 | 0.546 | 0.529 | 0.519 | 0.525 |
| Baco, red blotch | 0.506 | 0.504 | 0.505 | 0.466 | 0.474 |
| Chardonnay, leafroll | 0.537 | 0.591 | 0.559 | 0.476 | 0.521 |
| Chardonnay, red blotch | 0.605 | 0.523 | 0.519 | 0.531 | 0.538 |
| Vidal, leafroll (25 +) | 0.544 | 0.387 | 0.478 | 0.527 | 0.531 |
| Vidal, red blotch | 0.552 | 0.531 | 0.493 | 0.548 | 0.536 |

**Per variety, RGB.**

| Variety, task | Log. regression | CNN mean | CNN attention | Pooled CNN mean | Pooled CNN attention |
|---|---|---|---|---|---|
| Baco, leafroll | 0.418 | 0.370 | 0.440 | 0.462 | 0.540 |
| Baco, red blotch | 0.543 | 0.525 | 0.548 | 0.497 | 0.503 |
| Chardonnay, leafroll | 0.588 | 0.531 | 0.569 | 0.436 | 0.423 |
| Chardonnay, red blotch | 0.619 | 0.584 | 0.583 | 0.517 | 0.534 |
| Vidal, leafroll (25 +) | 0.416 | 0.396 | 0.407 | 0.493 | 0.502 |
| Vidal, red blotch | 0.538 | 0.499 | 0.477 | 0.471 | 0.503 |

**Pooled, over all its leaves.**

| Pool, task | Log. regression | CNN mean | CNN attention |
|---|---|---|---|
| All 998 / Baco + Chardonnay + Vidal, leafroll (same 748 leaves) | 0.577 | 0.604 | 0.620 |
| All 998, red blotch | 0.637 | 0.682 | 0.686 |
| Baco + Chardonnay + Vidal, red blotch | 0.597 | 0.608 | 0.605 |

**What it shows**
- Macro average: within a variety no CNN beats logistic regression (Vidal CNN mean −0.090 [−0.137, −0.040]).
  Pooled over the 3 varieties, attention pooling is slightly ahead (0.612 vs 0.587, Δ +0.026 [+0.004, +0.049]).
- Within a variety, disease is barely detectable: every score is 0.37–0.62. Chardonnay is the clearest case above chance
  (e.g. logistic regression on red blotch 0.605 [0.547, 0.662]); Baco and Vidal stay within about 0.05 of chance or
  fall below it.
- No CNN beats logistic regression reliably within a variety. Chardonnay leafroll (mean pool 0.591, Δ +0.055
  [−0.014, +0.119]) is the closest; Vidal leafroll, with 25 positive leaves, falls below chance.
- The pooled CNN scores higher overall (red blotch 0.686, leafroll 0.620), but its per-variety scores sit near chance,
  so the gain comes from varieties differing in disease rate (Cab Franc is 96% red blotch, Vidal 10% leafroll). Its
  P ≥ 0.5 cut-off follows the pooled class balance, which also pulls its per-variety scores down. Without Cab Franc
  the red-blotch gain is gone (0.608 vs 0.597).
- RGB does about as well as the full spectrum: the extra bands add nothing we can detect at this sample size.
- The 3-seed ensemble score is fragile to reruns: Chardonnay red blotch (CNN mean) was 0.613 on the earlier 136 px
  canvas and is 0.523 now, although the same seeds scored almost the same alone (0.52 / 0.60 / 0.54 then,
  0.52 / 0.56 / 0.54 now).

Figures: `*/figures/<run>/balanced_accuracy.png`; architecture diagrams in `within_variety_disease_cnn/figures/` and
`cnn/figures/`. How to rerun: each folder's README.
