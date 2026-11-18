# Disease classification: report

**Question.** Can a whole-leaf CNN detect leafroll or red blotch better than logistic regression on the leaf's mean
spectrum? Does it help to train one model per variety or to pool varieties? Do all 204 bands beat 3 RGB bands?

**Set-up.** Fold k tests on row Rk (of every variety in a pool, since each variety has its own rows); every model
trains on the same 3 other rows. Every model uses SNV, as in variety classification, and every CNN sees a 212 × 212
canvas (224 × 224 for ResNet-18). CNNs pool over leaf pixels by a plain mean or by attention and average 3 seeds. Score: balanced accuracy at
P ≥ 0.5 (0.5 = chance); intervals are in each run's `final_report.md`, other metrics (sensitivity, specificity,
precision, F1) in its `metrics.md`.

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

**Backbones, within a variety.** Mean over Baco, Chardonnay and Vidal of each variety's score (per-variety rows and
intervals: `within_variety_disease_cnn/results/<variety>[_rgb]/final_report.md`). The no-max-pool small ResNet swaps
its 2 max pools for stride-2 convolutions (Springenberg et al. 2015, "Striving for Simplicity: The All Convolutional
Net", ICLR workshop; as ResNet itself downsamples). ResNet-18 is He et al.'s full 18-layer network from scratch.

| Model | Leafroll, 204 bands | Leafroll, RGB | Red blotch, 204 bands | Red blotch, RGB |
|---|---|---|---|---|
| Logistic regression | 0.535 | 0.474 | 0.554 | 0.567 |
| Small CNN, mean / attention | 0.508 / 0.522 | 0.432 / 0.472 | 0.519 / 0.506 | 0.536 / 0.536 |
| Small ResNet, mean / attention | 0.518 / 0.518 | 0.497 / 0.464 | 0.526 / 0.516 | 0.535 / 0.556 |
| Small ResNet, no max pool, mean / attention | 0.504 / 0.543 | 0.475 / 0.521 | 0.538 / 0.531 | 0.550 / 0.529 |
| ResNet-18, mean / attention | 0.471 / 0.493 | 0.484 / 0.484 | 0.508 / 0.490 | 0.485 / 0.526 |

**Backbones, pooled, 204 bands** (RGB and per-variety scores: `cnn/results/<pool>[_rgb]/final_report.md`).

| Model | Leafroll (748 leaves) | Red blotch, all 998 | Red blotch, Baco + Chardonnay + Vidal |
|---|---|---|---|
| Logistic regression | 0.577 | 0.637 | 0.597 |
| Small CNN, mean / attention | 0.604 / 0.620 | 0.682 / 0.686 | 0.608 / 0.605 |
| Small ResNet, mean / attention | 0.591 / 0.622 | 0.675 / 0.686 | 0.624 / 0.604 |
| Small ResNet, no max pool, mean / attention | 0.598 / 0.606 | 0.673 / 0.681 | 0.616 / 0.610 |
| ResNet-18, mean / attention | 0.601 / 0.577 | 0.670 / 0.669 | 0.590 / 0.577 |

**What it shows**
- Within a variety, disease is barely detectable: every score is 0.37–0.62. Chardonnay is the clearest case above chance
  (e.g. logistic regression on red blotch 0.605 [0.547, 0.662]); Baco and Vidal stay within about 0.05 of chance or
  fall below it.
- No CNN beats logistic regression reliably within a variety. Chardonnay leafroll (mean pool 0.591, Δ +0.055
  [−0.014, +0.119]) is the closest; Vidal leafroll, with 25 positive leaves, falls below chance.
- The pooled CNN scores higher overall (red blotch 0.686, leafroll 0.620), but its per-variety scores sit near chance,
  so the gain comes from varieties differing in disease rate (Cab Franc is 96% red blotch, Vidal 10% leafroll). Its
  P ≥ 0.5 cut-off follows the pooled class balance, which also pulls its per-variety scores down. Without Cab Franc
  the red-blotch gain is gone (0.608 vs 0.597).
- No backbone changes the picture. Removing max pooling moves the small ResNet by −0.03 to +0.06, about rerun noise;
  ResNet-18 (11.8 M weights for about 250 leaves) is the weakest CNN on 204 bands. Of the 48 new comparisons with
  logistic regression, 5 intervals sit above 0: four are Baco RGB leafroll, where logistic regression is below chance
  (0.418), and one is Baco leafroll with no max pool and attention (+0.081 [+0.009, +0.150]). That one calls whole rows
  positive (R1, R5) or negative (R3, R4, the row with 20% infected); each row alone scores 0.50–0.64. Pooled, both
  new backbones land within 0.05 of the small CNN (at most 0.02 above it), and their leafroll and red-blotch scores
  for Baco, Chardonnay and Vidal stay at 0.42–0.59.
- RGB does about as well as the full spectrum: the extra bands add nothing we can detect at this sample size.
- The 3-seed ensemble score is fragile to reruns: Chardonnay red blotch (CNN mean) was 0.613 on the earlier 136 px
  canvas and is 0.523 now, although the same seeds scored almost the same alone (0.52 / 0.60 / 0.54 then,
  0.52 / 0.56 / 0.54 now).

**Beyond the CNN.** A wider search (over 500 other models: other features, tuning, TabPFN, DINOv2, leaf-trait
inversion) found nothing that reaches 0.70 or beats logistic regression beyond noise within a variety, so it was removed
from the repo (code at commit `fb310b9`). One finding from it: published leaf-level accuracies of 80–87% (Sawyer et al. 2023) shuffle leaves
across rows; with that split this data also scores only 0.50–0.63.

Figures: `*/figures/<run>/balanced_accuracy.png`; architecture diagrams in `within_variety_disease_cnn/figures/` and
`cnn/figures/`. How to rerun: each folder's README.
