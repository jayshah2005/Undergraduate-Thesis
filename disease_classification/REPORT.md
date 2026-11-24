# Disease classification: report

**Question.** Can a whole-leaf CNN detect leafroll or red blotch better than logistic regression on the leaf's mean
spectrum? Does it help to train one model per variety or to pool varieties? Do all 204 bands beat 3 RGB bands?

**Set-up.** Fold k tests on row Rk (of every variety in a pool, since each variety has its own rows); every model
trains on the same 3 other rows. Every model uses SNV, as in variety classification, and every CNN sees a 212 × 212
canvas (224 × 224 for ResNet-18). CNNs pool over leaf pixels by a plain mean or by attention and average 3 seeds. Score: balanced accuracy at
P ≥ 0.5 (0.5 = chance); intervals are in each run's `final_report.md`, other metrics (sensitivity, specificity,
precision, F1) in its `metrics.md`. Labels: the FINAL 2026 virus tests; 7 Vidal leaves differ from the interim labels
(the file with those: `../labelling/vine_hsi_interim_labels.h5`).

**Per variety, full spectrum.** Within = one model per variety (`within_variety_disease_cnn/`). Pooled = the CNN
trained on all 998 leaves (`cnn/results/all`), scored on that variety's leaves.

| Variety, task | Log. regression | CNN mean | CNN attention | Pooled CNN mean | Pooled CNN attention |
|---|---|---|---|---|---|
| Baco, leafroll | 0.524 | 0.546 | 0.529 | 0.494 | 0.516 |
| Baco, red blotch | 0.506 | 0.504 | 0.505 | 0.499 | 0.500 |
| Chardonnay, leafroll | 0.537 | 0.591 | 0.559 | 0.457 | 0.454 |
| Chardonnay, red blotch | 0.605 | 0.523 | 0.519 | 0.555 | 0.539 |
| Vidal, leafroll (26 +) | 0.557 | 0.474 | 0.447 | 0.518 | 0.468 |
| Vidal, red blotch | 0.547 | 0.557 | 0.492 | 0.539 | 0.505 |

**Per variety, RGB.**

| Variety, task | Log. regression | CNN mean | CNN attention | Pooled CNN mean | Pooled CNN attention |
|---|---|---|---|---|---|
| Baco, leafroll | 0.418 | 0.370 | 0.440 | 0.454 | 0.522 |
| Baco, red blotch | 0.543 | 0.525 | 0.548 | 0.500 | 0.492 |
| Chardonnay, leafroll | 0.588 | 0.531 | 0.569 | 0.456 | 0.419 |
| Chardonnay, red blotch | 0.619 | 0.584 | 0.583 | 0.525 | 0.539 |
| Vidal, leafroll (26 +) | 0.439 | 0.415 | 0.410 | 0.466 | 0.497 |
| Vidal, red blotch | 0.552 | 0.482 | 0.454 | 0.503 | 0.512 |

**Pooled, over all its leaves.**

| Pool, task | Log. regression | CNN mean | CNN attention |
|---|---|---|---|
| All 998 / Baco + Chardonnay + Vidal, leafroll (same 748 leaves) | 0.572 | 0.586 | 0.597 |
| All 998, red blotch | 0.647 | 0.691 | 0.666 |
| Baco + Chardonnay + Vidal, red blotch | 0.600 | 0.619 | 0.606 |

**Backbones, within a variety.** Mean over Baco, Chardonnay and Vidal of each variety's score (per-variety rows and
intervals: `within_variety_disease_cnn/results/<variety>[_rgb]/final_report.md`). The no-max-pool small ResNet swaps
its 2 max pools for stride-2 convolutions (Springenberg et al. 2015, "Striving for Simplicity: The All Convolutional
Net", ICLR workshop; as ResNet itself downsamples). ResNet-18 is He et al.'s full 18-layer network from scratch.

| Model | Leafroll, 204 bands | Leafroll, RGB | Red blotch, 204 bands | Red blotch, RGB |
|---|---|---|---|---|
| Logistic regression | 0.539 | 0.482 | 0.553 | 0.571 |
| Small CNN, mean / attention | 0.537 / 0.512 | 0.439 / 0.473 | 0.528 / 0.505 | 0.530 / 0.528 |
| Small ResNet, mean / attention | 0.537 / 0.511 | 0.494 / 0.481 | 0.515 / 0.522 | 0.534 / 0.547 |
| Small ResNet, no max pool, mean / attention | 0.509 / 0.541 | 0.501 / 0.506 | 0.537 / 0.536 | 0.542 / 0.535 |
| ResNet-18, mean / attention | 0.486 / 0.497 | 0.496 / 0.482 | 0.525 / 0.493 | 0.509 / 0.540 |

**Backbones, pooled, 204 bands** (RGB and per-variety scores: `cnn/results/<pool>[_rgb]/final_report.md`).

| Model | Leafroll (748 leaves) | Red blotch, all 998 | Red blotch, Baco + Chardonnay + Vidal |
|---|---|---|---|
| Logistic regression | 0.572 | 0.647 | 0.600 |
| Small CNN, mean / attention | 0.586 / 0.597 | 0.691 / 0.666 | 0.619 / 0.606 |
| Small ResNet, mean / attention | 0.585 / 0.612 | 0.682 / 0.686 | 0.595 / 0.594 |
| Small ResNet, no max pool, mean / attention | 0.595 / 0.602 | 0.683 / 0.673 | 0.624 / 0.590 |
| ResNet-18, mean / attention | 0.607 / 0.573 | 0.672 / 0.667 | 0.602 / 0.591 |

**Attention inside the CNN, and mean ∥ max pooling.** ECA (Wang et al. 2020, "ECA-Net: Efficient Channel Attention
for Deep Convolutional Neural Networks", CVPR) reweights channels; CBAM (Woo et al. 2018, "CBAM: Convolutional Block
Attention Module", ECCV) reweights channels, then positions. Both average over leaf pixels only and sit in each
residual block of the small ResNet, before the skip is added, as in the papers, and after each 3 × 3 conv of the small
CNN (our choice: neither paper has a plain 4-conv net). Mean ∥ max pooling joins each channel's mean and max over the
leaf (Howard & Ruder 2018, "Universal Language Model Fine-tuning for Text Classification", ACL). Each is compared with
the same model without it, paired on the same resamples; the rule, set before the runs, is that a change helps only
where that 95% interval excludes 0. Within a variety, mean over Baco, Chardonnay and Vidal:

| Model | Leafroll, 204 bands | Leafroll, RGB | Red blotch, 204 bands | Red blotch, RGB |
|---|---|---|---|---|
| Logistic regression | 0.539 | 0.482 | 0.553 | 0.571 |
| Small CNN, mean / attention | 0.537 / 0.512 | 0.439 / 0.473 | 0.528 / 0.505 | 0.530 / 0.528 |
| Small CNN + ECA, mean / attention | 0.523 / 0.562 | 0.497 / 0.482 | 0.503 / 0.535 | 0.532 / 0.539 |
| Small CNN + CBAM, mean / attention | 0.510 / 0.524 | 0.495 / 0.539 | 0.547 / 0.505 | 0.558 / 0.523 |
| Small CNN, mean ∥ max | 0.515 | 0.481 | 0.539 | 0.516 |
| Small ResNet, mean / attention | 0.537 / 0.511 | 0.494 / 0.481 | 0.515 / 0.522 | 0.534 / 0.547 |
| Small ResNet + ECA, mean / attention | 0.488 / 0.539 | 0.510 / 0.475 | 0.552 / 0.543 | 0.539 / 0.534 |
| Small ResNet + CBAM, mean / attention | 0.525 / 0.510 | 0.535 / 0.501 | 0.527 / 0.521 | 0.521 / 0.535 |
| Small ResNet, mean ∥ max | 0.538 | 0.492 | 0.522 | 0.500 |

Pooled, 204 bands (RGB: `cnn/results/<pool>_rgb/final_report.md`; on Baco + Chardonnay + Vidal these models ran red
blotch only, since its leafroll leaves are the same 748):

| Model | Leafroll (748 leaves) | Red blotch, all 998 | Red blotch, Baco + Chardonnay + Vidal |
|---|---|---|---|
| Logistic regression | 0.572 | 0.647 | 0.600 |
| Small CNN, mean / attention | 0.586 / 0.597 | 0.691 / 0.666 | 0.619 / 0.606 |
| Small CNN + ECA, mean / attention | 0.603 / 0.590 | 0.686 / 0.681 | 0.580 / 0.609 |
| Small CNN + CBAM, mean / attention | 0.610 / 0.624 | 0.702 / 0.701 | 0.623 / 0.589 |
| Small CNN, mean ∥ max | 0.599 | 0.689 | 0.611 |
| Small ResNet, mean / attention | 0.585 / 0.612 | 0.682 / 0.686 | 0.595 / 0.594 |
| Small ResNet + ECA, mean / attention | 0.607 / 0.580 | 0.683 / 0.680 | 0.609 / 0.626 |
| Small ResNet + CBAM, mean / attention | 0.610 / 0.583 | 0.686 / 0.678 | 0.607 / 0.623 |
| Small ResNet, mean ∥ max | 0.595 | 0.675 | 0.619 |

**What it shows**
- Within a variety, disease is barely detectable: every score is 0.37–0.62. Chardonnay is the clearest case above chance
  (e.g. logistic regression on red blotch 0.605 [0.547, 0.662]); Baco and Vidal stay within about 0.06 of chance or
  fall below it.
- No CNN beats logistic regression reliably within a variety. Chardonnay leafroll (mean pool 0.591, Δ +0.055
  [−0.014, +0.119]) is the closest; Vidal leafroll, with 26 positive leaves, falls below chance.
- Pooled over all 998 leaves, the CNN beats logistic regression on red blotch (mean pool 0.691 vs 0.647, Δ +0.044
  [+0.019, +0.070]; 4 of the 8 CNNs clear 0, all 8 did on the interim labels) but not reliably on leafroll (attention
  pool 0.597 vs 0.572, Δ +0.026 [−0.010, +0.061]; only the small ResNet with attention clears 0, two did before).
  Its per-variety scores sit near chance (0.42–0.56), so the gain comes from varieties differing in disease rate (Cab
  Franc is 96% red blotch, Vidal 10% leafroll). Its P ≥ 0.5 cut-off follows the pooled class balance, which also pulls
  its per-variety scores down. Without Cab Franc the red-blotch gain shrinks to +0.019 [−0.009, +0.046] (mean pool,
  0.619 vs 0.600).
- No backbone changes the picture. Removing max pooling moves the small ResNet by −0.03 to +0.03, within the spread
  between seeds; ResNet-18 (11.8 M weights for about 250 leaves) is the weakest CNN on 204 bands. Of the 48 new comparisons with
  logistic regression, 5 intervals sit above 0: four are Baco RGB leafroll, where logistic regression is below chance
  (0.418), and one is Baco leafroll with no max pool and attention (+0.081 [+0.009, +0.150]). That one calls whole rows
  positive (R1, R5) or negative (R3, R4, the row with 20% infected); each row alone scores 0.50–0.64. Pooled, both
  new backbones land within 0.05 of the small CNN (−0.03 to +0.05), and their leafroll and red-blotch scores for Baco,
  Chardonnay and Vidal stay at 0.41–0.60.
- ECA, CBAM and mean ∥ max change nothing reliably. Of 180 paired intervals against the same model without the change,
  41 exclude 0 (29 above, 12 below), against about 9 by chance; these intervals resample leaves, not training runs, so
  they miss run-to-run noise. Within a variety, 16 of the 17 gains are where the plain model was weak: Baco and Vidal
  RGB leafroll, from below chance (0.37–0.44) to 0.46–0.56, and Chardonnay red blotch, from 0.52–0.55 up to logistic
  regression's level (0.57–0.61 vs 0.605). No new model beats logistic regression within a variety except on those two
  RGB leafroll sets, where logistic regression itself is below chance (0.418, 0.439). The same module can help with one
  pooling and hurt with the other (small ResNet + ECA, pooled RGB leafroll: mean −0.044, attention +0.047); mean ∥ max
  is 3 above, 3 below out of 36. Pooled, CBAM in the small CNN scores highest (all 998, 204 bands: leafroll 0.624, Δ
  +0.052 [+0.016, +0.088] vs logistic regression; red blotch 0.702, Δ +0.055 [+0.028, +0.081]), but every new pooled
  model scores 0.42–0.59 on each variety's own leaves, so the gain is still the varieties' disease rates.
- RGB does about as well as the full spectrum: the extra bands add nothing we can detect at this sample size.
- CNN scores are fragile. Reruns with the same inputs are bit-identical, yet relabelling 7 of 998 leaves (all Vidal)
  moved the pooled CNN scores by up to 0.04 (median 0.008), about the size of their gains over logistic regression
  (so several gains clear 0 on one set of labels and not the other), and moved their Baco and Chardonnay leafroll
  and red-blotch scores, whose labels did not change, by up to 0.07; logistic regression moved by at most 0.02. Likewise Chardonnay red blotch (CNN mean) was 0.613 on the earlier
  136 px canvas and is 0.523 now, although the same seeds scored almost the same alone (0.52 / 0.60 / 0.54 then,
  0.52 / 0.56 / 0.54 now).

**Beyond the CNN.** A wider search on the interim labels (over 500 other models: other features, tuning, TabPFN, DINOv2, leaf-trait
inversion) found nothing that reaches 0.70 or beats logistic regression beyond noise within a variety, so it was removed
from the repo (code at commit `fb310b9`). One finding from it: published leaf-level accuracies of 80–87% (Sawyer et al. 2023) shuffle leaves
across rows; with that split this data also scores only 0.50–0.63.

Figures: `*/figures/<run>/balanced_accuracy.png`; architecture diagrams in `within_variety_disease_cnn/figures/` and
`cnn/figures/`. How to rerun: each folder's README.
