# Controlled PMMFNet experiments - comparison

Leaf-level, out-of-fold test metrics (998 leaves, 5 row folds). Decisions = most probable class. 'Original' is the historical run (checkpoint by minimum validation loss); every other row selects checkpoints by maximum validation balanced accuracy. L1 is the plan's patch-level reference for L2 and has exactly A2's configuration, so it is the A2 run; S3 is the same configuration rerun with the same seeds, which measures run-to-run noise.

## Main table

| experiment | balanced_accuracy | macro_f1 | recall healthy | recall RB only | recall LR only | recall both | precision healthy | precision RB only | precision LR only | precision both | mean OvR AUC | mean within-variety AUC | RB within-variety AUC | LR within-variety AUC | mean per-variety bal. acc. | runtime min | mean best val bal. acc. | mean best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | 0.398 | 0.373 | 0.195 | 0.588 | 0.518 | 0.293 | 0.229 | 0.776 | 0.282 | 0.214 | 0.683 | 0.514 | 0.558 | 0.487 | 0.274 | 35.2 | 0.416 | 2.2 |
| A1 | 0.413 | 0.385 | 0.173 | 0.652 | 0.525 | 0.302 | 0.267 | 0.780 | 0.272 | 0.230 | 0.698 | 0.512 | 0.558 | 0.482 | 0.288 | 41.9 | 0.439 | 4.0 |
| A2 | 0.402 | 0.378 | 0.190 | 0.629 | 0.496 | 0.293 | 0.253 | 0.793 | 0.265 | 0.215 | 0.685 | 0.522 | 0.583 | 0.473 | 0.292 | 38.6 | 0.436 | 3.2 |
| S1 | 0.437 | 0.417 | 0.239 | 0.673 | 0.475 | 0.362 | 0.297 | 0.795 | 0.300 | 0.266 | 0.715 | 0.530 | 0.580 | 0.509 | 0.307 | 44.2 | 0.443 | 4.6 |
| S2 | 0.403 | 0.373 | 0.159 | 0.625 | 0.525 | 0.302 | 0.245 | 0.788 | 0.265 | 0.211 | 0.693 | 0.519 | 0.577 | 0.488 | 0.282 | 47.7 | 0.433 | 5.4 |
| S3 | 0.406 | 0.375 | 0.133 | 0.646 | 0.554 | 0.293 | 0.201 | 0.763 | 0.280 | 0.250 | 0.691 | 0.509 | 0.560 | 0.484 | 0.274 | 41.2 | 0.433 | 3.8 |
| L1 (= A2 run) | 0.402 | 0.378 | 0.190 | 0.629 | 0.496 | 0.293 | 0.253 | 0.793 | 0.265 | 0.215 | 0.685 | 0.522 | 0.583 | 0.473 | 0.292 | 38.6 | 0.436 | 3.2 |
| L2 | 0.411 | 0.393 | 0.261 | 0.609 | 0.396 | 0.379 | 0.282 | 0.778 | 0.264 | 0.250 | 0.690 | 0.519 | 0.567 | 0.491 | 0.296 | 45.3 | 0.435 | 4.8 |

Descriptions: **Original** = historical baseline: checkpoint = min val loss, scaler on train+val; **A1** = original + checkpoint = max val bal. acc.; **A2** = A1 + scaler on training leaves only; **S1** = A2, no spatial shuffle; **S2** = A2, fixed shuffle also at validation/test; **S3** = A2 settings rerun (shuffle in training only); **L1 (= A2 run)** = patch-level loss reference: same configuration and run as A2; **L2** = A2 with leaf-level (MIL) training loss

## Per fold: test 4-class balanced accuracy

| fold | Original | A1 | A2 | S1 | S2 | S3 | L1 (= A2 run) | L2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.326 | 0.345 | 0.326 | 0.345 | 0.305 | 0.325 | 0.326 | 0.327 |
| 1 | 0.338 | 0.374 | 0.374 | 0.447 | 0.383 | 0.374 | 0.374 | 0.387 |
| 2 | 0.494 | 0.465 | 0.451 | 0.521 | 0.494 | 0.489 | 0.451 | 0.478 |
| 3 | 0.363 | 0.355 | 0.351 | 0.349 | 0.353 | 0.353 | 0.351 | 0.365 |
| 4 | 0.420 | 0.441 | 0.442 | 0.415 | 0.413 | 0.412 | 0.442 | 0.453 |

## Per fold: best validation 4-class balanced accuracy (the selection metric; Original = at its min-loss epoch)

| fold | Original | A1 | A2 | S1 | S2 | S3 | L1 (= A2 run) | L2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.313 | 0.362 | 0.343 | 0.362 | 0.340 | 0.344 | 0.343 | 0.368 |
| 1 | 0.396 | 0.416 | 0.427 | 0.451 | 0.428 | 0.416 | 0.427 | 0.443 |
| 2 | 0.382 | 0.416 | 0.416 | 0.390 | 0.400 | 0.399 | 0.416 | 0.467 |
| 3 | 0.501 | 0.515 | 0.517 | 0.526 | 0.500 | 0.507 | 0.517 | 0.460 |
| 4 | 0.489 | 0.485 | 0.476 | 0.487 | 0.498 | 0.497 | 0.476 | 0.438 |

## Rerun noise: experiments whose settings are identical in a fold

Same scaler values, shuffle policy, objective, checkpoint rule and seeds, so the differences below are training noise only (GPU kernel selection). Two or three reruns understate the full spread.

| fold | identical runs | test bal. acc. (min - max) | range | best val bal. acc. (min - max) |
|---|---|---|---|---|
| 0 | A2, S3 | 0.325 - 0.326 | 0.001 | 0.343 - 0.344 |
| 1 | A1, A2, S3 | 0.374 - 0.374 | 0.000 | 0.416 - 0.427 |
| 2 | A1, A2, S3 | 0.451 - 0.489 | 0.038 | 0.399 - 0.416 |
| 3 | A1, A2, S3 | 0.351 - 0.355 | 0.004 | 0.507 - 0.517 |
| 4 | A2, S3 | 0.412 - 0.442 | 0.030 | 0.476 - 0.497 |

## Paired comparisons (B minus A, test leaves)

The bootstrap resamples leaves with both trained models held fixed, so it does not include training noise; the last column counts folds where B - A is beyond +/-0.038, the widest rerun range above.

| question | B vs A | delta bal. acc. | 95% bootstrap CI | folds where B > A | delta macro-F1 | delta mean within-variety AUC | folds beyond +/-0.038: up / down |
|---|---|---|---|---|---|---|---|
| checkpoint selection: max val bal. acc. vs min val loss | A1 vs Original | +0.014 | [-0.013, +0.041] | 3/5 | +0.012 | -0.002 | 0 / 0 |
| scaler fitted on training leaves only vs training + validation leaves (settings differ only in folds 0 and 4; folds 1-3 are reruns) | A2 vs A1 | -0.011 | [-0.039, +0.017] | 1/5 | -0.007 | +0.010 | 0 / 0 |
| identical settings rerun: run-to-run noise floor | S3 vs A2 | +0.004 | [-0.020, +0.028] | 3/5 | -0.003 | -0.013 | 0 / 0 |
| no shuffle vs shuffle in training only | S1 vs S3 | +0.031 | [+0.006, +0.055] | 4/5 | +0.042 | +0.021 | 1 / 0 |
| no shuffle vs shuffle in training only (other run of the reference) | S1 vs A2 | +0.035 | [+0.004, +0.064] | 3/5 | +0.039 | +0.008 | 2 / 0 |
| shuffle in training and evaluation vs training only | S2 vs S3 | -0.004 | [-0.025, +0.019] | 3/5 | -0.003 | +0.011 | 0 / 0 |
| shuffle in training and evaluation vs training only (other run of the reference) | S2 vs A2 | +0.001 | [-0.021, +0.022] | 3/5 | -0.005 | -0.002 | 1 / 0 |
| leaf-level vs patch-level training loss | L2 vs L1 (= A2 run) | +0.009 | [-0.025, +0.043] | 5/5 | +0.015 | -0.002 | 0 / 0 |
| leaf-level vs patch-level training loss (other run of the reference) | L2 vs S3 | +0.005 | [-0.027, +0.036] | 4/5 | +0.018 | +0.011 | 1 / 0 |
