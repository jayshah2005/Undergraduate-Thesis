# Which bands drive variety prediction (all bands)

Balanced accuracy; chance 0.25.

## Regions

| region | bands | balanced accuracy |
|---|---|---|
| all | 204 | 0.939 |
| blue 397-500 nm | 36 | 0.766 |
| green 500-600 nm | 34 | 0.882 |
| red 600-680 nm | 27 | 0.889 |
| red edge 680-760 nm | 27 | 0.864 |
| NIR 760-1004 nm | 80 | 0.803 |
| visible 397-700 nm | 104 | 0.898 |

## Fewest bands (picked by ANOVA F inside each training split)

| k best bands | balanced accuracy |
|---|---|
| 1 | 0.637 |
| 3 | 0.759 |
| 5 | 0.754 |
| 10 | 0.797 |
| 20 | 0.848 |
| 50 | 0.895 |

## Best one-band models

| nm | one-band accuracy | permutation drop |
|---|---|---|
| 690.6 | 0.710 | 0.0060 |
| 684.7 | 0.648 | 0.0000 |
| 687.7 | 0.645 | 0.0030 |
| 681.7 | 0.630 | 0.0040 |
| 675.7 | 0.618 | -0.0020 |
| 678.7 | 0.617 | 0.0030 |
| 672.8 | 0.615 | 0.0020 |
| 640.0 | 0.609 | 0.0040 |

## Largest permutation drop

| nm | one-band accuracy | permutation drop |
|---|---|---|
| 1003.6 | 0.432 | 0.0200 |
| 756.6 | 0.325 | 0.0150 |
| 753.5 | 0.334 | 0.0130 |
| 747.5 | 0.392 | 0.0130 |
| 889.9 | 0.435 | 0.0120 |
| 750.5 | 0.338 | 0.0100 |
| 985.0 | 0.443 | 0.0090 |
| 759.6 | 0.366 | 0.0090 |
