# Which bands drive variety prediction (SNV, 204-band model)

Full 204-band nested-CV balanced accuracy is **0.939**. Univariate scores use the same 5 row folds with C = 1. Region scores retune C inside each training split, like the main model. Permutation uses each fold's chosen C.

## Regions (not a single band)

| region | n bands | balanced accuracy | macro-F1 |
|---|---|---|---|
| all 204 bands | 204 | 0.939 | — |
| blue 397-500 nm | 36 | 0.766 | 0.759 |
| green 500-600 nm | 34 | 0.883 | 0.884 |
| red 600-680 nm | 27 | 0.887 | 0.887 |
| red edge 680-760 nm | 27 | 0.864 | 0.865 |
| NIR 760-1004 nm | 80 | 0.803 | 0.803 |
| visible 397-700 nm | 104 | 0.896 | 0.895 |

## Fewest bands (ANOVA-F selection *inside* each training split)

| k best bands | balanced accuracy |
|---|---|
| 1 | 0.637 |
| 3 | 0.759 |
| 5 | 0.754 |
| 10 | 0.797 |
| 20 | 0.848 |
| 50 | 0.895 |

## Strongest individual wavelengths

### Best one-band models

| nm | univariate bacc | mean |coef| | permutation drop |
|---|---|---|---|
| 690.6 | 0.710 | 0.213 | 0.0090 |
| 684.7 | 0.648 | 0.111 | 0.0030 |
| 687.7 | 0.645 | 0.155 | 0.0050 |
| 681.7 | 0.630 | 0.102 | 0.0000 |
| 675.7 | 0.618 | 0.089 | -0.0020 |
| 678.7 | 0.617 | 0.095 | 0.0000 |
| 672.8 | 0.615 | 0.084 | -0.0010 |
| 640.0 | 0.609 | 0.061 | -0.0030 |

### Largest |coefficient| in the full model

| nm | univariate bacc | mean |coef| | permutation drop |
|---|---|---|---|
| 747.5 | 0.392 | 0.267 | 0.0180 |
| 1003.6 | 0.432 | 0.265 | 0.0150 |
| 750.5 | 0.338 | 0.242 | 0.0180 |
| 744.5 | 0.425 | 0.238 | 0.0070 |
| 690.6 | 0.710 | 0.213 | 0.0090 |
| 693.6 | 0.591 | 0.209 | 0.0020 |
| 753.5 | 0.334 | 0.198 | 0.0190 |
| 741.5 | 0.428 | 0.197 | 0.0050 |

### Largest permutation drop (band is not redundant with the others)

| nm | univariate bacc | mean |coef| | permutation drop |
|---|---|---|---|
| 753.5 | 0.334 | 0.198 | 0.0190 |
| 747.5 | 0.392 | 0.267 | 0.0180 |
| 750.5 | 0.338 | 0.242 | 0.0180 |
| 756.6 | 0.325 | 0.195 | 0.0160 |
| 1003.6 | 0.432 | 0.265 | 0.0150 |
| 982.0 | 0.455 | 0.142 | 0.0120 |
| 988.1 | 0.436 | 0.123 | 0.0110 |
| 932.7 | 0.400 | 0.140 | 0.0100 |

Shaded regions in `figures/band_importance.png`: blue, green, red, red edge, NIR.
