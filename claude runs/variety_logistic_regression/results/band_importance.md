# Which bands drive variety prediction (SNV, 204-band model)

Full 204-band nested-CV balanced accuracy is **0.939**. Univariate scores use the same 5 row folds with C = 1. Region scores retune C inside each training split, like the main model. Permutation uses each fold's chosen C.

## Regions (not a single band)

| region | n bands | balanced accuracy | macro-F1 |
|---|---|---|---|
| all 204 bands | 204 | 0.939 | — |
| blue 400-500 nm | 34 | 0.753 | 0.745 |
| green 500-600 nm | 34 | 0.888 | 0.889 |
| red 600-680 nm | 27 | 0.898 | 0.898 |
| red edge 680-760 nm | 27 | 0.853 | 0.853 |
| NIR 760-1000 nm | 82 | 0.820 | 0.821 |
| visible 400-700 nm | 102 | 0.895 | 0.894 |

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
| 695.6 | 0.710 | 0.213 | 0.0090 |
| 689.7 | 0.648 | 0.111 | 0.0030 |
| 692.6 | 0.645 | 0.155 | 0.0050 |
| 686.7 | 0.630 | 0.102 | 0.0000 |
| 680.8 | 0.618 | 0.089 | -0.0020 |
| 683.7 | 0.617 | 0.095 | 0.0000 |
| 677.8 | 0.615 | 0.084 | -0.0010 |
| 645.3 | 0.609 | 0.061 | -0.0030 |

### Largest |coefficient| in the full model

| nm | univariate bacc | mean |coef| | permutation drop |
|---|---|---|---|
| 751.7 | 0.392 | 0.267 | 0.0180 |
| 1000.0 | 0.432 | 0.265 | 0.0150 |
| 754.7 | 0.338 | 0.242 | 0.0180 |
| 748.8 | 0.425 | 0.238 | 0.0070 |
| 695.6 | 0.710 | 0.213 | 0.0090 |
| 698.5 | 0.591 | 0.209 | 0.0020 |
| 757.6 | 0.334 | 0.198 | 0.0190 |
| 745.8 | 0.428 | 0.197 | 0.0050 |

### Largest permutation drop (band is not redundant with the others)

| nm | univariate bacc | mean |coef| | permutation drop |
|---|---|---|---|
| 757.6 | 0.334 | 0.198 | 0.0190 |
| 751.7 | 0.392 | 0.267 | 0.0180 |
| 754.7 | 0.338 | 0.242 | 0.0180 |
| 760.6 | 0.325 | 0.195 | 0.0160 |
| 1000.0 | 0.432 | 0.265 | 0.0150 |
| 979.3 | 0.455 | 0.142 | 0.0120 |
| 985.2 | 0.436 | 0.123 | 0.0110 |
| 932.0 | 0.400 | 0.140 | 0.0100 |

Shaded regions in `figures/band_importance.png`: blue, green, red, red edge, NIR.
