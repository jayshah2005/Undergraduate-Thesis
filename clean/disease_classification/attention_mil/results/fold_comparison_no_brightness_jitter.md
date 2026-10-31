# Fold schemes compared (Chardonnay)

- `row`: fold k = field row k: the test row is never seen in training (new part of the vineyard)
- `photo`: random photos dealt evenly per row (2 per fold; R1 has 13 photos, so 3 or 2): new vines in rows seen in training
- `block`: fold k = the photos of each row whose first vine is in 10k+1 to 10k+10: like photo, but the held-out vines are neighbours of each other, not of the training vines (except at block edges)

Within-fold AUC (same fold and same row only), 3-seed ensemble [95% bootstrap CI]; Δ = paired difference to the 4-fold baseline of the same scheme.

## Leafroll (122 + / 128 −)

| model | `row` | `photo` | `block` |
|---|---|---|---|
| baseline (4 folds) | 0.460 [0.387, 0.538] | 0.489 [0.408, 0.572] | 0.433 [0.355, 0.509] |
| baseline (3 folds) | 0.464 [0.389, 0.540]<br>Δ +0.004 [-0.050, +0.055] | 0.515 [0.437, 0.596]<br>Δ +0.026 [-0.033, +0.085] | 0.496 [0.418, 0.574]<br>Δ +0.063 [+0.011, +0.117] |
| band z-score + mean pool | 0.535 [0.458, 0.619]<br>Δ +0.074 [-0.020, +0.169] | 0.463 [0.386, 0.544]<br>Δ -0.026 [-0.129, +0.072] | 0.528 [0.448, 0.608]<br>Δ +0.095 [-0.002, +0.193] |
| band z-score + attention | 0.546 [0.469, 0.624]<br>Δ +0.086 [-0.007, +0.177] | 0.476 [0.400, 0.550]<br>Δ -0.013 [-0.111, +0.081] | 0.513 [0.433, 0.591]<br>Δ +0.080 [-0.019, +0.184] |
| pixel SNV + mean pool | 0.500 [0.422, 0.573]<br>Δ +0.039 [-0.052, +0.125] | 0.594 [0.522, 0.670]<br>Δ +0.105 [+0.004, +0.203] | 0.459 [0.381, 0.541]<br>Δ +0.026 [-0.078, +0.128] |
| pixel SNV + attention | 0.520 [0.446, 0.597]<br>Δ +0.060 [-0.019, +0.139] | 0.568 [0.489, 0.646]<br>Δ +0.079 [-0.031, +0.183] | 0.496 [0.411, 0.580]<br>Δ +0.063 [-0.032, +0.154] |

## Red blotch (88 + / 162 −)

| model | `row` | `photo` | `block` |
|---|---|---|---|
| baseline (4 folds) | 0.565 [0.482, 0.646] | 0.604 [0.533, 0.671] | 0.577 [0.498, 0.660] |
| baseline (3 folds) | 0.602 [0.521, 0.679]<br>Δ +0.036 [-0.019, +0.096] | 0.579 [0.504, 0.650]<br>Δ -0.024 [-0.079, +0.026] | 0.547 [0.466, 0.628]<br>Δ -0.029 [-0.092, +0.034] |
| band z-score + mean pool | 0.640 [0.564, 0.713]<br>Δ +0.074 [-0.007, +0.162] | 0.587 [0.514, 0.654]<br>Δ -0.016 [-0.100, +0.067] | 0.572 [0.493, 0.655]<br>Δ -0.005 [-0.101, +0.095] |
| band z-score + attention | 0.640 [0.568, 0.709]<br>Δ +0.074 [-0.013, +0.160] | 0.614 [0.541, 0.687]<br>Δ +0.010 [-0.069, +0.089] | 0.559 [0.477, 0.644]<br>Δ -0.018 [-0.115, +0.079] |
| pixel SNV + mean pool | 0.588 [0.513, 0.656]<br>Δ +0.022 [-0.061, +0.106] | 0.591 [0.526, 0.659]<br>Δ -0.012 [-0.091, +0.067] | 0.559 [0.475, 0.640]<br>Δ -0.018 [-0.110, +0.077] |
| pixel SNV + attention | 0.491 [0.412, 0.572]<br>Δ -0.074 [-0.160, +0.016] | 0.620 [0.553, 0.687]<br>Δ +0.016 [-0.061, +0.091] | 0.536 [0.457, 0.615]<br>Δ -0.041 [-0.126, +0.041] |
