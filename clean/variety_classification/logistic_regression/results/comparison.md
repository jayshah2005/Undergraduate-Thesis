# Variety prediction: configurations compared

Balanced accuracy = the mean of the 4 recalls shown here; chance is 0.25. Every score comes from the same 5 held-out field rows, each model trained on 3 rows with C chosen on a fourth.

| configuration | balanced accuracy (95% CI) | Baco recall | Cab Franc recall | Chardonnay recall | Vidal recall | per-row range |
|---|---|---|---|---|---|---|
| `all` | 0.874 (0.853–0.894) | 0.907 | 0.848 | 0.944 | 0.796 | 0.61–0.97 |
| `rgb` | 0.503 (0.472–0.533) | 0.351 | 0.344 | 0.768 | 0.548 | 0.42–0.60 |
