# Variety classification

Predicting a leaf's variety (Baco / Cab Franc / Chardonnay / Vidal), and how much of the score needs more than a colour photograph. Results: [`REPORT.md`](REPORT.md).

| path | what it is |
|---|---|
| [`logistic_regression/`](logistic_regression/) | logistic regression on the leaf's mean spectrum, plus which wavelengths drive it |
| [`cnn/`](cnn/) | small whole-leaf CNN |

Both evaluate on the same held-out field rows, and both take RGB as a setting: `--bands all` uses all 204 bands, `--bands rgb` only the bands nearest 640 / 550 / 460 nm.
