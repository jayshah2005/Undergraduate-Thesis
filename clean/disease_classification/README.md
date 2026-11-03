# Disease classification

Detecting disease inside one variety, or across all of them, on held-out field rows.

| path | what it is |
|---|---|
| [`logistic_regression/`](logistic_regression/) | Logistic regression on the leaf's mean spectrum (the baseline), plus the leaves, labels, row folds and AUC both models use |
| [`cnn/`](cnn/) | Whole-leaf CNN with mean or attention pooling, for any variety or all of them, and any disease task |
| [`REPORT.md`](REPORT.md) | CNN vs logistic regression. Current results: Chardonnay, leafroll and red blotch, on the full spectrum and on RGB bands only |
