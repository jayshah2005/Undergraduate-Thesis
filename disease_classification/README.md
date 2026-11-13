# Disease classification

Detecting leafroll and red blotch on held-out field rows, one variety at a time or with varieties pooled.

| path | what it is |
|---|---|
| [`logistic_regression/`](logistic_regression/) | logistic regression on the leaf's mean spectrum (the baseline), plus the leaves, labels, row folds and scoring every model uses |
| [`within_variety_disease_cnn/`](within_variety_disease_cnn/) | one whole-leaf CNN per variety (Baco, Chardonnay, Vidal), trained on that variety only |
| [`cnn/`](cnn/) | one whole-leaf CNN for pooled varieties (all 998 leaves, or Baco + Chardonnay + Vidal); holds the network and training code both CNN folders use |
| [`model_search/`](model_search/) | search over 500+ other models (features, tuning, TabPFN, DINOv2, PROSPECT-D traits, …) toward 0.70, with honest selection on validation rows |
| [`REPORT.md`](REPORT.md) | short summary: CNN vs logistic regression, within-variety vs pooled, full spectrum vs RGB |

Every CNN sees a 212 × 212 canvas (the largest leaf, rounded up) with 204 bands or 3 RGB bands.
