---
name: prefer-pytorch
description: Use whenever writing or refactoring numerical / ML code in this repo (models, training loops, array math, statistics, data on the GPU). PyTorch is the default library; NumPy, scikit-learn, SciPy etc. only where PyTorch has no reasonable equivalent or swapping would change a validated result.
---

# Prefer PyTorch

PyTorch is the default library for every numerical step in this repo. The machine has an NVIDIA
RTX PRO 5000 (48 GB); use it.

## Rules

1. **Models and training: always PyTorch.** Neural networks, logistic/linear models trained by
   gradient descent, losses, optimisers, mixed precision (`torch.autocast`), data on the GPU.
2. **Array math in new code: torch tensors**, not NumPy: means, sums of squares, matrix products,
   distances, kernels, SVD / PCA (`torch.linalg.svd`, `torch.pca_lowrank`), permutations
   (`torch.randperm` with a seeded `torch.Generator`), effect sizes. Batch / vectorise on the GPU
   instead of Python loops.
3. **Device handling:** pick the device once (`torch.device("cuda" if torch.cuda.is_available() else
   "cpu")`), move data once, keep it there; convert to NumPy only at the boundary (writing files,
   plotting, pandas tables).
4. **Seeds:** seed every generator you use (`torch.manual_seed`, explicit `torch.Generator`), and
   say so in a comment.
5. **Other libraries are allowed only when** (write the reason in a one-line comment):
   - there is no practical PyTorch equivalent (e.g. LIBSVM's `SVC` with precomputed kernels,
     `scipy.sparse` graph utilities, `skimage` image ops, HDF5 / parquet I/O, matplotlib);
   - the code is a standard evaluation metric where an exact, well-tested implementation matters
     (`sklearn.metrics` ROC AUC, balanced accuracy, confusion matrix);
   - the code reproduces a published or already-reported result, and swapping the library would
     change those numbers (e.g. a reported scikit-learn logistic regression). Replacing it is a
     new experiment, not a refactor: ask first.
6. **When refactoring existing code:** convert to PyTorch only where the outputs stay the same to
   the reported precision (same random streams, same algorithm). Verify by rerunning and comparing
   the outputs before and after. If a swap would change reported numbers, keep the old library and
   add the reason comment instead.

## Checklist before finishing

- [ ] No new NumPy/sklearn/SciPy computation where torch would do the same job.
- [ ] Every remaining non-torch numerical call has a one-line "why not torch" comment (except I/O,
      plotting and `sklearn.metrics`).
- [ ] Device chosen once; no silent CPU<->GPU copies inside loops.
- [ ] Outputs verified unchanged (refactor) or the change is reported (new experiment).
