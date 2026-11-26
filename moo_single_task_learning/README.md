# Six-head CNN trained with multi-objective optimization

Being rebuilt (2026-11-25): the six-head runs move onto the shared small CNN and small ResNet (with max pooling), with
AdamW and SGD, 204 bands and RGB, 5 seeds × 5 row folds. The earlier runs (the original network below, and the small
ResNet without max pooling) were removed; they are in git history at `0538fdd`, with their outcome in `../TESTED.md`.

| file | what it is |
|---|---|
| `moo.py` | per-head losses and gradients; the aggregators: LS, MGDA (Sener & Koltun 2018, "Multi-Task Learning as Multi-Objective Optimization", NeurIPS) with or without gradient normalization (GN, LN, LGN, as listed in LibMTL, Lin & Zhang 2023), and torchjd's SVD-PGA, OSNMF-PGA (Principal Gradient Alignment, local torchjd, not published) and UPGrad (Quinton & Rey 2024, "Jacobian Descent for Multi-Objective Optimization") |
| `data.py`, `model.py`, `metrics.py`, `train.py` | the original six-head network and its training (1 × 1 spectral mix → two 3 × 3 convs → attention pooling → six logits), kept |
