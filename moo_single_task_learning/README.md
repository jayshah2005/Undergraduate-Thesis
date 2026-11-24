# Six-head CNN trained with multi-objective optimization

One whole-leaf CNN with six sigmoid heads (Baco, Cab Franc, Chardonnay, Vidal, red blotch, leafroll) on the 998
leaves and 204 bands. Each head is one objective. The methods differ only in how the six heads' gradients become one
update: their sum (LS), MGDA (Sener & Koltun 2018, "Multi-Task Learning as Multi-Objective Optimization", NeurIPS)
with or without gradient normalization (GN, LN, LGN, as listed in LibMTL, Lin & Zhang 2023), or the Principal
Gradient Alignment aggregators of the local torchjd (`../../torchjd`, not yet published). Results:
[`results/final_report.md`](results/final_report.md).

| file | what it is |
|---|---|
| `data.py` | loads the leaves, the six labels and the row folds; pads each mini-batch and applies SNV per pixel |
| `model.py` | `SixHeadCNN`: 1 × 1 spectral mix → two 3 × 3 convs → gated attention pooling → six logits |
| `moo.py` | per-head losses and gradients; LS, MGDA (+ GN / LN / LGN) and the torchjd aggregators |
| `metrics.py` | balanced accuracy and AUC per head, 4-way variety score |
| `train.py` | `train_fold`: one row-held-out fold with early stopping on the summed validation loss; `METHODS` |
| `run_moo.py` | every (method, seed, fold): seeds 100 · seed + fold, deterministic, resumable; saves test probabilities and the weight per head |
| `report_moo.py` | scores the runs with the variety and disease reports' own code, next to logistic regression and the single-task CNNs |
| `results/` | `final_report.md`, `config.json` (settings and code hashes per method), one `.npz` per method, seed and fold, logs |

Folds, seeds and the 3-seed ensemble are the same as in `variety_classification/` and `disease_classification/`, so
every comparison is paired.

```bash
cd moo_single_task_learning
../.venv/bin/python -u run_moo.py --method ls                 # also mgda, mgda_gn, mgda_ln, mgda_lgn
PYTHONPATH=../../torchjd/src ../.venv/bin/python -u run_moo.py --method svdpga   # also osnmfpga
../.venv/bin/python -u report_moo.py                          # -> results/final_report.md
../.venv/bin/python -u run_moo.py --method ls --smoke         # quick check: 1 seed, 5 folds, 1 epoch, results/smoke/
```
