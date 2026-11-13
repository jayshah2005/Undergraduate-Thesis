# Disease model search

**Question.** Can any leaf-level model reach 0.70 balanced accuracy for leafroll or red blotch on held-out field
rows, within one variety (Baco, Chardonnay, Vidal) or with the three pooled?

**Answer.** No. Over 500 candidates and four rounds of selection, the honest within-variety scores stay at
0.51–0.52 for leafroll and 0.53–0.57 for red blotch (repo baseline 0.535 / 0.554). Pooled, nothing beats predicting
from the variety's infection rate alone (0.679 / 0.615). Inside a row, the median candidate's AUC is 0.50 for
leafroll and 0.55 for red blotch. Numbers: [`final_report.md`](final_report.md); secondary scores, the split
comparison with Sawyer et al. 2023, permutation nulls and PROSPECT trait effects: [`metrics.md`](metrics.md).

## Protocol (fixed before the first run)

- **Folds.** The repo's: test row k, validation row k+1, train on the other 3 (`../logistic_regression`).
- **Tuning.** Inner leave-one-row-out on the 3 training rows (mean within-variety AUC). The validation and test rows
  never tune anything, and every transform (scaler, PCA, band selection, per-variety centring) is fitted on the
  training rows only.
- **Balance.** Class-balanced inside a variety. Pooled models balance class within each variety (group balancing,
  Idrissi et al. 2022), so the variety alone predicts nothing; the pooled score then adds the variety's training
  infection rate back by Bayes prior adjustment (Saerens et al. 2002, `pooled.py`).
- **CNNs.** Early-stopped on one training row, then retrained on all 3 for that many epochs (Goodfellow et al. 2016,
  Algorithm 7.2), so their validation scores are clean. The repo's CNNs, early-stopped on the validation row, are
  imported for reference only (`ES_` prefix) and never ranked.
- **Selection (headline).** For each test row k, the candidate with the best validation score on the other rows'
  leaves scores row k (`selection.honest`; pooled: `pooled.search_scores`). Secondary: own-validation-row pick,
  top-5 / top-10 averages.

## Files

| file | what it is |
|---|---|
| `harness.py` | protocol, data, folds, fitting one candidate, saving its predictions to `cache/disease_search/preds/` |
| `selection.py`, `pooled.py` | within-variety and pooled selection and scoring |
| `report.py` | writes `final_report.md` and `metrics.md` from the saved predictions (about a minute) |
| `build_*.py` | per-leaf feature caches in `cache/disease_search/`: indices and pixel quantiles (`build_features`), GLCM texture, 400 sampled pixels, DINOv2 embeddings, leaf shape, PROSPECT-D traits |
| `family_*.py` | the candidate families, one file each (table below) |
| `film.py` | the variety-conditioned CNN used by `family_cnn.py` |
| `check_baseline.py`, `check_pooled.py` | the harness reproduces the repo baseline exactly; a variety-only feature scores exactly chance when pooled |
| `permutation_null.py`, `sawyer_protocol.py`, `import_repo_cnn.py` | permutation nulls; Sawyer et al.'s shuffled split vs held-out rows; repo CNN runs into the prediction format |
| `run_search.sh` | every step in the order it ran |

## Candidate families

Each file's docstring has the full references and every adaptation. "Own design" = no paper found.

| file | candidates |
|---|---|
| `family_spectral.py` | leaf-mean features (SNV spectrum, RGB, 510–710 nm as in Sawyer et al. 2023, first derivative, 18 vegetation indices, pixel quantiles) × logistic regression, PCA + LR, RBF SVM, shrinkage LDA, random forest, gradient boosting; and the repo baseline |
| `family_dino.py` | frozen DINOv2 ViT-S/14 embeddings (Oquab et al.) of RGB and colour-infrared renders |
| `family_rownorm_texture.py` | per-row feature normalisation (AdaBN analogue, Li et al. 2018); GLCM texture of index maps (Haralick et al. 1973) |
| `family_pixel_sil.py` | pixel-level logistic regression with leaf labels, aggregated per leaf (Ray & Craven 2005) |
| `family_photo_centring.py` | subtract each camera image's mean feature (own design) |
| `family_multitask.py` | shared + per-variety weights (Daumé III 2007), pooled |
| `family_shape.py` | leaf shape (Hu 1962 moments) and edge-to-centre profiles for margin rolling (own design) |
| `family_cnn.py`, `film.py` | whole-leaf CNNs (repo backbones) with a clean validation row; FiLM variety conditioning (Perez et al. 2018) |
| `family_tabpfn.py` | TabPFN v2 (Hollmann et al. 2025) |
| `family_tuned.py`, `family_tuned_tabpfn.py` | wide hyperparameter search of every algorithm (random search for trees, Bergstra & Bengio 2012), PLS-DA (Barker & Rayens 2003), k-NN; prefix `tuned:` |
| `family_literature.py` | PROSPECT-D traits (Féret et al. 2017; Zarco-Tejada et al. 2018), CARS-PLS-DA (Li et al. 2009), SPA-LDA (Araújo et al. 2001; Pontes et al. 2005), joint 4-class co-infection model (Sawyer et al. 2023); prefix `lit:` |

## How to run

Each step skips candidates already saved. CPU families take minutes to an hour each; the CNNs take about 3
GPU-hours in all. `run_search.sh` lists every command.

```bash
cd disease_classification/model_search
../../.venv/bin/python build_features.py        # then the other build_*.py
../../.venv/bin/python family_spectral.py       # then the other family_*.py (see run_search.sh)
../../.venv/bin/python report.py                # → final_report.md, metrics.md
```

- `build_dino.py` needs `timm`; `build_texture.py` and `build_shape.py` need `scikit-image`;
  `build_prospect.py` needs `prosail` (all in `requirements.txt`).
- TabPFN (`family_tabpfn.py`, `family_tuned_tabpfn.py`) needs `tabpfn==2.0.9`, which needs scikit-learn < 1.7, so
  it runs in its own environment (the search used scikit-learn 1.6.1 on `PYTHONPATH`).
- `prospect.npz` behind the reported numbers came from a port of prosail's PROSPECT-D in the session (see
  `build_prospect.py`); rerunning with the package may move the traits slightly.

## Caveats

- The same test rows scored every round (search, pooled, tuned, literature), so every headline is a little
  optimistic, and more so the more rounds are compared.
- Balanced accuracy over all rows can rise when a model's positive-call rate follows each row's infection rate
  (rows range from 20% to 66% leafroll). `metrics.md` gives within-row AUC to check for that.
- Published leaf-level accuracies of 80–87% (Sawyer et al. 2023) use folds that shuffle leaves across rows. With
  that split this data still scores 0.50–0.63 (`metrics.md`, split comparison).
