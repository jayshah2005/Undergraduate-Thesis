# Logistic regression on the mean spectrum

The disease baseline: StandardScaler + class-balanced logistic regression on the leaf's SNV mean spectrum.
It is the score the CNN in `../cnn/` has to beat.

| path | what it is |
|---|---|
| `logistic_regression.py` | the leaves, task labels and row folds both models use; fitting; the metrics table |
| `scoring.py` | the within-row AUC and its bootstrap, used here, in the CNN's training log and in the CNN report |
| `results/<variety>[_rgb]/metrics.md` | each task's within-row AUC with 95% CI, per row, and per variety under `all` |

The baseline takes seconds and is deterministic, so the CNN's scripts refit it rather than read a saved copy, as the
variety CNN report does. The numbers are the same as here.

## Folds

Fold k is field row k of every variety. Test = row k, validation = row k+1, train = the other 3 rows. The baseline
has no setting to choose, so it leaves the validation row unused and trains on the same 3 rows as the CNN.
AUCs compare leaves of the same variety and row only, so neither recognising a variety nor a row's disease rate can
raise them. Every photo (the camera image the leaves were cut from) lies inside one row, so its shared lighting never
spans training and test.

## Tasks

| task | positive | negative |
|---|---|---|
| `leafroll` | leafroll present | no leafroll |
| `red_blotch` | red blotch present | no red blotch |
| `both` | both diseases present | everything else, including leaves with only one |
| `any` | any disease present | healthy |

A task leaves out any variety with fewer than 5 positive or 5 negative leaves. Under `all`, that means Cab Franc
sits out `leafroll` and `both` (1 positive leaf each). Cab Franc on `red_blotch` or `any` is allowed, but it has only
9 healthy leaves, so expect wide intervals. Run on its own, the script scores every task the chosen leaves allow.

## Bands (`--bands`)

`all` (default) uses the full spectrum, 204 bands. `rgb` keeps only the bands nearest 640, 550 and 460 nm, the same
picks as the variety models, and writes to `results/<variety>_rgb/`. SNV runs over the bands kept, so under `rgb` it
centres and scales 3 numbers per leaf, which leaves little more than their shape.

## Running

```bash
cd clean/disease_classification/logistic_regression
../../../.venv/bin/python -u logistic_regression.py                 # Chardonnay, a few seconds → results/chardonnay/
../../../.venv/bin/python -u logistic_regression.py --variety all   # all 998 leaves → results/all/
../../../.venv/bin/python -u logistic_regression.py --bands rgb     # RGB bands only → results/chardonnay_rgb/
```
