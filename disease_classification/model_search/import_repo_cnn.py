"""Turn CNN fold files (3-seed ensemble) into harness prediction files, so CNNs join the same comparison.

usage: import_repo_cnn.py <name> <folder> <variety or pool>   e.g. cnn_small_resnet_rgb results/baco_rgb_small_resnet Baco
A within-variety folder is saved as <name>_<arm>@within|<variety>, merged with the other varieties' folders
by selection.py; a pooled folder is saved as <name>_<arm>@pooled.
"""
import sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]
for p in (R, R / "disease_classification/logistic_regression", R / "disease_classification/cnn"):
    sys.path.insert(0, str(p))
from logistic_regression import fold_of, load_meta, task_labels, usable_leaves  # noqa: E402
from run_experiments import SEEDS, run_dir  # noqa: E402
from harness import PRED, TASKS, VARIETIES  # noqa: E402


def convert(name: str, folder: Path, variety: str) -> None:
    meta = load_meta(variety)
    fold, var = fold_of(meta), meta.variety.to_numpy()
    for arm in ("mean", "attention"):
        arrs = {}
        for task in TASKS:
            y = task_labels(meta, task)
            keep = usable_leaves(meta, y)
            val, test = np.full((len(SEEDS), len(y)), np.nan), np.full((len(SEEDS), len(y)), np.nan)
            for i, s in enumerate(SEEDS):
                for k in range(5):
                    f = np.load(run_dir(task, arm, s, folder) / f"fold{k}.npz")
                    assert (fold[f["test_idx"]] == k).all() and (fold[f["val_idx"]] == (k + 1) % 5).all()
                    test[i, f["test_idx"]], val[i, f["val_idx"]] = f["prob"], f["val_prob"]
            val, test = val.mean(0), test.mean(0)
            for v in VARIETIES:
                m = keep & (var == v)
                if not m.any():
                    continue
                assert not np.isnan(val[m]).any() and not np.isnan(test[m]).any(), (folder, task, v)
                arrs.update({f"{task}|{v}|val": val[m], f"{task}|{v}|test": test[m], f"{task}|{v}|y": y[m],
                             f"{task}|{v}|row": fold[m]})
        mode = "pooled" if variety not in VARIETIES else f"within|{variety}"
        np.savez(PRED / f"{name}_{arm}@{mode}.npz", **arrs)


if __name__ == "__main__":
    convert(sys.argv[1], R / sys.argv[2], sys.argv[3])
