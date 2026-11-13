"""Permutation null for one candidate: shuffle labels within each variety x row (keeps every row's disease
rate), rerun the whole candidate (inner tuning included), record mean validation and test balanced accuracy.

usage: permutation_null.py <family module> <candidate name> <mode> [n_perm]
e.g.   permutation_null.py family_spectral svm:snv_vis within 50
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import harness
from harness import bal_acc, evaluate, load_all, TASKS, VARIETIES

mod, cname, mode = sys.argv[1], sys.argv[2], sys.argv[3]
n_perm = int(sys.argv[4]) if len(sys.argv) > 4 else 50
cands = __import__(mod).candidates() if mod != "family_spectral" else None
if mod == "family_spectral":
    from family_spectral import candidate
    model, feat = cname.split(":")
    cand = candidate(feat, model)
else:
    cand = next(c for c in cands if c.name == cname)
orig = harness.task_labels
data = load_all()
null = {t: [] for t in TASKS}
for i in range(n_perm):


    def shuffled(meta, task, _rng_seed=1000 + i):
        y = orig(meta, task).copy()
        r = np.random.default_rng(_rng_seed + {"leafroll": 0, "red_blotch": 500}[task])
        g = (meta.variety + "/" + meta.row).to_numpy()
        for grp in np.unique(g):
            m = np.flatnonzero(g == grp)
            y[m] = y[r.permutation(m)]
        return y
    harness.task_labels = shuffled
    res = evaluate(cand, data, mode)
    for t in TASKS:
        null[t].append((np.mean([bal_acc(res["preds"][(t, v)]["val"], res["preds"][(t, v)]["y"]) for v in VARIETIES]),
                        np.mean([bal_acc(res["preds"][(t, v)]["test"], res["preds"][(t, v)]["y"]) for v in VARIETIES])))
harness.task_labels = orig
for t in TASKS:
    a = np.array(null[t])
    print(f"{cname}@{mode} {t}: null val mean {a[:,0].mean():.3f} 95th pct {np.percentile(a[:,0],95):.3f} max {a[:,0].max():.3f}"
          f" | null test 95th pct {np.percentile(a[:,1],95):.3f}", flush=True)
np.save(f"{harness.CACHE}/perm_{cname.replace(':','_')}_{mode}.npy", np.array([null[t] for t in TASKS]))
