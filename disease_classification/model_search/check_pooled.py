"""Check: in pooled mode a variety-only feature carries no information inside a variety (BA exactly 0.5)."""
import numpy as np
from harness import Candidate, evaluate, load_all, bal_acc
from family_spectral import lr
data = load_all()
for k in ("mean_refl", "snv_q", "snv_sd", "idx_stats"):
    assert np.isfinite(data["F"][k]).all(), k
codes = {v: i for i, v in enumerate(sorted(set(data["variety"])))}
def f_var(F, idx):
    onehot = np.zeros((len(idx), 4)); onehot[np.arange(len(idx)), [codes[v] for v in data["variety"][idx]]] = 1
    return onehot + 1e-3 * np.random.default_rng(0).normal(size=onehot.shape) * 0   # exactly the variety
res = evaluate(Candidate("variety_only", f_var, lr, [{"C": 1.0}], "proba"), data, "pooled")
for key, d in res["preds"].items():
    print(key, f"val {bal_acc(d['val'], d['y']):.3f}", f"P range {d['val'].min():.3f}-{d['val'].max():.3f}")
