"""Within-variety selection: load every candidate's predictions, score them on validation, and score a whole
search honestly (pick on validation, score the held-out test row). report.py calls these.
"""
import numpy as np
from harness import PRED, TASKS, VARIETIES, bal_acc


def load_preds() -> dict:
    """{candidate: {(task, variety): {"val","test","y","row"}}}; within-variety CNN pieces are merged."""
    out = {}
    for f in sorted(PRED.glob("*.npz")):
        name = f.stem
        if "|" in name:                          # within|<variety> piece of a CNN
            name = name.split("|")[0]
        d = np.load(f)
        c = out.setdefault(name, {})
        for k in d.files:
            if k == "params":
                continue
            task, v, part = k.split("|")
            c.setdefault((task, v), {})[part] = d[k]
    out = {n: c for n, c in out.items() if all((t, v) in c for t in TASKS for v in VARIETIES)}
    ref = next(iter(out.values()))
    for n, c in out.items():   # every candidate must cover the same leaves, in the same order
        for key in ref:
            assert np.array_equal(c[key]["y"], ref[key]["y"]) and np.array_equal(c[key]["row"], ref[key]["row"]), (n, key)
    return out


def val_score(c: dict, task: str, exclude_row: int | None = None) -> float:
    """Mean over varieties of validation balanced accuracy, optionally leaving out one row's leaves."""
    s = []
    for v in VARIETIES:
        d = c[(task, v)]
        m = d["row"] != exclude_row if exclude_row is not None else np.ones(len(d["y"]), bool)
        s.append(bal_acc(d["val"][m], d["y"][m]))
    return float(np.mean(s))


def honest(preds: dict, task: str) -> tuple[dict, list]:
    """For each test row k pick the candidate on validation leaves outside row k; score row k with it."""
    picks, test = [], {v: None for v in VARIETIES}
    assembled = {v: np.full(len(next(iter(preds.values()))[(task, v)]["y"]), np.nan) for v in VARIETIES}
    for k in range(5):
        best = max(preds, key=lambda n: val_score(preds[n], task, exclude_row=k))
        picks.append(best)
        for v in VARIETIES:
            d = preds[best][(task, v)]
            assembled[v][d["row"] == k] = d["test"][d["row"] == k]
    for v in VARIETIES:
        y = next(iter(preds.values()))[(task, v)]["y"]
        test[v] = bal_acc(assembled[v], y)
    return test, picks


def strict(preds: dict, task: str) -> tuple[dict, list]:
    """Standard nested pick: for test row k use only fold k's own validation row (k+1), whose model never saw
    row k. Noisier than honest() (one row) but no other fold's training touches row k."""
    picks = []
    assembled = {v: np.full(len(next(iter(preds.values()))[(task, v)]["y"]), np.nan) for v in VARIETIES}
    for k in range(5):
        def sc(n):
            s = []
            for v in VARIETIES:
                d = preds[n][(task, v)]
                m = d["row"] == (k + 1) % 5
                s.append(bal_acc(d["val"][m], d["y"][m]))
            return float(np.mean(s))
        best = max(preds, key=sc)
        picks.append(best)
        for v in VARIETIES:
            d = preds[best][(task, v)]
            assembled[v][d["row"] == k] = d["test"][d["row"] == k]
    return {v: bal_acc(assembled[v], next(iter(preds.values()))[(task, v)]["y"]) for v in VARIETIES}, picks


def honest_ensemble(preds: dict, task: str, k_top: int = 5) -> dict:
    """For each test row r: average the k_top candidates ranked best on validation leaves outside row r."""
    out = {}
    for v in VARIETIES:
        y = next(iter(preds.values()))[(task, v)]["y"]
        out[v] = np.full(len(y), np.nan)
    for r in range(5):
        top = sorted(preds, key=lambda n: -val_score(preds[n], task, exclude_row=r))[:k_top]
        for v in VARIETIES:
            rows = preds[top[0]][(task, v)]["row"] == r
            out[v][rows] = np.mean([preds[n][(task, v)]["test"][rows] for n in top], 0)
    return {v: bal_acc(out[v], next(iter(preds.values()))[(task, v)]["y"]) for v in VARIETIES}
