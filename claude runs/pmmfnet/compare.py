"""Comparison of the controlled PMMFNet experiments (plan.md) -> runs/comparison.md and .csv.

Run from claude runs/pmmfnet/ after run_all.sh:  ../../.venv/bin/python -u compare.py

All test metrics are leaf-level, out-of-fold (every leaf predicted once by the model of the fold
that held out its row). "Validation" columns come from the validation row of each fold and are
the only numbers used to rank configurations; test differences are reported as experimental
results with per-fold consistency and paired bootstrap intervals (resampling leaves).

Training noise: a fixed seed does not reproduce a run here (cudnn.benchmark picks kernels by
timing), so experiments whose settings are identical within a fold (same scaler values, shuffle,
objective, checkpoint rule and seeds) are reruns. Their per-fold range is reported as the rerun
noise band. Each paired comparison counts the folds where |B - A| is larger than the widest
rerun range seen in any fold (a conservative threshold: per-fold ranges from 2-3 reruns are too
narrow on their own).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
from four_class import CLASSES, decisions, fmt, score            # noqa: E402

RUNS = HERE / "runs"
SHORT = {"healthy": "healthy", "red_blotch_only": "RB only", "leafroll_only": "LR only", "both": "both"}
# (row label, run folder, file prefix, description)
ROWS = [
    ("Original", "original", "11_pmmfnet", "historical baseline: checkpoint = min val loss, scaler on train+val"),
    ("A1", "A1", "A1", "original + checkpoint = max val bal. acc."),
    ("A2", "A2", "A2", "A1 + scaler on training leaves only"),
    ("S1", "S1", "S1", "A2, no spatial shuffle"),
    ("S2", "S2", "S2", "A2, fixed shuffle also at validation/test"),
    ("S3", "S3", "S3", "A2 settings rerun (shuffle in training only)"),
    ("L1 (= A2 run)", "A2", "A2", "patch-level loss reference: same configuration and run as A2"),
    ("L2", "L2", "L2", "A2 with leaf-level (MIL) training loss"),
]
# (B, A, question) - B minus A
PAIRS = [
    ("A1", "Original", "checkpoint selection: max val bal. acc. vs min val loss"),
    ("A2", "A1", "scaler fitted on training leaves only vs training + validation leaves "
                 "(settings differ only in folds 0 and 4; folds 1-3 are reruns)"),
    ("S3", "A2", "identical settings rerun: run-to-run noise floor"),
    ("S1", "S3", "no shuffle vs shuffle in training only"),
    ("S1", "A2", "no shuffle vs shuffle in training only (other run of the reference)"),
    ("S2", "S3", "shuffle in training and evaluation vs training only"),
    ("S2", "A2", "shuffle in training and evaluation vs training only (other run of the reference)"),
    ("L2", "L1 (= A2 run)", "leaf-level vs patch-level training loss"),
    ("L2", "S3", "leaf-level vs patch-level training loss (other run of the reference)"),
]
ORIGINAL_RUNTIME_MIN = 35.25                  # 'real 35m15s' at the end of runs/original/logs/11.md


def load(folder, prefix):
    """One experiment's out-of-fold table (sorted by leaf), per-fold records and runtime in minutes."""
    oof = pd.read_csv(RUNS / folder / f"{prefix}_oof.csv")
    meta = json.loads((RUNS / folder / f"{prefix}_metrics.json").read_text())
    if folder == "original":
        folds = {int(k): {"best_val_balanced_accuracy": v["val_balanced_accuracy_at_best"],
                          "best_epoch": v["best_epoch"], "stopped_epoch": v["stopped_epoch"]}
                 for k, v in meta["early_stopping"].items()}
        runtime = ORIGINAL_RUNTIME_MIN
    else:
        folds = {int(k): v for k, v in meta["folds"].items()}
        runtime = meta["runtime_seconds"] / 60
        # the settings that actually reach the model in each fold (for finding reruns)
        shared = {k: v for k, v in meta["config"].items() if k not in ("experiment", "scaler")}
        for v in folds.values():
            v["setting"] = json.dumps([shared, v["scaler"]["min"], v["scaler"]["max"], v["seeds"]], sort_keys=True)
    return oof.sort_values("path").reset_index(drop=True), folds, runtime


def rerun_groups(runs):
    """{fold: [experiments with identical settings in that fold]} for groups of 2 or more."""
    groups = {}
    for label, (_, folds) in runs.items():
        if label.startswith("L1"):                                   # same run as A2, not a rerun
            continue
        for k, v in folds.items():
            if "setting" in v:
                groups.setdefault(k, {}).setdefault(v["setting"], []).append(label)
    return {k: [g for g in by.values() if len(g) > 1] for k, by in groups.items()}


def bal_acc(oof, idx=None):
    """4-class balanced accuracy, optionally on the leaves idx only."""
    o = oof if idx is None else oof.iloc[idx]
    return balanced_accuracy_score(o["class"], decisions(o))


def macro_f1(oof):
    """4-class macro-F1."""
    return f1_score(oof["class"], decisions(oof), labels=range(len(CLASSES)), average="macro", zero_division=0)


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """95% interval of bal.acc(b) - bal.acc(a), resampling leaves (same resample for both)."""
    rng = np.random.default_rng(seed)
    ya, pa, pb = a["class"].to_numpy(), decisions(a), decisions(b)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(ya), len(ya))
        diffs.append(balanced_accuracy_score(ya[i], pb[i]) - balanced_accuracy_score(ya[i], pa[i]))
    return np.percentile(diffs, [2.5, 97.5])


def main():
    """Write runs/comparison.md/.csv: main table, per-fold tables, rerun-noise table, paired comparisons."""
    runs, rows, fold_rows = {}, [], []
    for label, folder, prefix, desc in ROWS:
        if not (RUNS / folder / f"{prefix}_metrics.json").exists():
            print(f"missing {folder}, skipped")
            continue
        oof, folds, runtime = load(folder, prefix)
        runs[label] = (oof, folds)
        s = score(oof)
        m = s["four_class"]
        row = {"experiment": label, "description": desc,
               "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"]}
        for c in CLASSES:
            row[f"recall {SHORT[c]}"] = m["per_class"][c]["recall"]
        for c in CLASSES:
            row[f"precision {SHORT[c]}"] = m["per_class"][c]["precision"]
        row |= {"mean OvR AUC": m["auc_ovr_mean"]["pooled"], "mean within-variety AUC": m["auc_ovr_mean"]["within_variety"],
                "RB within-variety AUC": s["per_disease"]["red_blotch"]["auc_within_variety"],
                "LR within-variety AUC": s["per_disease"]["leafroll3"]["auc_within_variety"],
                "mean per-variety bal. acc.": np.mean([v["four_class"]["balanced_accuracy"]
                                                       for v in s["per_variety"].values()]),
                "runtime min": runtime,
                "mean best val bal. acc.": np.mean([f["best_val_balanced_accuracy"] for f in folds.values()]),
                "mean best epoch": np.mean([f["best_epoch"] for f in folds.values()])}
        rows.append(row)
        for k in sorted(folds):
            fold_rows.append({"experiment": label, "fold": k, "test bal. acc.": bal_acc(oof, np.where(oof.fold == k)[0]),
                              "best val bal. acc.": folds[k]["best_val_balanced_accuracy"],
                              "best epoch": folds[k]["best_epoch"], "stopped": folds[k]["stopped_epoch"]})
    table, per_fold = pd.DataFrame(rows), pd.DataFrame(fold_rows)
    table.to_csv(RUNS / "comparison.csv", index=False)
    per_fold.to_csv(RUNS / "comparison_folds.csv", index=False)

    num = [c for c in table.columns if c not in ("experiment", "description")]
    L = ["# Controlled PMMFNet experiments - comparison", "",
         "Leaf-level, out-of-fold test metrics (998 leaves, 5 row folds). Decisions = most probable class. "
         "'Original' is the historical run (checkpoint by minimum validation loss); every other row selects "
         "checkpoints by maximum validation balanced accuracy. L1 is the plan's patch-level reference for L2 "
         "and has exactly A2's configuration, so it is the A2 run; S3 is the same configuration rerun with the "
         "same seeds, which measures run-to-run noise.", "",
         "## Main table", "",
         "| experiment | " + " | ".join(num) + " |", "|---|" + "---|" * len(num)]
    for r in rows:
        L.append(f"| {r['experiment']} | " + " | ".join(f"{r[c]:.1f}" if c in ("runtime min", "mean best epoch")
                                                       else fmt(r[c]) for c in num) + " |")
    L += ["", "Descriptions: " + "; ".join(f"**{r['experiment']}** = {r['description']}" for r in rows), ""]

    labels = [r["experiment"] for r in rows]
    piv = per_fold.pivot(index="fold", columns="experiment", values="test bal. acc.")[labels]
    pv = per_fold.pivot(index="fold", columns="experiment", values="best val bal. acc.")[labels]
    L += ["## Per fold: test 4-class balanced accuracy", "", "| fold | " + " | ".join(labels) + " |",
          "|---|" + "---|" * len(labels)]
    L += [f"| {k} | " + " | ".join(fmt(piv.loc[k, e]) for e in labels) + " |" for k in piv.index]
    L += ["", "## Per fold: best validation 4-class balanced accuracy (the selection metric; Original = at its min-loss epoch)",
          "", "| fold | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    L += [f"| {k} | " + " | ".join(fmt(pv.loc[k, e]) for e in labels) + " |" for k in pv.index]

    groups = rerun_groups(runs)
    noise = 0.0                                     # widest test bal. acc. range among reruns
    L += ["", "## Rerun noise: experiments whose settings are identical in a fold", "",
          "Same scaler values, shuffle policy, objective, checkpoint rule and seeds, so the differences "
          "below are training noise only (GPU kernel selection). Two or three reruns understate the "
          "full spread.", "",
          "| fold | identical runs | test bal. acc. (min - max) | range | best val bal. acc. (min - max) |",
          "|---|---|---|---|---|"]
    for k in sorted(groups):
        for g in groups[k]:
            t, v = piv.loc[k, g], pv.loc[k, g]
            noise = max(noise, t.max() - t.min())
            L.append(f"| {k} | {', '.join(g)} | {fmt(t.min())} - {fmt(t.max())} | {t.max() - t.min():.3f} | "
                     f"{fmt(v.min())} - {fmt(v.max())} |")

    L += ["", "## Paired comparisons (B minus A, test leaves)", "",
          "The bootstrap resamples leaves with both trained models held fixed, so it does not include "
          f"training noise; the last column counts folds where B - A is beyond +/-{noise:.3f}, the widest "
          "rerun range above.", "",
          "| question | B vs A | delta bal. acc. | 95% bootstrap CI | folds where B > A | delta macro-F1 | "
          f"delta mean within-variety AUC | folds beyond +/-{noise:.3f}: up / down |",
          "|---|---|---|---|---|---|---|---|"]
    for b, a, q in PAIRS:
        if a not in runs or b not in runs:
            continue
        (oa, _), (ob, _) = runs[a], runs[b]
        assert (oa.path.to_numpy() == ob.path.to_numpy()).all()
        lo, hi = paired_bootstrap(oa, ob)
        wins = int((piv[b] > piv[a]).sum())
        ra, rb = table.set_index("experiment").loc[a], table.set_index("experiment").loc[b]
        d = piv[b] - piv[a]
        beyond = f"{int((d > noise).sum())} / {int((d < -noise).sum())}"
        L.append(f"| {q} | {b} vs {a} | {bal_acc(ob) - bal_acc(oa):+.3f} | [{lo:+.3f}, {hi:+.3f}] | {wins}/5 | "
                 f"{macro_f1(ob) - macro_f1(oa):+.3f} | {rb['mean within-variety AUC'] - ra['mean within-variety AUC']:+.3f} | "
                 f"{beyond} |")
    md = "\n".join(L) + "\n"
    (RUNS / "comparison.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
