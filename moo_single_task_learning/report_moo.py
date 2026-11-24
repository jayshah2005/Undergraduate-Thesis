"""Score the six-head CNN (LS and MGDA) the way the variety and disease CNNs are scored, and compare.

- Variety: the 4 variety heads; a leaf is called its most likely variety. Scored with the variety
  report's own code (variety_classification/cnn/report.py), next to logistic regression and the
  variety CNNs, with the same bootstrap.
- Disease: head 5 (leafroll, 748 leaves without Cab Franc) and head 4 (red blotch, all 998), each
  called positive at P >= 0.5. Scored with the pooled disease report's own code
  (disease_classification/cnn/report.py) on the `all` pool, next to logistic regression and the
  pooled disease CNNs, with the same resamples (so every difference is paired).

The variety and disease report folders both have modules named report, scoring and model, so
each part runs in its own process (``--part``) and writes JSON; the default run calls both and
writes results/final_report.md. Nothing in the other folders is changed.

Usage (from this folder)::

    ../.venv/bin/python -u report_moo.py                       # results/ -> results/final_report.md
    ../.venv/bin/python -u report_moo.py --results results/smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METHODS = {"ls": "six-head, LS", "mgda": "six-head, MGDA", "mgda_gn": "six-head, MGDA-GN",
           "mgda_ln": "six-head, MGDA-LN", "mgda_lgn": "six-head, MGDA-LGN",
           "svdpga": "six-head, SVD-PGA", "osnmfpga": "six-head, OSNMF-PGA"}
VARIETIES = ["Baco", "Cab Franc", "Chardonnay", "Vidal"]   # head order in data.py
HEADS = VARIETIES + ["red blotch", "leafroll"]
DISEASE_HEAD = {"red_blotch": 4, "leafroll": 5}
NO_CF = "red_blotch_no_cab_franc"   # red blotch scored on Baco, Chardonnay and Vidal only (same trained models)
MAX_EPOCHS = 40   # TrainConfig.max_epochs in train.py
REFERENCE = "CNN, attention pool"   # the closest single-task model: small CNN, same pooling as the six-head CNN


# --------------------------------------------------------------------------
# Six-head predictions
# --------------------------------------------------------------------------
def load_six_head(results: Path, method: str, paths: list[str]) -> dict:
    """Collect one method's test probabilities, reordered to a leaf table's paths.

    S: seeds on disk. n: leaves in the table.

    :param results: Results folder of run_moo.py.
    :param method: "ls" or "mgda".
    :param paths: Leaf paths ("Variety/Row/Vine") in the order of the table to score against.
    :return: prob (S, n, 6), y (n, 6) labels saved by the run, best_epochs and epochs_run per training,
        alpha (epochs, 6) stacked over trainings.
    """
    pos = {p: i for i, p in enumerate(paths)}
    seed_dirs = sorted((results / method).glob("seed*"))
    prob = np.full((len(seed_dirs), len(paths), 6), np.nan)
    y = np.full((len(paths), 6), np.nan)
    epochs, ran, alphas = [], [], []
    for s, d in enumerate(seed_dirs):
        for k in range(5):
            f = np.load(d / f"fold{k}.npz")
            idx = np.array([pos[str(p)] for p in f["paths"]])
            prob[s, idx], y[idx] = f["prob"], f["y"]
            epochs.append(int(f["best_epoch"]))
            ran.append(int(f["epochs_run"]))
            alphas.append(f["alpha"])
    assert len(seed_dirs) and not np.isnan(prob).any(), f"{results / method}: missing folds"
    return {"prob": prob, "y": y, "best_epochs": epochs, "epochs_run": ran, "alpha": np.concatenate(alphas)}


def methods_on_disk(results: Path, only: list[str] | None = None) -> dict[str, str]:
    """The methods with runs in results (or only those listed), in METHODS order: {method: report label}."""
    return {m: label for m, label in METHODS.items() if (results / m).exists() and (not only or m in only)}


def pairs_of(methods: dict[str, str]) -> list[tuple[str, str]]:
    """Paired comparisons: each other method against LS, then each method against the single-task CNN."""
    return ([(label, methods["ls"]) for m, label in methods.items() if m != "ls" and "ls" in methods]
            + [(label, REFERENCE) for label in methods.values()])


def sys_path_for(folder: Path) -> None:
    """Import from one report folder only: drop this folder (its train.py, model.py) from the path."""
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
    sys.path.insert(0, str(folder))


# --------------------------------------------------------------------------
# Variety part
# --------------------------------------------------------------------------
def score_variety(results: Path, only: list[str] | None = None) -> dict:
    """Variety balanced accuracy of each six-head method, the variety CNNs and logistic regression."""
    sys_path_for(ROOT / "variety_classification" / "cnn")
    import report as VR   # variety_classification/cnn/report.py

    device = VR.get_device()
    d = VR.load_features(VR.LRConfig("all"), device)
    y, rows, paths = d["y"], d["rows"], list(d["meta"].path)
    ref = VR.compute(device)   # logistic regression and the variety CNNs, both band sets
    lr = ref["lr"]["all"]

    def summary(m: dict) -> dict:
        return {"bal_acc": m["balanced_accuracy"], "ci95": list(m["balanced_accuracy_ci95"]),
                "per_row": {r: float(v) for r, v in m["per_row"].items()},
                "recall": [float(v) for v in m["recall"]], "seed_scores": m.get("seed_scores")}

    out = {"models": {VR.BASELINE: summary(lr)}, "vs_lr": {}, "pairs": {}}
    preds = {VR.BASELINE: lr["pred"]}
    for name, per_band in ref["cnns"].items():
        out["models"][name] = summary(per_band["all"])
        out["vs_lr"][name] = ref["vs_lr"][name]["all"]
        preds[name] = per_band["all"]["pred"]
    methods = methods_on_disk(results, only)
    for method, label in methods.items():
        six = load_six_head(results, method, paths)
        assert (six["y"][:, :4].argmax(1) == y).all(), "variety labels differ from the leaf table"
        p4 = six["prob"][..., :4]
        p4 = p4 / p4.sum(-1, keepdims=True)   # 4 sigmoids → one distribution; argmax (the score) is unchanged
        m = VR.evaluate(y, p4.mean(0), device)
        m["seed_scores"] = [VR.evaluate(y, p, device)["balanced_accuracy"] for p in p4]
        m["per_row"] = {r: VR.balanced_accuracy_score(y[rows == r], m["pred"][rows == r]) for r in VR.ROWS}
        out["models"][label] = summary(m)
        out["vs_lr"][label] = VR.paired_delta(y, m["pred"], lr["pred"], device)
        preds[label] = m["pred"]
    out["pairs"] = {f"{a} − {b}": VR.paired_delta(y, preds[a], preds[b], device) for a, b in pairs_of(methods)}
    return out


# --------------------------------------------------------------------------
# Disease part
# --------------------------------------------------------------------------
def score_disease(results: Path, only: list[str] | None = None) -> dict:
    """Leafroll and red blotch of each six-head method, the pooled disease CNNs and logistic regression."""
    sys_path_for(ROOT / "disease_classification" / "cnn")
    import report as DR   # disease_classification/cnn/report.py
    import torch
    from run_experiments import results_root

    device = DR.get_device()
    meta = DR.load_meta("all")
    root = results_root(False, variety="all", bands="all")
    others = [("small_resnet", root.parent / f"{root.name}_small_resnet")]
    preds = DR.load_predictions(meta, root, "all", "all", device, others)
    preds = {task: preds[task] for task in DISEASE_HEAD}
    for t in preds.values():   # mean and attention pooling only, as in the variety comparison
        for label in [lab for lab, (_bb, arm) in t["arms"].items() if arm not in ("mean", "attention")]:
            for key in ("cnn", "epochs", "arms"):
                t[key].pop(label)
    methods = methods_on_disk(results, only)
    six = {m: load_six_head(results, m, list(meta.path)) for m in methods}
    for task, head in DISEASE_HEAD.items():
        t = preds[task]
        for method, label in methods.items():
            s = six[method]
            assert np.array_equal(s["y"][t["leaf_idx"], head], t["y"]), f"{task} labels differ from the leaf table"
            t["cnn"][label] = s["prob"][:, t["leaf_idx"], head]
            t["epochs"][label] = s["best_epochs"]
    # Cab Franc is 241 of 250 red blotch: score the same predictions without its leaves as well.
    rb = preds["red_blotch"]
    keep = rb["variety"] != "Cab Franc"
    preds[NO_CF] = {**{k: rb[k][keep] for k in ("y", "fold", "group", "variety", "leaf_idx", "base")},
                    "cnn": {k: v[:, keep] for k, v in rb["cnn"].items()}, "att": {},
                    "epochs": rb["epochs"], "arms": rb["arms"]}
    metrics = DR.compute_metrics(preds, device)

    out = {}
    for task, t in preds.items():
        # Paired differences between CNNs on the report's own resamples (compute_metrics pairs only with the baseline).
        tt = lambda a: torch.tensor(a, dtype=torch.float64, device=device)  # noqa: E731
        y = tt(t["y"])
        w = DR.bootstrap_weights(y, torch.tensor(t["group"], device=device))
        q = torch.tensor([0.025, 0.975], dtype=torch.float64, device=device)

        def diff(a: str, b: str) -> list[float]:
            pa, pb = tt(t["cnn"][a]).mean(0), tt(t["cnn"][b]).mean(0)
            d = DR.balanced_accuracy(pa, y, w) - DR.balanced_accuracy(pb, y, w)
            return [(DR.balanced_accuracy(pa, y) - DR.balanced_accuracy(pb, y)).item(), *torch.quantile(d, q).tolist()]

        res = metrics[task]
        out[task] = {
            "n": len(t["y"]), "positives": int(t["y"].sum()),
            "models": {name: {"bal_acc": r["bal_acc"], "ci95": r["ci95"],
                              "vs_lr": [r["diff_vs_base"], *r["diff_ci95"]] if name != "baseline" else None,
                              "per_variety": r["per_variety"], "per_fold": r["per_fold"], "seeds": r["seeds"],
                              "median_best_epoch": r.get("median_best_epoch")}
                       for name, r in res.items()},
            "pairs": {f"{a} − {b}": diff(a, b) for a, b in pairs_of(methods)},
        }
    # Each epoch's mean weights over the sum of their absolute values: MGDA's α (on the simplex) are unchanged;
    # PGA's are not normalized and can be negative.
    share = {m: six[m]["alpha"] / np.abs(six[m]["alpha"]).sum(1, keepdims=True)
             for m in methods if len(six[m]["alpha"])}
    out["alpha"] = {m: a.mean(0).tolist() for m, a in share.items()}
    out["alpha_range"] = {m: [a.min(0).tolist(), a.max(0).tolist()] for m, a in share.items()}
    out["best_epochs"] = {m: float(np.median(six[m]["best_epochs"])) for m in methods}
    out["n_trainings"] = {m: len(six[m]["best_epochs"]) for m in methods}
    out["hit_epoch_limit"] = {m: int(sum(r == MAX_EPOCHS for r in six[m]["epochs_run"])) for m in methods}
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def ci(v: list[float]) -> str:
    """'0.123 [0.100, 0.150]'."""
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def dci(v: list[float]) -> str:
    """'+0.012 [-0.010, +0.030]', with * when the interval excludes 0."""
    star = " *" if v[1] > 0 or v[2] < 0 else ""
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]{star}"


def short(label: str) -> str:
    """'six-head, MGDA-GN' -> 'MGDA-GN'."""
    return label.removeprefix("six-head, ")


TASK_COLUMNS = (("leafroll", "leafroll"), ("red_blotch", "red blotch, all 998"),
                (NO_CF, "red blotch, without Cab Franc"))
DETAIL_TITLE = {"leafroll": "Leafroll", "red_blotch": "Red blotch, all leaves", NO_CF: "Red blotch without Cab Franc"}


def to_markdown(var: dict, dis: dict) -> str:
    """final_report.md: a summary table, the paired differences, then the details. Balanced accuracy only."""
    LR = "logistic regression"
    disease = {task: {LR if name == "baseline" else name: m for name, m in dis[task]["models"].items()}
               for task, _ in TASK_COLUMNS}
    models = list(var["models"])   # logistic regression, the single-task CNNs, then the six-head methods
    six = [m for m in models if m.startswith("six-head")]
    rb, nocf, lr_ = dis["red_blotch"], dis[NO_CF], dis["leafroll"]

    L = ["# Six-head CNN: LS, MGDA and PGA against the single-task models", "",
         "One CNN with six sigmoid heads (Baco, Cab Franc, Chardonnay, Vidal, red blotch, leafroll), 998 leaves, "
         "204 bands. The methods differ only in how the six heads' gradients become one update:",
         "",
         "- **LS**: their sum (one backward pass on the summed loss).",
         "- **MGDA** (Sener & Koltun 2018): the shortest mix of the six gradients, over all weights. **-GN / -LN / "
         "-LGN** first divide each gradient by its norm, its loss, or both (the variants in LibMTL, Lin & Zhang 2023).",
         "- **SVD-PGA / OSNMF-PGA**: the Principal Gradient Alignment aggregators of the local torchjd (commit "
         "04d1bef, not yet published), default settings.",
         "",
         "Same row folds (test Rk, validation R(k+1), train on the other 3), seeds and 3-seed ensemble as the variety "
         "and disease CNNs, scored with their own report code. Rule, set before the runs: a difference counts only "
         "where its paired 95% interval excludes 0 (marked *).", ""]

    # 1. Summary
    L += ["## Summary", "",
          "Balanced accuracy (chance: variety 0.25, disease 0.5). Leafroll never includes Cab Franc (it has too few "
          f"leafroll leaves): {lr_['n']} leaves. Red blotch is shown on all 998 leaves and without Cab Franc "
          f"({nocf['n']} leaves): 241 of Cab Franc's 250 leaves have red blotch, so a model gets them right from the "
          "variety alone. Both red-blotch columns score the same trained models.", "",
          "| model | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |",
          "|---|---|---|---|---|"]
    for name in models:
        label = name if name.startswith("six-head") or name == LR else f"{name} (single-task)"
        cells = [f"{var['models'][name]['bal_acc']:.3f}"]
        cells += [f"{disease[task][name]['bal_acc']:.3f}" for task, _ in TASK_COLUMNS]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    # 2. Paired differences
    def pair_table(title: str, intro: str, keys: list[tuple[str, str]]) -> list[str]:
        rows = [f"## {title}", "", intro, "",
                "| comparison | variety | leafroll | red blotch, all 998 | red blotch, without Cab Franc |",
                "|---|---|---|---|---|"]
        for a, b in keys:
            key = f"{a} − {b}"
            if key not in var["pairs"]:
                continue
            cells = [dci(var["pairs"][key])] + [dci(dis[task]["pairs"][key]) for task, _ in TASK_COLUMNS]
            rows.append(f"| {short(a)} − {short(b)} | " + " | ".join(cells) + " |")
        return rows + [""]

    ls = METHODS["ls"]
    L += pair_table("Each method against LS",
                    "Paired difference in balanced accuracy [95% CI], same resampled leaves; * = interval excludes 0.",
                    [(m, ls) for m in six if m != ls])
    L += pair_table(f"Six-head against the single-task {REFERENCE}",
                    f"The {REFERENCE} is the closest single-task model (same pooling). Paired, as above.",
                    [(m, REFERENCE) for m in six])

    # 3. Weights
    if dis["alpha"]:
        L += ["## Weight per head", "",
              "Each epoch's mean weights divided by the sum of their absolute values, averaged over every epoch of "
              "a method's 15 trainings (range over epochs). For MGDA these are α; a PGA weight can be negative.", "",
              "| method | " + " | ".join(HEADS) + " |", "|---|" + "---|" * len(HEADS)]
        for m, a in dis["alpha"].items():
            lo, hi = dis["alpha_range"][m]
            L.append(f"| {short(METHODS[m])} | " + " | ".join(f"{x:.2f} ({l:.2f}–{h:.2f})" for x, l, h in zip(a, lo, hi))
                     + " |")
        L.append("")

    # 4. Details
    L += ["## Details", "", "### Variety, per test row", "",
          "| model | bal. acc. [95% CI] | Δ vs logistic regression | R1 | R2 | R3 | R4 | R5 |", "|---|---|---|---|---|---|---|---|"]
    for name, m in var["models"].items():
        vs = dci(var["vs_lr"][name]) if name in var["vs_lr"] else "—"
        rows = " | ".join(f"{m['per_row'][r]:.3f}" for r in ("R1", "R2", "R3", "R4", "R5"))
        L.append(f"| {name} | {ci([m['bal_acc'], *m['ci95']])} | {vs} | {rows} |")
    L.append("")
    for task, _ in TASK_COLUMNS:
        d = dis[task]
        vs = list(next(iter(d["models"].values()))["per_variety"])
        L += [f"### {DETAIL_TITLE[task]} ({d['n']} leaves, {d['positives']} with the disease), per variety", "",
              "| model | bal. acc. [95% CI] | Δ vs logistic regression | " + " | ".join(vs) + " |",
              "|---|---|---|" + "---|" * len(vs)]
        for name, m in disease[task].items():
            vsl = dci(m["vs_lr"]) if m["vs_lr"] else "—"
            L.append(f"| {name} | {ci([m['bal_acc'], *m['ci95']])} | {vsl} | "
                     + " | ".join(f"{m['per_variety'][v]:.3f}" for v in vs) + " |")
        L.append("")

    # 5. Notes
    L += ["## Notes", "",
          "- Median best epoch: " + ", ".join(f"{short(METHODS[m])} {e:g}" for m, e in dis["best_epochs"].items())
          + f". Trainings that reached the {MAX_EPOCHS}-epoch limit (early stopping never triggered): "
          + ", ".join(f"{short(METHODS[m])} {n} of {dis['n_trainings'][m]}" for m, n in dis["hit_epoch_limit"].items())
          + ".",
          "- The single-task models in the red blotch columns trained on all 998 leaves. Trained on Baco, Chardonnay "
          "and Vidal only, the disease CNN scores 0.619 (mean pool) and 0.606 (attention pool) and logistic regression "
          "0.600 (`disease_classification/REPORT.md`).",
          "- The six-head CNN is not the same network as the single-task CNNs: two 3 × 3 convolutions at full "
          "resolution and no batch norm, no rotations or flips, a 211 px canvas, one checkpoint for all six heads "
          "(lowest summed validation loss), and its leafroll and red blotch heads also train on Cab Franc. So the "
          "clean comparison is between the six-head methods; against the single-task models the network differs too.",
          ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--part", choices=("variety", "disease"), help="(internal) score one part, write JSON")
    ap.add_argument("--methods", nargs="+", choices=list(METHODS), help="score only these (default: all on disk)")
    args = ap.parse_args()
    results = args.results.resolve()
    if args.part:
        score = score_variety if args.part == "variety" else score_disease
        (results / f"scores_{args.part}.json").write_text(json.dumps(score(results, args.methods), indent=1,
                                                                     default=float))
        return
    for part in ("variety", "disease"):
        only = ["--methods", *args.methods] if args.methods else []
        subprocess.run([sys.executable, "-u", __file__, "--results", str(results), "--part", part, *only], check=True)
    var = json.loads((results / "scores_variety.json").read_text())
    dis = json.loads((results / "scores_disease.json").read_text())
    (results / "final_report.md").write_text(to_markdown(var, dis))
    print(f"[saved] {results / 'final_report.md'}")


if __name__ == "__main__":
    main()
