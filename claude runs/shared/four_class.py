"""The 4-class formulation and its metrics, shared by the paper pipelines (11, 12, ...).

USE   md = save_results(name, title, meta, P, config)   # P: (n_leaves, 4) class probabilities
      metrics = score(oof); md = to_markdown(title, metrics, config)
      (compare.py re-scores saved OOF tables with score / fmt.)

CLASSES  The two labels become 4 mutually exclusive classes, class = red_blotch + 2 * leafroll3:
      0 healthy    1 red blotch only    2 leafroll only    3 both
  so P(red blotch) = P(1) + P(3), P(leafroll) = P(2) + P(3), P(healthy) = P(0).
  Why: the paper models are single-label (one class per leaf) classifiers, but a leaf can carry both
  diseases; the bit encoding (bit 0 = RB, bit 1 = LR) keeps every combination as its own class
  and lets the per-disease scores be read back by summing class probabilities. All 4 classes
  are used for every variety; only the derived *leafroll* metrics drop Cab Franc.

METRICS  (written by save_results to results/<name>_metrics.md and .json)
  * 4 classes: confusion matrix of the argmax class, balanced accuracy (mean recall over the
    classes), one-vs-rest ROC AUC per class and their mean.
  * Per disease (red blotch; leafroll without Cab Franc, which has 1 positive; any disease =
    not healthy): 2x2 confusion matrix and balanced accuracy from the argmax class, ROC AUC
    from the summed class probabilities.
  * ROC AUC pooled over all leaves and within-variety (leaves compared only with leaves of the
    same variety and fold - the honest number, see common.py).
  * The same per variety.

Everything here computes and returns dicts except the two reporting functions: to_markdown
(builds the report text) and save_results (writes the files and one "[saved]" line, printed or
sent to `log=`). Metrics use sklearn.metrics (exact, well-tested) and common._auc; no torch.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from common import (NO_LEAFROLL_VARIETY, RESULTS_DIR, VARIETIES, _auc, _stratified_auc,
                    write_json)

CLASSES = ["healthy", "red_blotch_only", "leafroll_only", "both"]
PROB_COLS = [f"p_{c}" for c in CLASSES]

# Per variety, the model often predicts a class that variety has no leaf of (e.g. "both" in a variety
# without double infections); balanced_accuracy_score then warns on every call although the value is
# well defined (mean recall over the true classes), so silence that one warning.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def leaf_classes(meta: pd.DataFrame) -> np.ndarray:
    """(N,) int class per leaf in 0..3: red_blotch + 2 * leafroll3 (see CLASSES)."""
    return (meta.red_blotch + 2 * meta.leafroll3).to_numpy()


def _bacc(y, pred):
    """Balanced accuracy (mean recall over the classes present in y); NaN when y holds a single
    class, where it would only measure that one class's recall."""
    return balanced_accuracy_score(y, pred) if len(np.unique(y)) > 1 else np.nan


def multiclass_scores(y, P, strata, pred=None) -> dict:
    """4-class metrics: confusion matrix, balanced accuracy, one-vs-rest AUC (pooled and
    within strata = variety|fold), precision / recall / F1.

    y: (n,) true class 0..3; P: (n, 4) class probabilities; strata: (n,) "variety|fold" labels;
    pred: (n,) predicted class (default: argmax of P). Returns a JSON-ready dict.
    """
    pred = P.argmax(1) if pred is None else pred
    auc = {c: {"pooled": _auc((y == k).astype(int), P[:, k]),
               "within_variety": _stratified_auc((y == k).astype(int), P[:, k], strata)}
           for k, c in enumerate(CLASSES)}
    return {"n": int(len(y)), "class_counts": {c: int((y == k).sum()) for k, c in enumerate(CLASSES)},
            "confusion": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist(),
            "balanced_accuracy": _bacc(y, pred), "auc_ovr": auc,
            "auc_ovr_mean": {w: np.nanmean([a[w] for a in auc.values()])
                             for w in ("pooled", "within_variety")},
            **precision_recall(y, pred)}


def precision_recall(y, pred) -> dict:
    """Per-class precision / recall / F1 (0 when a class is never predicted) and macro-F1
    over the classes present in y."""
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)),
                                                         zero_division=0)
    present = [k for k in range(len(CLASSES)) if (y == k).any()]
    return {"macro_f1": f1_score(y, pred, labels=present, average="macro", zero_division=0)
            if present else np.nan,
            "per_class": {c: {"precision": float(prec[k]), "recall": float(rec[k]), "f1": float(f1[k]),
                              "support": int(sup[k])} for k, c in enumerate(CLASSES)}}


def binary_scores(y, s, pred, strata) -> dict:
    """Metrics of one yes/no target: 2x2 confusion, balanced accuracy, AUC pooled and within strata.

    y: (n,) 0/1; s: (n,) score; pred: (n,) bool/0-1 decision; strata: (n,) "variety|fold" labels."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "positives": int(y.sum()),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "balanced_accuracy": _bacc(y, pred),
            "auc_pooled": _auc(y, s), "auc_within_variety": _stratified_auc(y, s, strata)}


def decisions(oof: pd.DataFrame) -> np.ndarray:
    """Predicted class per leaf: the classifier's own decision ('pred' column) if the pipeline
    saved one, otherwise the most probable class."""
    return oof["pred"].to_numpy() if "pred" in oof else oof[PROB_COLS].to_numpy().argmax(1)


def disease_views(oof: pd.DataFrame):
    """(name, keep, y, score, predicted) per disease, all derived from the 4-class output.

    keep: bool mask of leaves scored for that disease (leafroll drops Cab Franc: 1 positive leaf,
    so its AUC / balanced accuracy would be noise). score = summed class probabilities;
    predicted = the decided class contains the disease."""
    P = oof[PROB_COLS].to_numpy()
    pred = decisions(oof)
    every = np.ones(len(oof), dtype=bool)
    return [
        ("red_blotch", every, oof.red_blotch.to_numpy(), P[:, 1] + P[:, 3], np.isin(pred, [1, 3])),
        ("leafroll3", (oof.variety != NO_LEAFROLL_VARIETY).to_numpy(), oof.leafroll3.to_numpy(),
         P[:, 2] + P[:, 3], np.isin(pred, [2, 3])),
        ("any_disease", every, (oof["class"] != 0).to_numpy().astype(int), 1 - P[:, 0], pred != 0),
    ]


def score(oof: pd.DataFrame) -> dict:
    """Every metric of an out-of-fold table: 4-class, per disease (derived) and per variety.

    oof: one row per leaf with columns class, variety, fold, red_blotch, leafroll3, p_healthy ..
    p_both and optionally pred. Returns {"four_class", "per_disease", "per_variety"} (JSON-ready).
    """
    y, P = oof["class"].to_numpy(), oof[PROB_COLS].to_numpy()
    var = oof.variety.to_numpy()
    strata = (oof.variety.astype(str) + "|" + oof.fold.astype(str)).to_numpy()
    pred4 = decisions(oof)
    out = {"four_class": multiclass_scores(y, P, strata, pred4), "per_disease": {}, "per_variety": {}}
    for name, keep, yd, s, pred in disease_views(oof):
        out["per_disease"][name] = binary_scores(yd[keep], s[keep], pred[keep], strata[keep])
    for v in VARIETIES:
        m = var == v
        out["per_variety"][v] = {
            "four_class": multiclass_scores(y[m], P[m], strata[m], pred4[m]),
            "per_disease": {name: binary_scores(yd[m & keep], s[m & keep], pred[m & keep], strata[m & keep])
                            for name, keep, yd, s, pred in disease_views(oof) if (m & keep).any()}}
    return out


def fmt(v) -> str:
    """3-decimal string; 'n/a' for a missing value."""
    return "n/a" if v is None or np.isnan(v) else f"{v:.3f}"


DEFAULT_DECISION_NOTE = "Decisions = the most probable of the 4 classes."


def to_markdown(title: str, metrics: dict, config: dict, extra: str = "",
                decision_note: str = DEFAULT_DECISION_NOTE, extended: bool = False) -> str:
    """REPORTING: the markdown report of score()'s output; `extra` (markdown) is appended, e.g. a
    pipeline's early-stopping table. extended=True adds macro-F1 and per-class precision/recall."""
    f = fmt
    short = ["healthy", "RB only", "LR only", "both"]
    mc = metrics["four_class"]
    L = [f"# {title}", "",
         "Settings: " + ", ".join(f"{k}={v}" for k, v in config.items()), "",
         "Every leaf is predicted by the model of the fold that held out its row. "
         + decision_note + " 'Within-variety' AUC compares leaves only with "
         "leaves of the same variety and fold (0.5 = chance); pooled AUC can be inflated by "
         "recognising the variety. Leafroll excludes Cab Franc (1 positive leaf).", "",
         "## 4 classes", "",
         f"Balanced accuracy: **{f(mc['balanced_accuracy'])}** (chance = 0.25)", "",
         "Confusion matrix (rows = true class, columns = predicted class):", "",
         "| true \\ predicted | " + " | ".join(short) + " | total |",
         "|---|" + "---|" * (len(short) + 1)]
    for k, row in enumerate(mc["confusion"]):
        L.append(f"| **{short[k]}** | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    if extended:                                  # added for the controlled PMMFNet experiments
        L += ["", f"Macro-F1: **{f(mc['macro_f1'])}**", "",
              "| class | precision | recall | F1 | n |", "|---|---|---|---|---|"]
        for k, c in enumerate(CLASSES):
            pc = mc["per_class"][c]
            L.append(f"| {short[k]} | {f(pc['precision'])} | {f(pc['recall'])} | {f(pc['f1'])} | {pc['support']} |")
    L += ["", "| class (one vs rest) | n | ROC AUC pooled | ROC AUC within-variety |", "|---|---|---|---|"]
    for k, c in enumerate(CLASSES):
        a = mc["auc_ovr"][c]
        L.append(f"| {short[k]} | {mc['class_counts'][c]} | {f(a['pooled'])} | {f(a['within_variety'])} |")
    L.append(f"| **mean** | | {f(mc['auc_ovr_mean']['pooled'])} | {f(mc['auc_ovr_mean']['within_variety'])} |")

    L += ["", "## Per disease (derived from the 4-class output)", "",
          "| target | n (pos) | TN | FP | FN | TP | balanced acc. | ROC AUC pooled | ROC AUC within-variety |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, r in metrics["per_disease"].items():
        c = r["confusion"]
        L.append(f"| {name} | {r['n']} ({r['positives']}) | {c['tn']} | {c['fp']} | {c['fn']} | "
                 f"{c['tp']} | {f(r['balanced_accuracy'])} | {f(r['auc_pooled'])} | "
                 f"{f(r['auc_within_variety'])} |")

    L += ["", "## Per variety", "",
          "AUC here = within the variety, compared within folds. n/a = only one class present.", "",
          "| variety | n | 4-class bal. acc. | mean OvR AUC | RB bal. acc. | RB AUC | "
          "LR bal. acc. | LR AUC | any-disease bal. acc. | any-disease AUC |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in metrics["per_variety"].items():
        d = r["per_disease"]
        cells = []
        for name in ("red_blotch", "leafroll3", "any_disease"):
            cells += [f(d[name]["balanced_accuracy"]), f(d[name]["auc_within_variety"])] if name in d \
                else ["n/a", "n/a"]
        L.append(f"| {v} | {r['four_class']['n']} | {f(r['four_class']['balanced_accuracy'])} | "
                 f"{f(r['four_class']['auc_ovr_mean']['within_variety'])} | " + " | ".join(cells) + " |")
    L += ["", "4-class confusion matrices per variety (rows = true, columns = predicted: "
          + " / ".join(short) + "):", ""]
    for v, r in metrics["per_variety"].items():
        L.append(f"- **{v}**: " + "; ".join(f"{short[k]} -> {row}"
                                             for k, row in enumerate(r["four_class"]["confusion"])))
    return "\n".join(L) + "\n" + extra


def save_results(name: str, title: str, meta: pd.DataFrame, P: np.ndarray, config: dict,
                 quick: bool = False, extra_md: str = "", extra_json: dict | None = None,
                 pred: np.ndarray | None = None, decision_note: str = DEFAULT_DECISION_NOTE,
                 folder=None, extended: bool = False, log=None):
    """REPORTING: write <name>_oof.csv, _metrics.md and _metrics.json; return the markdown.

    P: (len(meta), 4) class probabilities, NaN for leaves that were not predicted.
    pred: optional (len(meta),) predicted class from the classifier itself (used for the
    confusion matrices and balanced accuracy instead of the most probable class).
    folder: output folder (default RESULTS_DIR, or RESULTS_DIR/quick when quick=True).
    log: optional logging.Logger for the "[saved]" line; None prints it (the historic behaviour).
    """
    oof = meta[["path", "variety", "row", "fold", "red_blotch", "leafroll3"]].copy()
    oof["class"] = leaf_classes(meta)
    oof[PROB_COLS] = P
    oof["p_rb"], oof["p_lr"] = P[:, 1] + P[:, 3], P[:, 2] + P[:, 3]
    if pred is not None:
        oof["pred"] = pred
    oof = oof.dropna(subset=PROB_COLS).reset_index(drop=True)
    if folder is None:
        folder = RESULTS_DIR / "quick" if quick else RESULTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    oof.to_csv(folder / f"{name}_oof.csv", index=False)
    metrics = score(oof)
    md = to_markdown(title, metrics, config, extra_md, decision_note, extended)
    (folder / f"{name}_metrics.md").write_text(md)
    write_json(folder / f"{name}_metrics.json",
               {"config": config, "classes": CLASSES, "metrics": metrics, **(extra_json or {})})
    msg = f"[saved] {folder / (name + '_oof.csv')}, _metrics.md and _metrics.json"
    (log.info if log is not None else print)(msg)
    return md
