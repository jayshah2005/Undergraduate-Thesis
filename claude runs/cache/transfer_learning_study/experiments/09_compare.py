"""STEP 9 - read every results/*_oof.csv and build ONE comparison table.

Run:   python experiments/09_compare.py              (reads results/)
       python experiments/09_compare.py --quick      (reads results/quick/, smoke-test runs)

Every approach saved out-of-fold (OOF) probabilities: each leaf was predicted by
a model that never saw it (or its row) during training. This script scores them
all identically (see evaluate() in common.py) and writes:

  results/summary.csv        all numbers, one row per approach
  results/summary.md         the same as a readable markdown table
  results/per_variety.csv    AUCs separately for each variety
  results/paired_tests.md    'is A really better than B?' bootstrap comparisons

HOW TO READ THE TABLE
  * AUC = probability that a random sick leaf gets a higher score than a random
    healthy one. 0.5 = coin flip, 1.0 = perfect.
  * "pooled" AUC mixes all varieties. It is flattered by variety shortcuts
    (see the variety-prior row, which looks at nothing but the variety).
  * "within-variety" AUC only compares a leaf with leaves of the SAME variety (and
    the same CV fold). This is the honest measure of "can it tell sick from
    healthy leaves of the same variety?" The variety-prior row scores 0.50 here
    by construction.
  * Numbers after +- are 95% bootstrap half-widths (resampling leaves). Gaps
    smaller than the +- are not distinguishable from noise.
  * 'any disease' = P(red blotch or leafroll) = healthy vs not healthy.
  * 'bal.acc ANY' is balanced accuracy at probability 0.5 and is ~0.5 for most runs. That is
    a threshold artefact (P(any) combines two class-weighted heads, so most leaves land
    above 0.5), not a bug; use the AUC columns to compare approaches.
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

from common import RESULTS_DIR, auc_metrics, bootstrap_ci, evaluate, oof_arrays

# (A, B, what the comparison answers)
PAIRS = [
    ("06_resnet18_pretrained", "06_resnet18_scratch",
     "Transfer learning: ImageNet-pretrained ResNet-18 vs the SAME net from random weights"),
    ("08_ssl_pretrained", "08_ssl_scratch",
     "Self-supervised pretraining vs the SAME net from random weights"),
    ("06_resnet18_pretrained", "04_pixel_cnn1d",
     "Pretrained whole-leaf ResNet vs pixel-level 1D CNN from scratch"),
    ("05_attention_mil", "04_pixel_mlp",
     "Attention-MIL pooling vs plain averaging of pixel predictions"),
    ("07_dofa_probe", "06_resnet18_pretrained",
     "Hyperspectral foundation model (frozen) vs ImageNet ResNet"),
    # descriptive only: this pair was chosen AFTER seeing the results (best run vs best classical)
    ("07_dinov2_probe", "02_plsda__mean_snv__byvariety",
     "[post-hoc] best run overall vs best classical run"),
]


def _ap_gain_within(y, s, strata):
    """Average precision minus its chance level (= prevalence), computed inside each
    (variety, fold) stratum and averaged weighted by stratum size. 0 = no better than
    chance, positive = real signal. (Plain AP cannot be pooled across strata fairly because
    its chance level equals the prevalence, which differs a lot between varieties.)"""
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        if y[m].sum() == 0 or y[m].sum() == m.sum():
            continue
        num += (average_precision_score(y[m], s[m]) - y[m].mean()) * m.sum()
        den += m.sum()
    return num / den if den else np.nan


def extra_metrics(oof: pd.DataFrame) -> dict:
    """Average-precision view (see 'Why ROC AUC' in the README).

    Three targets: red blotch (positive class = has it), leafroll, and HEALTHY (positive
    class = healthy leaf, scored by P(healthy) = (1-p_rb)(1-p_lr)). Healthy is the rare
    class (23% of leaves), so it is where AP and AUC can disagree. For each target:
      ap_*       pooled average precision      (chance level = prevalence, printed too)
      apgain_*   within-variety AP minus prevalence (0 = chance)
    """
    y_rb, y_lr, s_rb, s_lr, ok_lr, strata = oof_arrays(oof)
    y_h = 1 - ((y_rb == 1) | (y_lr == 1)).astype(int)
    s_h = (1 - s_rb) * (1 - s_lr)
    out = {}
    for key, y, s, ok in (("rb", y_rb, s_rb, np.ones(len(y_rb), bool)),
                          ("lr", y_lr, s_lr, ok_lr), ("healthy", y_h, s_h, np.ones(len(y_h), bool))):
        out[f"ap_{key}"] = average_precision_score(y[ok], s[ok])
        out[f"prev_{key}"] = y[ok].mean()
        out[f"apgain_{key}"] = _ap_gain_within(y[ok], s[ok], strata[ok])
    return out


def score_one(name: str, oof: pd.DataFrame, n_boot: int):
    """Metrics + bootstrap intervals for one approach (runs in a worker process)."""
    r = evaluate(oof)
    pv = r.pop("per_variety").assign(approach=name)
    r.update(bootstrap_ci(oof, n_boot=n_boot))
    r.update(extra_metrics(oof))
    return {"approach": name, **r}, pv


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids needing the 'tabulate' package)."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def family_of(name: str) -> str:
    """Group runs by script, e.g. '02_svm__mean_raw__byvariety' -> '02 classical, per-variety models'."""
    fam = {"01": "01 variety prior (not a model)", "02": "02 classical on leaf-mean spectra",
           "03": "03 classical on pixels", "04": "04 pixel MLP / 1D CNN", "05": "05 attention MIL",
           "06": "06 ResNet-18 (transfer vs scratch)", "07": "07 foundation models (DOFA, DINOv2)",
           "08": "08 self-supervised pretraining"}
    base = fam.get(name[:2], name[:2])
    return base + (", per-variety models" if name.endswith("__byvariety") else "")


def family_table(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per script family: how many variants and the spread of within-variety AUC."""
    s = summary.assign(family=summary.approach.map(family_of))
    rows = []
    for fam, g in s.groupby("family"):
        best = g.loc[g.auc_any_within.idxmax()]
        rows.append({"family": fam, "variants": len(g),
                     "within ANY: worst": f"{g.auc_any_within.min():.3f}",
                     "median": f"{g.auc_any_within.median():.3f}",
                     "best": f"{g.auc_any_within.max():.3f}",
                     "within RB (best)": f"{g.auc_rb_within.max():.3f}",
                     "within LR (best)": f"{g.auc_lr_within.max():.3f}",
                     "best variant": best.approach})
    return pd.DataFrame(rows).sort_values("best", ascending=False)


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot=300, seed=0):
    """95% CI of (metric_A - metric_B) on the same resampled leaves."""
    base = a.drop(columns=["p_rb", "p_lr"])                 # path, variety, fold, labels
    both = a[["path", "p_rb", "p_lr"]].merge(b[["path", "p_rb", "p_lr"]], on="path",
                                              suffixes=("_a", "_b"))
    base = base.merge(both[["path"]], on="path")           # only leaves present in both runs
    A = oof_arrays(base.assign(p_rb=both.p_rb_a.to_numpy(), p_lr=both.p_lr_a.to_numpy()))
    B = oof_arrays(base.assign(p_rb=both.p_rb_b.to_numpy(), p_lr=both.p_lr_b.to_numpy()))
    keys = ("auc_any", "auc_any_within", "auc_rb_within", "auc_lr_within")
    full_a, full_b = auc_metrics(*A), auc_metrics(*B)
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(base), len(base))
        ra, rb = auc_metrics(*(x[idx] for x in A)), auc_metrics(*(x[idx] for x in B))
        for k in keys:
            diffs[k].append(ra[k] - rb[k])
    return {k: (full_a[k] - full_b[k], np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
            for k, v in diffs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    folder = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR

    oofs = {p.stem.removesuffix("_oof"): pd.read_csv(p) for p in sorted(folder.glob("*_oof.csv"))}
    if not oofs:
        raise SystemExit(f"no *_oof.csv files in {folder}")

    def build(subset):
        """Score every approach in subset -> (display table, numeric summary, per-variety table)."""
        scored = Parallel(n_jobs=-1)(delayed(score_one)(n, o, args.boot) for n, o in subset.items())
        rows = [r for r, _ in scored]
        per_var = [pv for _, pv in scored]
        summ = pd.DataFrame(rows).sort_values("auc_any_within", ascending=False)

        def fmt(v, ci=None):
            if pd.isna(v):
                return "-"
            return f"{v:.3f}" + (f" ±{ci:.3f}" if ci is not None and not pd.isna(ci) else "")

        table = pd.DataFrame({
            "approach": summ.approach,
            "n": summ.n_leaves,
            "AUC red blotch": [fmt(v, c) for v, c in zip(summ.auc_rb, summ.auc_rb_ci)],
            "AUC leafroll": [fmt(v, c) for v, c in zip(summ.auc_lr, summ.auc_lr_ci)],
            "AUC any disease": [fmt(v, c) for v, c in zip(summ.auc_any, summ.auc_any_ci)],
            "within-variety RB": [fmt(v, c) for v, c in zip(summ.auc_rb_within, summ.auc_rb_within_ci)],
            "within-variety LR": [fmt(v, c) for v, c in zip(summ.auc_lr_within, summ.auc_lr_within_ci)],
            "within-variety ANY": [fmt(v, c) for v, c in zip(summ.auc_any_within, summ.auc_any_within_ci)],
            "bal.acc ANY": summ.bacc_any.map(fmt),
        })
        ap_table = pd.DataFrame({
            "approach": summ.approach,
            "AP healthy": summ.ap_healthy.map(fmt),
            "(chance)": summ.prev_healthy.map(fmt),
            "within-variety AP gain, healthy": summ.apgain_healthy.map(lambda v: f"{v:+.3f}"),
            "AP red blotch": summ.ap_rb.map(fmt),
            "(chance) ": summ.prev_rb.map(fmt),
            "within-variety AP gain, RB": summ.apgain_rb.map(lambda v: f"{v:+.3f}"),
            "within-variety AP gain, LR": summ.apgain_lr.map(lambda v: f"{v:+.3f}"),
        }).iloc[np.argsort(-summ.apgain_healthy.to_numpy())]
        return table, summ, pd.concat(per_var), ap_table

    # leave-one-variety-out runs (names end in __lovo) use a different protocol -> own table
    main_runs = {k: v for k, v in oofs.items() if not k.endswith("__lovo")}
    lovo_runs = {k: v for k, v in oofs.items() if k.endswith("__lovo")}
    table, summary, per_var, ap_table = build(main_runs)
    fam = family_table(summary)
    # How many runs show evidence of signal beyond the variety? (interval entirely above 0.5)
    n_runs = len(summary)
    signal = {lab: int((summary[f"auc_{lab}_within_lo"] > 0.5).sum()) for lab in ("rb", "lr", "any")}
    signal_line = (f"Runs whose within-variety 95% interval lies entirely above 0.5 (of {n_runs}): "
                   f"red blotch {signal['rb']}, leafroll {signal['lr']}, any disease {signal['any']}. "
                   "Runs are highly correlated and ~46 were screened, so a few exceed 0.5 by chance alone.")
    print("\n" + signal_line)
    md = (signal_line + "\n\n## Summary by approach family (within-variety AUC; 'variants' = settings tried)\n\n"
          + to_markdown(fam)
          + "\n\n## Main comparison, every run (5 folds, each fold = one row of every variety)\n\n"
          + to_markdown(table))
    print("\n## Summary by approach family\n\n" + to_markdown(fam))
    md += ("\n\n## Average-precision view (positive class = healthy, red blotch; sorted by within-variety "
           "healthy AP gain; 'AP gain' = AP minus chance level, 0 = chance)\n\n" + to_markdown(ap_table))
    print("\n## Average precision, top 8 by healthy AP gain\n\n" + to_markdown(ap_table.head(8)))
    print(f"\n## Top 12 of {len(table)} runs (full table in results/summary.md)\n\n"
          + to_markdown(table.head(12)) + "\n")
    if lovo_runs:
        t2, s2, pv2, _ = build(lovo_runs)
        md2 = ("## Stress test: leave-one-variety-out (the model never saw the test variety)\n\n"
               + to_markdown(t2))
        print("\n" + md2.split("\n\n")[0] + f"\n\n(see results/summary.md for all {len(t2)} runs)\n")
        md += "\n\n" + md2
        summary = pd.concat([summary, s2])
        per_var = pd.concat([per_var, pv2])
    if not args.quick:
        summary.to_csv(folder / "summary.csv", index=False)
        (folder / "summary.md").write_text(md + "\n")
        per_var.to_csv(folder / "per_variety.csv", index=False)

    # ---- paired comparisons -------------------------------------------------
    lines = ["| comparison | metric | A - B | 95% CI | verdict |", "|---|---|---|---|---|"]
    for a, b, question in PAIRS:
        if a in oofs and b in oofs:
            res = paired_bootstrap(oofs[a], oofs[b])
            lines.append(f"| **{a}** vs **{b}**: {question} | | | | |")
            for k, label in (("auc_any", "AUC any (pooled)"), ("auc_any_within", "AUC any (within-variety)"),
                             ("auc_rb_within", "AUC red blotch (within)"), ("auc_lr_within", "AUC leafroll (within)")):
                d, lo, hi = res[k]
                verdict = "A better" if lo > 0 else "B better" if hi < 0 else "no clear difference"
                lines.append(f"| | {label} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict} |")
    paired_md = "\n".join(lines)
    print(paired_md)
    if not args.quick:
        (folder / "paired_tests.md").write_text(paired_md + "\n")
        print(f"\nwritten to {folder}")


if __name__ == "__main__":
    main()
