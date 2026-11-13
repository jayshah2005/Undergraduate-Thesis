"""Write final_report.md (balanced accuracy only) and metrics.md (everything else) from the saved predictions.

Every number comes from cache/disease_search/ (preds/, prospect.npz, perm_*.npy, sawyer_protocol.json), so the
reports can be rebuilt in about a minute without retraining anything.

Search pools (the ES_ CNNs, early-stopped on the validation row, are never ranked):
- untuned families: every candidate without the "tuned:" or "lit:" prefix (the first two days of the search);
- tuned: the "tuned:" family (family_tuned.py, family_tuned_tabpfn.py, the tuned CNNs);
- literature: the "lit:" family (family_literature.py);
- every candidate: all of them.
The tuned and literature headlines were fixed before their test rows were scored; the untuned pool grew over
several rounds, each revealed on the same test rows, so its pick is reported as it stands now.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harness import CACHE, TASKS, VARIETIES, auc, bal_acc
from pooled import ba, oracle_ba, search_scores, table
from selection import honest, honest_ensemble, load_preds, strict, val_score

HERE = Path(__file__).resolve().parent
TASK_NAME = {"leafroll": "Leafroll", "red_blotch": "Red blotch"}
POOLS = {
    "untuned families": lambda n: not n.startswith(("tuned:", "lit:")),
    "tuned": lambda n: n.startswith("tuned:"),
    "literature": lambda n: n.startswith("lit:"),
    "every candidate": lambda n: True,
}
BASELINE = "lr_c1:snv_mean@within"      # the repo's logistic regression, reproduced exactly (check_baseline.py)
TABPFN_HEADLINE = "tabpfn:snv_mean@within"


# --------------------------------------------------------------------------
# Computation
# --------------------------------------------------------------------------
def test_ba(c: dict, task: str) -> dict:
    """Test balanced accuracy of one candidate in each variety."""
    return {v: bal_acc(c[(task, v)]["test"], c[(task, v)]["y"]) for v in VARIETIES}


def within_row_auc(c: dict, task: str, picks: list | None = None, pool: dict | None = None) -> float:
    """Mean test AUC inside each (variety, row); with picks, row k uses candidate picks[k] of pool."""
    s = []
    for k in range(5):
        cand = pool[picks[k]] if picks else c
        for v in VARIETIES:
            d = cand[(task, v)]
            m = d["row"] == k
            if 0 < d["y"][m].mean() < 1:
                s.append(auc(d["test"][m], d["y"][m]))
    return float(np.mean(s))


def score_pool(pool: dict, task: str) -> dict:
    """Headline picks (within and pooled) and the secondary scores of one search pool."""
    within, picks = honest(pool, task)
    own, _ = strict(pool, task)
    T, y, rows = table(pool, task)
    pooled = search_scores(T, y, rows)
    return {
        "n": len(pool), "within": within, "within_picks": picks,
        "within_own_row": float(np.mean(list(own.values()))),
        "within_top5": float(np.mean(list(honest_ensemble(pool, task, 5).values()))),
        "within_top10": float(np.mean(list(honest_ensemble(pool, task, 10).values()))),
        "within_row_auc": within_row_auc(None, task, picks, pool),
        "pooled": pooled["leave-test-row-out pick"][0], "pooled_picks": pooled["leave-test-row-out pick"][1],
        "pooled_own_row": pooled["own validation row pick"][0],
        "pooled_top5": pooled["top-5 mean logit"][0], "pooled_top10": pooled["top-10 mean logit"][0],
        "variety_only": ba(T["prior_only"]["test"], y),
    }


def untuned_name(n: str) -> str | None:
    """The untuned candidate with the same algorithm, features and mode as a "tuned:" one (None if none)."""
    _, algo, rest = n.split(":", 2)
    if algo == "cnn":
        return "cnn:small_cnn_mean_vis@within"
    return f"{algo}:{rest}" if algo in ("lr", "svm", "lda", "rf", "hgb", "tabpfn") else None


def tuning_pairs(P: dict) -> dict:
    """Per algorithm: mean (tuned - untuned) validation and test balanced accuracy over matched pairs."""
    out = {}
    for n in P:
        if n.startswith("tuned:") and untuned_name(n) in P:
            u = untuned_name(n)
            row = [val_score(P[n], t) - val_score(P[u], t) for t in TASKS]
            row += [np.mean(list(test_ba(P[n], t).values())) - np.mean(list(test_ba(P[u], t).values())) for t in TASKS]
            out.setdefault(n.split(":")[1], []).append(row)
    return {a: np.array(r) for a, r in sorted(out.items())}


def candidate_spread(P: dict) -> dict:
    """Over every candidate: median and 95th percentile of test balanced accuracy and within-row AUC."""
    out = {}
    for task in TASKS:
        b = np.array([np.mean(list(test_ba(c, task).values())) for c in P.values()])
        a = np.array([within_row_auc(c, task) for c in P.values()])
        out[task] = {"bal_acc": (np.median(b), np.percentile(b, 95), b.max()),
                     "within_row_auc": (np.median(a), np.percentile(a, 95))}
    return out


def oracle(P: dict) -> dict:
    """Validation-only ceiling of the pooled score: best per-variety thresholds chosen on the same labels."""
    return {t: np.array([oracle_ba(c, t, "val") for c in P.values()]) for t in TASKS}


def permutation_nulls(P: dict) -> list[dict]:
    """Observed validation score of each permuted candidate against its null (labels shuffled within variety x row)."""
    out = []
    for f in sorted(CACHE.glob("perm_*.npy")):
        stem = f.stem[len("perm_"):]
        mode = stem.rsplit("_", 1)[1]
        algo, feat = stem[: -len(mode) - 1].split("_", 1)
        name = f"{algo}:{feat}@{mode}"
        null = np.load(f)                                        # (task, shuffle, [val, test])
        for i, task in enumerate(TASKS):
            out.append({"name": name, "task": task, "observed": val_score(P[name], task),
                        "null_mean": null[i, :, 0].mean(), "null_95": np.percentile(null[i, :, 0], 95),
                        "n": null.shape[1]})
    return out


def trait_effects() -> dict:
    """Within-row standardised difference (diseased - healthy) of each PROSPECT trait, averaged over rows."""
    from logistic_regression import fold_of, load_meta, task_labels
    d = np.load(CACHE / "prospect.npz")
    T, names = d["traits"][:, :8], [str(n) for n in d["names"][:8]]
    meta = load_meta("all")
    rows = fold_of(meta)
    out = {"names": names}
    for task in TASKS:
        y = task_labels(meta, task)
        for v in VARIETIES:
            sel = (meta.variety == v).to_numpy()
            es = []
            for r in range(5):
                m = sel & (rows == r)
                if 2 <= y[m].sum() <= m.sum() - 2:
                    es.append((T[m][y[m] == 1].mean(0) - T[m][y[m] == 0].mean(0)) / T[m].std(0))
            out[(task, v)] = np.mean(es, 0)
    return out


def compute() -> dict:
    """Every number both reports need."""
    P = {n: c for n, c in load_preds().items() if not n.startswith("ES_")}
    proto = CACHE / "sawyer_protocol.json"
    return {
        "P": P,
        "pools": {task: {name: score_pool({n: c for n, c in P.items() if keep(n)}, task)
                         for name, keep in POOLS.items()} for task in TASKS},
        "baseline": {t: test_ba(P[BASELINE], t) for t in TASKS},
        "tabpfn": {t: test_ba(P[TABPFN_HEADLINE], t) for t in TASKS},
        "tuning": tuning_pairs(P),
        "spread": candidate_spread(P),
        "oracle": oracle(P),
        "perm": permutation_nulls(P),
        "traits": trait_effects(),
        "protocol": json.loads(proto.read_text()) if proto.exists() else None,
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def _row(name: str, by_variety: dict) -> str:
    return (f"| {name} | " + " | ".join(f"{by_variety[v]:.3f}" for v in VARIETIES)
            + f" | {np.mean(list(by_variety.values())):.3f} |")


def final_report(r: dict) -> str:
    lines = ["# Disease model search: results", "",
             "Can any leaf-level model detect leafroll or red blotch on held-out field rows? Over 500 candidate models "
             "(see README.md) were trained on the repo's row folds and chosen on validation rows only. Test balanced "
             "accuracy (0.5 = chance; a leaf is positive at P ≥ 0.5). The two diseases are scored separately.", "",
             "- **Within variety**: one score per variety, then their mean. **Pick**: for each test row, the candidate "
             "with the best validation score on the other rows' leaves scores that row.",
             "- **Pooled**: Baco + Chardonnay + Vidal leaves scored together, each model's output adjusted by its "
             "variety's infection rate in the training rows; *variety only* predicts from that rate alone.",
             "- Every pool was scored on the same test rows, several times over the search, so these numbers are "
             "slightly optimistic. Secondary scores and diagnostics: `metrics.md`.", ""]
    for task in TASKS:
        pools = r["pools"][task]
        lines += [f"## {TASK_NAME[task]}", "", "Within variety:", "",
                  "| model | " + " | ".join(VARIETIES) + " | mean |", "|---|" + "---|" * (len(VARIETIES) + 1),
                  _row("logistic regression on the SNV leaf mean (repo baseline)", r["baseline"][task])]
        lines += [_row(f"pick, {name} ({p['n']})", p["within"]) for name, p in pools.items()]
        lines += [_row("TabPFN on the SNV leaf mean (fixed in advance)", r["tabpfn"][task]), "",
                  "Baco + Chardonnay + Vidal pooled:", "", "| model | bal. acc. |", "|---|---|",
                  f"| variety only | {pools['every candidate']['variety_only']:.3f} |"]
        lines += [f"| pick, {name} ({p['n']}) | {p['pooled']:.3f} |" for name, p in pools.items()]
        lines += [""]
    return "\n".join(lines)


def metrics_report(r: dict) -> str:
    lines = ["# Disease model search: all scores", "",
             "Secondary scores and diagnostics behind `final_report.md`. Test rows unless marked validation.", ""]
    for task in TASKS:
        lines += [f"## {TASK_NAME[task]}", "", "Secondary search scores (within: mean of the 3 varieties):", "",
                  "| pool | within pick | own-row pick | top-5 | top-10 | within-row AUC of the pick | pooled pick | "
                  "own-row pick | top-5 | top-10 |", "|---|" + "---|" * 9]
        for name, p in r["pools"][task].items():
            lines.append(f"| {name} | {np.mean(list(p['within'].values())):.3f} | {p['within_own_row']:.3f} | "
                         f"{p['within_top5']:.3f} | {p['within_top10']:.3f} | {p['within_row_auc']:.3f} | "
                         f"{p['pooled']:.3f} | {p['pooled_own_row']:.3f} | {p['pooled_top5']:.3f} | {p['pooled_top10']:.3f} |")
        lines += ["", "*Own-row pick*: test row k uses fold k's own validation row only. *Top-k*: mean of the k best "
                  "on validation. *Within-row AUC*: AUC inside each variety × row, then averaged; rows differ in "
                  "infection rate, so a model can score above 0.5 across rows with no signal inside any row.", "",
                  "Picks, test rows R1–R5:", ""]
        for name, p in r["pools"][task].items():
            lines += [f"- {name}, within: " + ", ".join(f"`{x}`" for x in p["within_picks"]),
                      f"- {name}, pooled: " + ", ".join(f"`{x}`" for x in p["pooled_picks"])]
        s = r["spread"][task]
        o = r["oracle"][task]
        lines += ["", f"Over all {len(r['P'])} candidates: test balanced accuracy median {s['bal_acc'][0]:.3f}, 95th "
                  f"percentile {s['bal_acc'][1]:.3f}, best {s['bal_acc'][2]:.3f} (chosen after seeing test, so not a "
                  f"result); within-row AUC median {s['within_row_auc'][0]:.3f}, 95th percentile "
                  f"{s['within_row_auc'][1]:.3f}.", "",
                  f"Pooled ceiling on validation (each candidate's best per-variety thresholds, chosen on the same "
                  f"labels): median {np.median(o):.3f}, best {o.max():.3f}.", ""]
    lines += ["## Tuning", "", "Tuned minus untuned, same algorithm, features and mode (mean balanced accuracy over "
              "the 3 varieties; the test column was computed after the tuned headline was scored):", "",
              "| algorithm | pairs | validation leafroll | validation red blotch | test leafroll | test red blotch |",
              "|---|---|---|---|---|---|"]
    for a, d in r["tuning"].items():
        lines.append(f"| {a} | {len(d)} | " + " | ".join(f"{x:+.3f}" for x in d.mean(0)) + " |")
    if r["protocol"]:
        lines += ["", "## Split comparison (Sawyer et al. 2023)", "",
                  "The same models with Sawyer et al.'s split (stratified 5-fold over shuffled leaves, 5 repeats) and "
                  "with whole rows held out (leave one row out, so 4 training rows). Accuracy / balanced accuracy. "
                  "Not a result: shuffled folds share rows between training and test.", "",
                  "| leaves | task | model | shuffled | rows held out | majority-class accuracy |", "|---|---|---|---|---|---|"]
        for t in r["protocol"]:
            lines.append(f"| {t['leaves']} | {TASK_NAME[t['task']]} | {t['model']} | {t['shuffled_acc']:.3f} / "
                         f"{t['shuffled_bal_acc']:.3f} | {t['rows_acc']:.3f} / {t['rows_bal_acc']:.3f} | "
                         f"{t['majority_acc']:.3f} |")
    if r["perm"]:
        lines += ["", "## Permutation nulls", "", "Labels shuffled within each variety × row (keeps every row's "
                  "infection rate), the whole candidate rerun, inner tuning included. Validation, mean of the 3 "
                  "varieties.", "", "| candidate | task | observed | null mean | null 95th pct | shuffles |",
                  "|---|---|---|---|---|---|"]
        for p in r["perm"]:
            lines.append(f"| `{p['name']}` | {TASK_NAME[p['task']]} | {p['observed']:.3f} | {p['null_mean']:.3f} | "
                         f"{p['null_95']:.3f} | {p['n']} |")
    tr = r["traits"]
    lines += ["", "## PROSPECT-D traits", "", "Within-row standardised difference (diseased − healthy) of each fitted "
              "trait, averaged over rows (all 5 rows, computed after the literature headline was scored). Pigments "
              "are relative PROSPECT-fitted values.", "",
              "| task | variety | " + " | ".join(tr["names"]) + " |", "|---|---|" + "---|" * len(tr["names"])]
    for task in TASKS:
        for v in VARIETIES:
            lines.append(f"| {TASK_NAME[task]} | {v} | " + " | ".join(f"{x:+.2f}" for x in tr[(task, v)]) + " |")
    return "\n".join(lines + [""])


def main() -> None:
    """Load every prediction → compute → write final_report.md and metrics.md here."""
    r = compute()
    (HERE / "final_report.md").write_text(final_report(r))
    (HERE / "metrics.md").write_text(metrics_report(r))
    print(f"[saved] final_report.md, metrics.md ({len(r['P'])} candidates)")


if __name__ == "__main__":
    main()
