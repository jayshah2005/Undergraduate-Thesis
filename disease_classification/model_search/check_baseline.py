"""Check: the harness reproduces the repo baseline's within-variety TEST scores exactly (known, published)."""
from harness import evaluate, load_all, bal_acc
from family_spectral import baseline
res = evaluate(baseline(), load_all(), "within")
expect = {("leafroll", "Baco"): 0.524, ("red_blotch", "Baco"): 0.506, ("leafroll", "Chardonnay"): 0.537,
          ("red_blotch", "Chardonnay"): 0.605, ("leafroll", "Vidal"): 0.544, ("red_blotch", "Vidal"): 0.552}
for key, e in expect.items():
    d = res["preds"][key]
    got = bal_acc(d["test"], d["y"])
    print(key, f"{got:.3f}", "expected", e, "OK" if f"{got:.3f}" == f"{e:.3f}" else "MISMATCH", "n", len(d["y"]))
