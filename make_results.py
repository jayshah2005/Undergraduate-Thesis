"""Build RESULTS.md: every model of ANALYSIS_PLAN.md scored on one page, with every planned comparison."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DISEASE_LR = ROOT / "disease_classification" / "logistic_regression"
for p in (ROOT, DISEASE_LR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from helpers.backbones import LABEL, WITHOUT_BLOCK_ATTENTION  # noqa: E402
from helpers.cnn_common import PIXEL_NORM  # noqa: E402
from helpers.data import load_dataset, load_leaf_features  # noqa: E402
from helpers.log import get_logger  # noqa: E402
from helpers.logreg import fit_with_validation  # noqa: E402
from helpers.pca_common import VARIETIES, get_device  # noqa: E402
from helpers.report_tables import BASELINE, delta, score, table  # noqa: E402
from helpers.scoring import auc, balanced_accuracy_multiclass, bootstrap_weights, macro_auc  # noqa: E402
# The disease folder's module: task labels, the leaves each task uses, the folds and the LR baseline.
# Only this one category's module is imported: the variety folder has modules of the same names.
import logistic_regression as disease_lr  # noqa: E402

assert Path(disease_lr.__file__).resolve().parent == DISEASE_LR, f"wrong module: {disease_lr.__file__}"

OUT = ROOT / "RESULTS.md"
VARIETY_CNN = ROOT / "variety_classification" / "cnn" / "results"
DISEASE_CNN = ROOT / "disease_classification" / "cnn" / "results"
WITHIN_CNN = ROOT / "disease_classification" / "within_variety_disease_cnn" / "results"
MOO = ROOT / "moo_single_task_learning" / "results"

SEEDS, FOLDS = (0, 1, 2, 3, 4), range(5)
N_LEAVES = 998
MIN_SEEDS = 4                                   # rule 2: A beats B in at least 4 of the 5 seeds
BANDS = ("all", "rgb")
BANDS_NAME = {"all": "204 bands", "rgb": "RGB"}
NETWORKS = ("small_cnn", "small_resnet")
MODULES = ("", "_eca", "_cbam")
POOLINGS = ("mean", "attention")
BACKBONES = [n + m for n in NETWORKS for m in MODULES]
OPTIMIZERS = {"adamw": "AdamW", "sgd": "SGD"}
METHODS = {"ls": "LS", "mgda": "MGDA", "mgda_gn": "MGDA-GN", "mgda_ln": "MGDA-LN", "mgda_lgn": "MGDA-LGN",
           "svdpga": "SVD-PGA", "osnmfpga": "OSNMF-PGA", "upgrad": "UPGrad"}
# The six heads of the MOO network (moo_single_task_learning/run_moo.py HEADS, CHECKPOINTS).
MOO_CHECKPOINTS = {"variety": [0, 1, 2, 3], "red_blotch": [4], "leafroll": [5], "all": [0, 1, 2, 3, 4, 5]}
BCV = ("Baco", "Chardonnay", "Vidal")
WITHIN = BCV
RULE = "variety-only rule"
TASK_NAME = {"leafroll": "Leafroll", "red_blotch": "Red blotch"}
FAMILY = {
    "1": "against logistic regression, same data",
    "1b": "against the variety-only rule (pooled sets)",
    "2a": "ECA / CBAM against the same network and pooling without them",
    "2b": "attention against mean pooling, same network",
    "3": "small ResNet against CNN, same module and pooling",
    "4": "204 bands against RGB, same model",
    "5a": "each aggregator against LS, same network, optimizer and bands",
    "5b": "SGD against AdamW, same network, method and bands",
    "6": "six-head LS (AdamW) against the single-task model, same network, attention pooling",
}


def cnn_name(backbone: str, pooling: str) -> str:
    """Report name of a single-task CNN, e.g. "CNN + ECA, mean pool" (as in the per-folder reports)."""
    return f"{LABEL[backbone]}, {pooling} pool"


def moo_name(network: str, optimizer: str, method: str) -> str:
    """Report name of a six-head model, e.g. "six-head CNN, AdamW, LS"."""
    return f"six-head {LABEL[network]}, {OPTIMIZERS[optimizer]}, {METHODS[method]}"


CNNS = [(bb, pool) for bb in BACKBONES for pool in POOLINGS]
CNN_NAMES = [cnn_name(*c) for c in CNNS]
MOO_MODELS = [(nw, opt, m) for nw in NETWORKS for opt in OPTIMIZERS for m in METHODS]


# --------------------------------------------------------------------------
# Scored sets and scores
# --------------------------------------------------------------------------
@dataclass
class Panel:
    """One scored set of leaves: every model scored on it shares its resamples, so differences are paired.

    n: leaves scored. B: resamples.
    """

    idx: np.ndarray        # (n,) positions in the 998-leaf order of helpers.data.load_dataset
    y: torch.Tensor        # (n,) class 0..K-1 (0/1 for a disease)
    n_classes: int         # K
    w: torch.Tensor        # (B, n) resampling counts from helpers.scoring.bootstrap_weights


@dataclass
class Score:
    """One model on one panel: the seed ensemble's scores with 95% intervals, each seed's scores, the resamples."""

    ba: float
    ba_ci: list[float]
    auc: float
    auc_ci: list[float]
    seeds: dict[str, np.ndarray]       # {"ba", "auc"}: score of each seed alone (one value for a single fit)
    boot: dict[str, torch.Tensor]      # {"ba", "auc"}: the ensemble's score in each resample, (B,)


@dataclass
class Diff:
    """A − B for one score: the paired difference, its interval, the seed count and the mark."""

    value: float
    ci: list[float]
    wins: int          # seeds where A scores above B
    losses: int        # seeds where A scores below B
    seeded: bool       # False when both are single fits (rule 2 then reduces to the sign of the difference)
    mark: str          # "✓", "✗" or ""


@dataclass
class Row:
    """One comparison line."""

    family: str
    a: str
    b: str
    bands: str
    score_a: Score
    score_b: Score
    ba: Diff
    auc: Diff


@dataclass
class Block:
    """One setting's scores and comparisons (e.g. red blotch, Chardonnay)."""

    title: str
    panel: Panel
    models: list[str]                                        # display order
    scores: dict = field(default_factory=dict)               # {(model, bands): Score, or a str why missing}
    rows: list[Row] = field(default_factory=list)


def panel_of(idx: np.ndarray, y: np.ndarray, group: np.ndarray, n_classes: int, device: torch.device) -> Panel:
    """A scored set with its resamples: leaves drawn within each (group, class) cell, seeded.

    The same leaves in the same order with the same labels and groups always get the same resamples, so the
    LR intervals here equal the ones in the per-folder reports.
    """
    yt = torch.tensor(y[idx], dtype=torch.long, device=device)
    w = bootstrap_weights(yt, torch.tensor(group[idx], device=device))
    return Panel(idx=idx, y=yt, n_classes=n_classes, w=w)


def metric_pair(panel: Panel, P: torch.Tensor, w: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced accuracy and AUC of predictions P on a panel (or on each resample, with w).

    A disease leaf is called positive at P ≥ 0.5; a variety leaf gets its most likely variety. Variety AUC is
    the mean of the 4 one-vs-rest AUCs.

    :param P: P(positive), shape (n,), or class scores, shape (n, K).
    :return: Two tensors, shape () or (B,).
    """
    if panel.n_classes == 2:
        return balanced_accuracy_multiclass((P >= 0.5).long(), panel.y, 2, w), auc(P, panel.y, w)
    return balanced_accuracy_multiclass(P.argmax(-1), panel.y, panel.n_classes, w), macro_auc(P, panel.y, w)


def score_model(panel: Panel, P_seeds: np.ndarray) -> Score:
    """Score one model's seed ensemble (the mean probability) with intervals, and each seed alone.

    :param P_seeds: Probabilities on the panel's leaves, shape (S, n) or (S, n, K); S = 1 for a single fit.
    """
    P = torch.tensor(P_seeds, dtype=torch.float64, device=panel.y.device)
    ens = P.mean(0)
    ba, au = metric_pair(panel, ens)
    ba_boot, au_boot = metric_pair(panel, ens, panel.w)
    per_seed = [metric_pair(panel, p) for p in P]
    q = torch.tensor([0.025, 0.975], dtype=torch.float64, device=P.device)
    return Score(ba=ba.item(), ba_ci=torch.quantile(ba_boot, q).tolist(),
                 auc=au.item(), auc_ci=torch.quantile(au_boot, q).tolist(),
                 seeds={"ba": np.array([s[0].item() for s in per_seed]),
                        "auc": np.array([s[1].item() for s in per_seed])},
                 boot={"ba": ba_boot, "auc": au_boot})


def compare(a: Score, b: Score, metric: str) -> Diff:
    """A − B with its paired 95% interval and the two-part mark of ANALYSIS_PLAN.md.

    ✓ when the interval is above 0 and A beats B in at least MIN_SEEDS seeds (seed s against seed s; a single
    fit is compared with every seed); ✗ the same way round. Tied seeds count for neither.
    """
    diff = a.boot[metric] - b.boot[metric]
    ci = torch.quantile(diff, torch.tensor([0.025, 0.975], dtype=torch.float64, device=diff.device)).tolist()
    value = getattr(a, metric) - getattr(b, metric)
    sa, sb = a.seeds[metric], b.seeds[metric]
    seeded = len(sa) > 1 or len(sb) > 1
    sa, sb = np.broadcast_arrays(sa, sb)
    wins, losses = int((sa > sb).sum()), int((sa < sb).sum())
    up = wins >= MIN_SEEDS if seeded else value > 0
    down = losses >= MIN_SEEDS if seeded else value < 0
    mark = "✓" if ci[0] > 0 and up else "✗" if ci[1] < 0 and down else ""
    return Diff(value=value, ci=ci, wins=wins, losses=losses, seeded=seeded, mark=mark)


def add_row(block: Block, family: str, a: tuple[str, str], b: tuple[str, str]) -> None:
    """Add the comparison A − B ((model, bands) each) to a block, if both models have scores."""
    sa, sb = block.scores.get(a), block.scores.get(b)
    if not isinstance(sa, Score) or not isinstance(sb, Score):
        return
    same = a[1] == b[1] or b[0] == RULE   # the rule uses no spectrum
    bands = BANDS_NAME[a[1]] if same else f"{BANDS_NAME[a[1]]} − {BANDS_NAME[b[1]]}"
    block.rows.append(Row(family=family, a=a[0], b=b[0], bands=bands, score_a=sa, score_b=sb,
                          ba=compare(sa, sb, "ba"), auc=compare(sa, sb, "auc")))


# --------------------------------------------------------------------------
# Loading predictions (every array is in the 998-leaf order; NaN where a leaf was not tested)
# --------------------------------------------------------------------------
def read_folds(folder: Path, keys: tuple[str, ...], to998: np.ndarray, shape: tuple,
               paths: np.ndarray | None = None) -> dict | None:
    """Out-of-fold predictions of one model, 5 seeds × 5 folds; None if any of the 25 files is missing.

    A model with only some seeds is left out rather than averaged over fewer seeds.

    :param folder: Holds seed<s>/fold<k>.npz.
    :param keys: npz arrays to collect, each (n_test, *shape).
    :param to998: Position in the 998-leaf order of each leaf the run's test_idx indexes.
    :param shape: Shape per leaf: () for a disease, (4,) for variety, (6,) for the six heads.
    :param paths: The 998 leaf paths, to check runs that save their test leaves' paths (MOO).
    :return: {key: (5, 998, *shape)} or None.
    """
    files = [folder / f"seed{s}" / f"fold{k}.npz" for s in SEEDS for k in FOLDS]
    if not all(f.exists() for f in files):
        return None
    out = {key: np.full((len(SEEDS), N_LEAVES, *shape), np.nan) for key in keys}
    for s in SEEDS:
        for k in FOLDS:
            f = np.load(folder / f"seed{s}" / f"fold{k}.npz")
            leaves = to998[f["test_idx"]]
            if paths is not None:
                assert list(f["paths"]) == list(paths[leaves]), f"{folder}: leaf order differs"
            for key in keys:
                out[key][s, leaves] = f[key]
    return out


def check_config(folder: Path, want: dict) -> None:
    """Stop if a results folder's config.json does not match the model the page puts it under."""
    cfg = json.loads((folder / "config.json").read_text())
    flat = {**cfg, **cfg.get("training", {}), **cfg.get("train_config", {})}
    flat.setdefault("pixel_norm", "snv")   # folders made before pixel_norm was recorded used SNV
    bad = {k: (flat.get(k), v) for k, v in want.items() if flat.get(k) != v}
    if bad:
        raise SystemExit(f"{folder}/config.json differs from what the page expects (have, want): {bad}")
    assert set(SEEDS) <= set(cfg["seeds"]), f"{folder}: seeds {cfg['seeds']}"


def positions(meta998, meta) -> np.ndarray:
    """Position in the 998-leaf order of each leaf of a subset's leaf table."""
    pos = {p: i for i, p in enumerate(meta998.path)}
    return np.array([pos[p] for p in meta.path])


def variety_cnn(backbone: str, pooling: str, bands: str) -> np.ndarray | None:
    """One variety CNN's probabilities, (5, 998, 4), or None if not all run."""
    folder = VARIETY_CNN / (bands + ("_attention" if pooling == "attention" else "")
                            + ("" if backbone == "small_cnn" else f"_{backbone}"))
    if not folder.exists():
        return None
    check_config(folder, {"bands": bands, "backbone": backbone, "pooling": pooling, "pixel_norm": PIXEL_NORM[bands]})
    out = read_folds(folder, ("prob",), np.arange(N_LEAVES), (4,))
    return None if out is None else out["prob"]


def disease_cnn(source: str, task: str, backbone: str, pooling: str, bands: str, to998: np.ndarray) -> np.ndarray | None:
    """One single-task disease CNN's P(positive), (5, 998), or None if not all run.

    :param source: The leaves it trained on: a variety (within-variety folder), "all" or "baco_chardonnay_vidal".
    :param to998: Positions of the source's leaves in the 998-leaf order (its test_idx index those leaves).
    """
    root = WITHIN_CNN if source in WITHIN else DISEASE_CNN
    folder = root / (disease_lr.results_name(source, bands) + ("" if backbone == "small_cnn" else f"_{backbone}"))
    run = folder / "runs" / task / pooling
    if not run.exists():
        return None
    check_config(folder, {"variety": source, "bands": bands, "backbone": backbone, "pixel_norm": PIXEL_NORM[bands]})
    out = read_folds(run, ("prob",), to998, ())
    return None if out is None else out["prob"]


def moo_run(network: str, optimizer: str, method: str, bands: str, paths: np.ndarray) -> dict | None:
    """One six-head model's probabilities per checkpoint choice, {choice: (5, 998, 6)}, or None if not all run."""
    folder = MOO / (f"{network}_{optimizer}" + ("_rgb" if bands == "rgb" else ""))
    if not (folder / method).exists():
        return None
    check_config(folder, {"backbone": network, "optimizer": optimizer, "pooling": "attention",
                          "pixel_norm": PIXEL_NORM[bands], "checkpoints": MOO_CHECKPOINTS})
    out = read_folds(folder / method, tuple(f"prob_{c}" for c in MOO_CHECKPOINTS), np.arange(N_LEAVES), (6,), paths)
    return None if out is None else {c: out[f"prob_{c}"] for c in MOO_CHECKPOINTS}


def variety_lr(meta998, bands: str) -> np.ndarray:
    """Logistic regression on the leaf features (helpers.logreg recipe), out of fold, (1, 998, 4)."""
    X = load_leaf_features(bands)
    y = meta998.variety.map(VARIETIES.index).to_numpy()
    fold = disease_lr.fold_of(meta998)
    P = np.full((N_LEAVES, len(VARIETIES)), np.nan)
    for k in FOLDS:
        tr, va, te = disease_lr.fold_split(fold, k)   # test Rk, validation R(k+1), train the other 3
        _C, model = fit_with_validation(X, y, tr, va)
        P[te] = model.predict_proba(X[te])
    return P[None]


def variety_rule(meta, task: str) -> np.ndarray:
    """P(positive) from a leaf's variety alone, out of fold, over meta's leaves (0 where the task leaves one out).

    For each fold, each variety's disease rate r on the 3 training rows, weighted like the CNN loss
    (positives × #neg / #pos): P = r·w / (r·w + 1 − r). It uses no spectrum; within one variety every leaf
    gets the same P, so it scores exactly chance there.
    """
    y = disease_lr.task_labels(meta, task)
    keep = disease_lr.usable_leaves(meta, y)
    fold, variety = disease_lr.fold_of(meta), meta.variety.to_numpy()
    p = np.zeros(len(y))
    for k in FOLDS:
        tr, _va, te = disease_lr.fold_split(fold, k, keep)
        w = (1 - y[tr]).sum() / y[tr].sum()
        for v in np.unique(variety[te]):
            r = y[tr][variety[tr] == v].mean()
            p[te[variety[te] == v]] = r * w / (r * w + 1 - r)
    return p


# --------------------------------------------------------------------------
# The three categories
# --------------------------------------------------------------------------
def single_task_rows(block: Block, pooled: bool) -> None:
    """Comparisons 1-4 of ANALYSIS_PLAN.md (and 1b for a pooled set), in that order."""
    for b in BANDS:
        for name in CNN_NAMES:
            add_row(block, "1", (name, b), (BASELINE, b))
    if pooled:
        for b in BANDS:
            for name in [BASELINE, *CNN_NAMES]:
                add_row(block, "1b", (name, b), (RULE, "all"))   # the rule uses no spectrum
    for b in BANDS:
        for bb, base in WITHOUT_BLOCK_ATTENTION.items():
            for pool in POOLINGS:
                add_row(block, "2a", (cnn_name(bb, pool), b), (cnn_name(base, pool), b))
    for b in BANDS:
        for bb in BACKBONES:
            add_row(block, "2b", (cnn_name(bb, "attention"), b), (cnn_name(bb, "mean"), b))
    for b in BANDS:
        for m in MODULES:
            for pool in POOLINGS:
                add_row(block, "3", (cnn_name("small_resnet" + m, pool), b), (cnn_name("small_cnn" + m, pool), b))
    for name in [BASELINE, *CNN_NAMES]:
        add_row(block, "4", (name, "all"), (name, "rgb"))


def variety_block(meta998, device: torch.device) -> Block:
    """Variety (4 classes) on all 998 leaves: logistic regression and the 12 CNNs, 204 bands and RGB."""
    y = meta998.variety.map(VARIETIES.index).to_numpy()
    panel = panel_of(np.arange(N_LEAVES), y, disease_lr.fold_of(meta998), len(VARIETIES), device)
    block = Block("Variety (4 classes), all 998 leaves", panel, [BASELINE, *CNN_NAMES])
    for b in BANDS:
        block.scores[(BASELINE, b)] = score_model(panel, variety_lr(meta998, b))
        for (bb, pool), name in zip(CNNS, CNN_NAMES):
            P = variety_cnn(bb, pool, b)
            block.scores[(name, b)] = "pending" if P is None else score_model(panel, P)
    single_task_rows(block, pooled=False)
    return block


@dataclass(frozen=True)
class Setting:
    """One disease setting: which leaves the models trained on (source) and which leaves are scored."""

    task: str
    title: str
    source: str                 # a variety, "all" or "baco_chardonnay_vidal" (as disease --variety)
    scored: tuple[str, ...]     # varieties whose leaves are scored

    @property
    def pooled(self) -> bool:
        """True when the models saw several varieties."""
        return self.source not in WITHIN


# Leafroll never uses Cab Franc (1 leaf with leafroll), so "all 998" and "without Cab Franc" are one model.
LEAFROLL_POOLED = Setting("leafroll", "pooled, Baco + Chardonnay + Vidal (leafroll never uses Cab Franc)", "all", BCV)
RED_BLOTCH_ALL = Setting("red_blotch", "pooled, all 998 leaves", "all", tuple(VARIETIES))
RED_BLOTCH_WITHOUT_CF = Setting("red_blotch", "pooled, trained without Cab Franc (Baco + Chardonnay + Vidal)",
                                "baco_chardonnay_vidal", BCV)
RED_BLOTCH_ALL_SCORED_WITHOUT_CF = Setting(
    "red_blotch", "pooled, all 998 leaves, scored without Cab Franc (same models as all 998)", "all", BCV)
SETTINGS = [*(Setting("leafroll", v, v, (v,)) for v in WITHIN), LEAFROLL_POOLED,
            *(Setting("red_blotch", v, v, (v,)) for v in WITHIN), RED_BLOTCH_ALL, RED_BLOTCH_WITHOUT_CF,
            RED_BLOTCH_ALL_SCORED_WITHOUT_CF]


def disease_blocks(meta998, device: torch.device) -> list[Block]:
    """Every disease setting: logistic regression, the variety-only rule (pooled), the 12 CNNs; both band sets.

    Settings that score the same leaves for the same task share one panel (one set of resamples), so a model
    trained without Cab Franc and the all-998 model scored without Cab Franc are paired.

    :return: The blocks, in SETTINGS order.
    """
    group = disease_lr.row_groups(meta998)   # variety × row; for one variety simply the row
    panels, base, rules, blocks = {}, {}, {}, []
    for s in SETTINGS:
        y = disease_lr.task_labels(meta998, s.task)
        idx = np.flatnonzero(meta998.variety.isin(s.scored).to_numpy())
        key = (s.task, s.scored)
        if key not in panels:
            panels[key] = panel_of(idx, y, group, 2, device)
        panel = panels[key]
        meta = disease_lr.load_meta(s.source)
        to998 = positions(meta998, meta)
        block = Block(f"{TASK_NAME[s.task]}, {s.title}", panel,
                      [BASELINE, *([RULE] if s.pooled else []), *CNN_NAMES])
        # Each model's P over the 998 leaves (NaN where it made none), then the panel's leaves.
        on_panel = lambda p998: p998[..., panel.idx]   # noqa: E731
        for b in BANDS:
            if (s.source, s.task, b) not in base:
                p = np.full(N_LEAVES, np.nan)
                p[to998] = disease_lr.run_baselines(meta, disease_lr.load_features(s.source, device, b), [s.task])[s.task]
                base[(s.source, s.task, b)] = p
            block.scores[(BASELINE, b)] = score_model(panel, on_panel(base[(s.source, s.task, b)])[None])
            for (bb, pool), name in zip(CNNS, CNN_NAMES):
                P = disease_cnn(s.source, s.task, bb, pool, b, to998)
                block.scores[(name, b)] = "pending" if P is None else score_model(panel, on_panel(P))
        if s.pooled:
            if (s.source, s.task) not in rules:
                p = np.full(N_LEAVES, np.nan)
                p[to998] = variety_rule(meta, s.task)
                rules[(s.source, s.task)] = p
            block.scores[(RULE, "all")] = score_model(panel, on_panel(rules[(s.source, s.task)])[None])
        for name in block.models:   # every scored leaf must have a prediction
            for b in BANDS:
                sc = block.scores.get((name, b))
                assert not isinstance(sc, Score) or np.isfinite(sc.ba), f"{block.title}: {name} {b}"
        single_task_rows(block, s.pooled)
        blocks.append(block)
    return blocks


# The four MOO tasks: which heads, which checkpoint choice, which disease setting gives the panel and references.
MOO_TASKS = [
    ("variety", "Variety (4 classes), all 998 leaves", None),
    ("leafroll", "Leafroll, Baco + Chardonnay + Vidal (Cab Franc masked)", LEAFROLL_POOLED),
    ("red_blotch", "Red blotch, all 998 leaves", RED_BLOTCH_ALL),
    ("red_blotch", "Red blotch, all 998 leaves, scored without Cab Franc", RED_BLOTCH_ALL_SCORED_WITHOUT_CF),
]


def moo_blocks(meta998, variety: Block, disease: list[Block], device: torch.device) -> tuple[list[Block], dict]:
    """Six-head models on each task, beside the single-task references on the same panel.

    Each task is scored at its own checkpoint (the epoch with that task's lowest validation loss); the
    one-checkpoint score (all six heads' summed loss) is returned apart, for its own table.

    :return: The blocks (MOO_TASKS order) and {(model, bands, task index): Score} at the one checkpoint.
    """
    paths = meta998.path.to_numpy()
    runs = {(m, b): moo_run(*m, b, paths) for m in MOO_MODELS for b in BANDS}
    blocks, one_checkpoint = [], {}
    for t, (task, title, setting) in enumerate(MOO_TASKS):
        ref = variety if setting is None else disease[SETTINGS.index(setting)]
        heads = MOO_CHECKPOINTS[task]
        single = [cnn_name(nw, "attention") for nw in NETWORKS]
        refs = [BASELINE, *([RULE] if setting is not None else []), *single]
        names = [moo_name(*m) for m in MOO_MODELS]
        block = Block(title, ref.panel, refs + names)
        block.scores.update({k: v for k, v in ref.scores.items() if k[0] in refs})
        for m, name in zip(MOO_MODELS, names):
            for b in BANDS:
                r = runs[(m, b)]
                if r is None:
                    block.scores[(name, b)] = "pending"
                    continue
                for choice, store in ((task, block.scores), ("all", one_checkpoint)):
                    P = r[choice][:, ref.panel.idx][..., heads]
                    P = P[..., 0] if len(heads) == 1 else P
                    n_bad = int(np.isnan(P).any(axis=tuple(range(1, P.ndim))).sum())
                    key = (name, b) if choice == task else (name, b, t)
                    # A run that diverged before its first improvement has no predictions (run_moo.py writes NaN).
                    store[key] = f"diverged ({n_bad} of 5 seeds)" if n_bad else score_model(ref.panel, P)
        for b in BANDS:
            for m in MOO_MODELS:
                if m[2] != "ls":
                    add_row(block, "5a", (moo_name(*m), b), (moo_name(m[0], m[1], "ls"), b))
            for nw in NETWORKS:
                for meth in METHODS:
                    add_row(block, "5b", (moo_name(nw, "sgd", meth), b), (moo_name(nw, "adamw", meth), b))
            for nw in NETWORKS:
                add_row(block, "6", (moo_name(nw, "adamw", "ls"), b), (cnn_name(nw, "attention"), b))
        if setting is not None:
            for b in BANDS:
                for name in names:
                    add_row(block, "1b", (name, b), (RULE, "all"))
        blocks.append(block)
    return blocks, one_checkpoint


def sgd_rates() -> dict:
    """The SGD learning rate chosen for each network and method on the validation rows (None if not yet)."""
    out = {}
    for nw in NETWORKS:
        for m in METHODS:
            f = MOO / f"{nw}_sgd" / f"sgd_lr_{m}.json"
            out[(nw, m)] = json.loads(f.read_text()) if f.exists() else None
    return out


def compute(device: torch.device) -> dict:
    """Score every model of the plan that has finished, and every planned comparison between them."""
    meta998 = load_dataset()[0]
    variety = variety_block(meta998, device)
    disease = disease_blocks(meta998, device)
    moo, one_checkpoint = moo_blocks(meta998, variety, disease, device)
    return {"variety": variety, "disease": disease, "moo": moo, "one_checkpoint": one_checkpoint,
            "sgd": sgd_rates()}


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------
def cell(sc, metric: str) -> str:
    """A score with its interval, or why there is none."""
    if not isinstance(sc, Score):
        return sc or "—"
    return score(getattr(sc, metric), getattr(sc, metric + "_ci"))


def score_table(block: Block) -> list[str]:
    """Each model's balanced accuracy and AUC with 95% intervals, 204 bands and RGB."""
    header = ["model"] + [f"{BANDS_NAME[b]}: {m}" for b in BANDS for m in ("bal. acc. [95% CI]", "AUC [95% CI]")]
    rows = []
    for name in block.models:
        if name == RULE:   # no spectrum: one score, shown under both band sets
            rows.append([name] + [cell(block.scores.get((RULE, "all")), m) for _b in BANDS for m in ("ba", "auc")])
            continue
        rows.append([name] + [cell(block.scores.get((name, b)), m) for b in BANDS for m in ("ba", "auc")])
    return table(header, rows)


def diff_cells(d: Diff) -> list[str]:
    """Δ [95% CI], the seed count and the mark of one score."""
    seeds = f"{d.wins} of 5" if d.seeded else "single fits"
    return [delta(d.value, d.ci), seeds, d.mark]


def count_line(rows: list[Row]) -> str:
    """How many comparisons were marked, against the 1 in 20 an interval alone would mark by chance."""
    n = len(rows)
    c = {m: {s: sum(getattr(r, m).mark == s for r in rows) for s in "✓✗"} for m in ("ba", "auc")}
    return (f"Marked, of {n} comparisons: balanced accuracy {c['ba']['✓']} ✓ and {c['ba']['✗']} ✗; AUC "
            f"{c['auc']['✓']} ✓ and {c['auc']['✗']} ✗. Chance alone would put about {n / 20:.1f} intervals per "
            "score off 0 (fewer once the seeds must agree).")


def comparison_table(rows: list[Row]) -> list[str]:
    """Every comparison of a block, in the plan's order: both models' scores, A − B, seeds, mark."""
    header = ["#", "A − B", "bands", "bal. acc. A vs B", "Δ [95% CI]", "seeds A > B", "",
              "AUC A vs B", "Δ [95% CI]", "seeds A > B", ""]
    lines = [[r.family, f"{r.a} − {r.b}", r.bands, f"{r.score_a.ba:.3f} vs {r.score_b.ba:.3f}", *diff_cells(r.ba),
              f"{r.score_a.auc:.3f} vs {r.score_b.auc:.3f}", *diff_cells(r.auc)] for r in rows]
    return table(header, lines)


def families_note(families: list[str]) -> list[str]:
    """What each comparison number in the # column means."""
    return ["Comparisons (# column): " + "; ".join(f"**{f}** {FAMILY[f]}" for f in families) + ".", ""]


def block_section(block: Block, level: str = "###") -> list[str]:
    """One setting: scores, then every comparison with its count line."""
    lines = [f"{level} {block.title}", ""] + score_table(block) + [""]
    if block.rows:
        families = sorted({r.family for r in block.rows}, key=list(FAMILY).index)
        lines += families_note(families) + comparison_table(block.rows) + ["", count_line(block.rows), ""]
    return lines


def overview(blocks: list[Block], task: str, bands: str) -> list[str]:
    """Balanced accuracy · AUC of every single-task model in every setting of one task, one band set."""
    blocks = [bl for bl in blocks if bl.title.startswith(TASK_NAME[task])]
    names = list(dict.fromkeys(n for bl in blocks for n in bl.models))
    short = [bl.title.split(", ", 1)[1].split(" (")[0] for bl in blocks]

    def pair(bl: Block, name: str) -> str:
        sc = bl.scores.get((name, "all" if name == RULE else bands))
        return f"{sc.ba:.3f} · {sc.auc:.3f}" if isinstance(sc, Score) else (sc or "")
    return table(["model", *short], [[n, *(pair(bl, n) for bl in blocks)] for n in names])


def summary(blocks: list[Block]) -> list[str]:
    """Marks per comparison type over a category's blocks."""
    fam = sorted({r.family for bl in blocks for r in bl.rows}, key=list(FAMILY).index)
    out = []
    for f in fam:
        rows = [r for bl in blocks for r in bl.rows if r.family == f]
        c = {m: {s: sum(getattr(r, m).mark == s for r in rows) for s in "✓✗"} for m in ("ba", "auc")}
        out.append([f, FAMILY[f], str(len(rows)), f"{len(rows) / 20:.1f}", str(c["ba"]["✓"]), str(c["ba"]["✗"]),
                    str(c["auc"]["✓"]), str(c["auc"]["✗"])])
    return table(["#", "comparison", "rows", "chance (1 in 20)", "bal. acc. ✓", "bal. acc. ✗", "AUC ✓", "AUC ✗"], out)


def marked(blocks: list[Block]) -> list[str]:
    """Every marked comparison of a category, in one list."""
    rows = [[bl.title, r.family, f"{r.a} − {r.b}", r.bands, f"{delta(r.ba.value, r.ba.ci)} {r.ba.mark}",
             f"{delta(r.auc.value, r.auc.ci)} {r.auc.mark}"]
            for bl in blocks for r in bl.rows if r.ba.mark or r.auc.mark]
    if not rows:
        return ["No comparison is marked."]
    return table(["setting", "#", "A − B", "bands", "Δ bal. acc. [95% CI]", "Δ AUC [95% CI]"], rows)


def pending(r: dict) -> list[str]:
    """The models of the plan that have not finished yet (left out of every table and count)."""
    blocks = [r["variety"], *r["disease"], *r["moo"]]
    missing = sorted({f"{k[0]} ({BANDS_NAME[k[1]]})" for bl in blocks for k, v in bl.scores.items() if v == "pending"})
    if not missing:
        return []
    return [f"**Not finished yet ({len(missing)} models; left out of every table and count):** "
            + "; ".join(missing) + ".", ""]


def one_checkpoint_table(r: dict, bands: str) -> list[str]:
    """Six-head models scored at one checkpoint (all six heads' summed validation loss), every task."""
    header = ["model"] + [title for _t, title, _s in MOO_TASKS]
    rows = []
    for m in MOO_MODELS:
        name = moo_name(*m)
        cells = []
        for t in range(len(MOO_TASKS)):
            sc = r["one_checkpoint"].get((name, bands, t))
            cells.append(f"{sc.ba:.3f} · {sc.auc:.3f}" if isinstance(sc, Score) else (sc or "pending"))
        rows.append([name, *cells])
    return table(header, rows)


def sgd_table(rates: dict) -> list[str]:
    """The SGD learning rate each method got, per network."""
    rows = [[METHODS[m]] + [("pending" if rates[(nw, m)] is None else f"{rates[(nw, m)]['chosen_lr']:g}")
                            for nw in NETWORKS] for m in METHODS]
    return table(["method", *(f"six-head {LABEL[nw]}" for nw in NETWORKS)], rows)


def to_markdown(r: dict) -> str:
    """RESULTS.md: how to read it, a summary of the marks, then variety, disease and MOO in full."""
    disease, moo = r["disease"], r["moo"]
    lines = [
        "# Results", "",
        "Built by `make_results.py` from the saved predictions; the plan, fixed before the runs, is "
        "[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md).", "",
        "**How to read it.** Each score is the seed ensemble (the mean probability of 5 seeds; logistic "
        "regression and the variety-only rule are one fit) on the held-out rows, with a 95% interval from 2000 "
        "resamples of leaves within each row × class (× variety when pooled). Balanced accuracy: chance 0.5 for a "
        "disease, 0.25 for variety. AUC: chance 0.5. Each comparison A − B is paired (the same resamples for both "
        "models). It gets ✓ (A better) or ✗ (A worse) only when its 95% interval excludes 0 **and** A beats "
        "(or loses to) B in at least 4 of the 5 seeds trained separately (seed s against seed s; a CNN seed "
        "against the one logistic-regression fit). Two single fits (logistic regression 204 bands − RGB) have "
        "only the interval and the sign. Unmarked rows are ties within the noise.", "",
        *pending(r),
        "## Contents", "",
        "1. [Summary of the marks](#summary-of-the-marks)",
        "2. [Variety](#variety)",
        "3. [Disease](#disease): " + ", ".join(f"[{TASK_NAME[t]}](#{t.replace('_', '-')}-overview)"
                                               for t in TASK_NAME),
        "4. [MOO (six heads)](#moo-six-heads)", "",
        "## Summary of the marks", "",
        "### Variety", "", *summary([r["variety"]]), "",
        "### Disease (every setting)", "", *summary(disease), "",
        "### MOO", "", *summary(moo), "",
        "### Every marked comparison", "",
        "Variety:", "", *marked([r["variety"]]), "",
        "Disease:", "", *marked(disease), "",
        "MOO:", "", *marked(moo), "",
        "## Variety", "", *block_section(r["variety"], "###"),
        "## Disease", "",
        "Leafroll leaves out Cab Franc everywhere (1 of its 250 leaves has leafroll). Red blotch on all 998 is "
        "also scored without Cab Franc's leaves (241 of its 250 leaves have red blotch), with the same models. "
        "The **variety-only rule** predicts from the leaf's variety alone (each variety's disease rate on the "
        "training rows); a pooled model has to beat it to show more than the varieties' different disease rates.",
        "",
    ]
    for task in TASK_NAME:
        lines += [f"### {TASK_NAME[task]} overview", "",
                  "Balanced accuracy · AUC of every model in every setting; intervals and comparisons in the "
                  "sections below.", ""]
        for b in BANDS:
            lines += [f"{BANDS_NAME[b]}:", "", *overview(disease, task, b), ""]
    for bl in disease:
        lines += block_section(bl, "###")
    lines += ["## MOO (six heads)", "",
              "One network with six outputs (4 varieties, red blotch, leafroll; Cab Franc masked for leafroll), "
              "small CNN or small ResNet with attention pooling, AdamW or SGD, 8 ways to combine the six "
              "gradients. Each task is scored at its own best validation epoch; the single-task CNNs, logistic "
              "regression and the variety-only rule are the same rows as above, on the same resamples.", "",
              "SGD learning rate chosen per method on 204 bands (seed 0, every fold, lowest mean validation "
              "loss of the six heads); RGB reuses it:", "", *sgd_table(r["sgd"]), ""]
    for bl in moo:
        lines += block_section(bl, "###")
    lines += ["### One checkpoint for all six heads", "",
              "Balanced accuracy · AUC when one epoch (the lowest summed validation loss of all six heads) "
              "serves every task. No comparisons; shown beside the per-task checkpoints above.", ""]
    for b in BANDS:
        lines += [f"{BANDS_NAME[b]}:", "", *one_checkpoint_table(r, b), ""]
    return "\n".join(lines)


def main() -> None:
    """Score everything that has finished → RESULTS.md."""
    log = get_logger("make_results")
    r = compute(get_device())
    OUT.write_text(to_markdown(r))
    n_rows = sum(len(bl.rows) for bl in [r["variety"], *r["disease"], *r["moo"]])
    log.info("[saved] %s: %d comparisons", OUT.name, n_rows)


if __name__ == "__main__":
    main()
