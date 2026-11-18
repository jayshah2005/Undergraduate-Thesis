"""Review the virus labels by hand: both lab sheets and vine_hsi.h5 side by side, then stop at a breakpoint.

Usage (from labelling/):
    ../.venv/bin/python review_labels.py                  # checks + summary, then a breakpoint to explore
    ../.venv/bin/python review_labels.py --no-break       # checks + summary only
    ../.venv/bin/python review_labels.py --h5 vine_hsi.h5 # review another copy of the HDF5

At the breakpoint (pdb):
    df[df.changed]          the leaves whose label changed (df has one row per vine in the sheet, 1000)
    show("Vidal/R3/V5")     every test and label for one leaf
    look("Vidal/R3/V5")     its preview image and mean spectrum
    interact                a full Python prompt (Ctrl-D goes back to pdb); c ends the script

Sources: "New Testing Results.xlsx" (FINAL 2026 = columns E and M, the lab's decision; F-I and N-Q are the
record of the original 2026 test and the retests), "Virus testing data for interim report.xlsx" (the
labels used before 2026-11-17) and the HDF5, whose red_blotch / leafroll3 attributes are what the models
train on (red_blotch_interim / leafroll3_interim keep the old values).
"""
from __future__ import annotations

import argparse
import io
import tempfile
from pathlib import Path

import h5py
import numpy as np
import openpyxl
import pandas as pd

from build_h5_single import XLSX as INTERIM_XLSX, read_labels as read_interim_labels
from update_labels_final_2026 import PREFIX, XLSX as FINAL_XLSX

ROOT = Path(__file__).parent
CLASSIFICATION = ROOT.parent
H5_COPIES = [CLASSIFICATION / "vine_hsi.h5", ROOT / "vine_hsi.h5"]   # the analysis reads the first
CACHE = CLASSIFICATION / "cache"

# Sheet1 column numbers per virus: 2024, 2025, FINAL 2026, original 2026 test, then the three retests.
FIELDS = ["2024", "2025", "final", "original", "retest_buffer", "retest_leaf_1", "retest_leaf_2"]
COLUMNS = {"rb": dict(zip(FIELDS, range(3, 10))), "lr": dict(zip(FIELDS, range(11, 18)))}
VIRUS = {"rb": "red blotch", "lr": "leafroll-3"}

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 30)


def read_sheet(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """One row per vine of Sheet1, every test cell as written, plus the reason the FINAL cell's colour gives.

    :param path: "New Testing Results.xlsx".
    :return: the table and {column: header text in the sheet}, e.g. "rb_retest_buffer": "2026 buffer retest".
    """
    ws = openpyxl.load_workbook(path, data_only=True)["Sheet1"]
    # Row 1 is the colour legend: cells A1-C1 are filled with the colour they explain.
    legend = {ws.cell(1, c).fill.fgColor.rgb: str(ws.cell(1, c).value).strip() for c in (1, 2, 3)}
    headers = {f"{v}_{f}": " ".join(str(ws.cell(3, c).value).split()) for v, cols in COLUMNS.items()
               for f, c in cols.items()}
    rows = []
    for r in range(4, ws.max_row + 1):
        name = ws.cell(r, 2).value
        if not name or name == "Sample Name":   # blank rows and the header repeated above Vidal
            continue
        prefix, row, vine = name.strip().split("/")
        rec = {"path": f"{PREFIX[prefix]}/{row}/V{int(vine.lstrip('V'))}", "variety": PREFIX[prefix], "row": row,
               "vine": int(vine.lstrip("V")), "ccovi": str(ws.cell(r, 1).value).strip(), "sheet_row": r}
        for v, cols in COLUMNS.items():
            for f, c in cols.items():
                value = ws.cell(r, c).value
                rec[f"{v}_{f}"] = "" if value is None else str(value).strip()
            colour = ws.cell(r, cols["final"]).fill.fgColor
            rec[f"{v}_reason"] = legend.get(colour.rgb, "") if colour.type == "rgb" else ""
        rows.append(rec)
    return pd.DataFrame(rows), headers


def read_h5_labels(h5: h5py.File) -> pd.DataFrame:
    """The label attributes of every leaf in the HDF5 (998: Baco R3/V9 and V10 have no image)."""
    rows = []
    for variety in h5:
        if variety == "wavelengths":
            continue
        for row in h5[variety]:
            for vine in h5[variety][row]:
                a = h5[variety][row][vine].attrs
                rows.append({"path": f"{variety}/{row}/{vine}", "h5_ccovi": str(a["ccovi_code"]).strip(),
                             "rb_h5": int(a["red_blotch"]), "lr_h5": int(a["leafroll3"]),
                             "rb_h5_interim": int(a["red_blotch_interim"]), "lr_h5_interim": int(a["leafroll3_interim"]),
                             "label_attr_ok": list(a["label"]) == [int(a["red_blotch"]), int(a["leafroll3"])]})
    return pd.DataFrame(rows)


def build_table(h5: h5py.File) -> tuple[pd.DataFrame, dict[str, str]]:
    """Sheet, interim sheet and HDF5 joined on the leaf path; one row per vine in the sheet."""
    df, headers = read_sheet(FINAL_XLSX)
    interim = read_interim_labels(openpyxl.load_workbook(INTERIM_XLSX, data_only=True))
    df["rb_interim_sheet"] = [interim[(v, r, n)][1][0] for v, r, n in zip(df.variety, df.row, df.vine)]
    df["lr_interim_sheet"] = [interim[(v, r, n)][1][1] for v, r, n in zip(df.variety, df.row, df.vine)]
    for v in COLUMNS:
        df[f"{v}_final_int"] = (df[f"{v}_final"].str.lower() == "positive").astype(int)
    df = df.merge(read_h5_labels(h5), on="path", how="left")
    df["image"] = df.rb_h5.notna()
    for c in ["rb_h5", "lr_h5", "rb_h5_interim", "lr_h5_interim"]:
        df[c] = df[c].astype("Int64")   # missing for the 2 vines without an image
    df["changed"] = (df.image & ((df.rb_h5 != df.rb_h5_interim) | (df.lr_h5 != df.lr_h5_interim)).fillna(False)).astype(bool)
    return df, headers


def run_checks(df: pd.DataFrame, h5_path: Path) -> list[tuple[str, list[str]]]:
    """Each check's name and the leaves that fail it (empty = pass)."""
    img = df[df.image]
    fail = lambda mask: list(img.path[mask])
    checks = [
        ("FINAL 2026 cells are all Positive or Negative",
         list(df.path[~df.rb_final.str.lower().isin(["positive", "negative"]) |
                      ~df.lr_final.str.lower().isin(["positive", "negative"])])),
        ("HDF5 labels = FINAL 2026 (columns E and M)",
         fail((img.rb_h5 != img.rb_final_int) | (img.lr_h5 != img.lr_final_int))),
        ("HDF5 'label' array = [red_blotch, leafroll3]", fail(~img.label_attr_ok.astype(bool))),
        ("HDF5 *_interim = interim sheet",
         fail((img.rb_h5_interim != img.rb_interim_sheet) | (img.lr_h5_interim != img.lr_interim_sheet))),
        ("CCOVI codes agree (sheet vs HDF5)", fail(img.ccovi != img.h5_ccovi)),
    ]
    # Every change should be backed by a retest or a reason colour in the sheet.
    ch = img[img.changed]
    unexplained = [p for p, row in zip(ch.path, ch.itertuples()) if not any(
        getattr(row, f"{v}_{f}") for v in COLUMNS for f in ("retest_buffer", "retest_leaf_1", "retest_leaf_2", "reason"))]
    checks.append(("each change has a retest or a reason colour", unexplained))
    for other in H5_COPIES:
        if other.exists() and other.resolve() != h5_path.resolve():
            with h5py.File(other, "r") as h:
                o = read_h5_labels(h).set_index("path")
            mine = img.set_index("path")
            checks.append((f"other HDF5 copy ({other.parent.name}/{other.name}) has the same labels",
                           [p for p in mine.index if (o.loc[p, "rb_h5"], o.loc[p, "lr_h5"]) != (mine.loc[p, "rb_h5"], mine.loc[p, "lr_h5"])]))
    if (CACHE / "meta.parquet").exists():
        meta = pd.read_parquet(CACHE / "meta.parquet").set_index("path")
        mine = img.set_index("path")
        checks.append(("analysis cache (Classification/cache/meta.parquet) = HDF5",
                       [p for p in mine.index if p not in meta.index or
                        (meta.loc[p, "red_blotch"], meta.loc[p, "leafroll3"]) != (mine.loc[p, "rb_h5"], mine.loc[p, "lr_h5"])]))
    return checks


def print_summary(df: pd.DataFrame, checks: list[tuple[str, list[str]]]) -> None:
    """Checks, positives per variety (now vs before) and every changed leaf with its test history."""
    print("Checks:")
    for name, failed in checks:
        print(f"  {'PASS' if not failed else 'FAIL'}  {name}" + (f": {len(failed)} leaves, e.g. {failed[:5]}" if failed else ""))
    img = df[df.image]
    counts = img.groupby("variety").agg(leaves=("path", "size"),
                                        rb_now=("rb_h5", "sum"), rb_before=("rb_h5_interim", "sum"),
                                        lr_now=("lr_h5", "sum"), lr_before=("lr_h5_interim", "sum"))
    print("\nPositive leaves with an image (now = FINAL 2026, before = interim):")
    print(counts.to_string())
    retested = df[[any(df.loc[i, f"{v}_{f}"] for v in COLUMNS for f in ("retest_buffer", "retest_leaf_1", "retest_leaf_2"))
                   for i in df.index]]
    print("\nVines with a 2026 retest, per variety and row:",
          retested.groupby(["variety", "row"]).size().to_dict())
    print("\nLabels the retests changed (FINAL 2026 vs the first 2026 test), and whether the interim labels already had them:")
    for v in COLUMNS:
        moved = df[(df[f"{v}_final"].str.lower() != df[f"{v}_original"].str.lower())]
        for r in moved.itertuples():
            already = getattr(r, f"{v}_interim_sheet") == getattr(r, f"{v}_final_int")
            print(f"  {r.path:14s} {VIRUS[v]:10s} {getattr(r, f'{v}_original'):8s} -> {getattr(r, f'{v}_final'):8s} "
                  f"{'already in the interim labels' if already else 'NEW in FINAL 2026'}")
    print(f"\n{int(img.changed.sum())} changed leaves:")
    for r in img[img.changed].itertuples():
        for v in COLUMNS:
            before, now = getattr(r, f"{v}_h5_interim"), getattr(r, f"{v}_h5")
            if before != now:
                tests = ", ".join(f"{f} {getattr(r, f'{v}_{f}')}" for f in FIELDS if f != "final" and getattr(r, f"{v}_{f}"))
                print(f"  {r.path:14s} {VIRUS[v]:10s} {before} -> {now}   ({tests}; reason: {getattr(r, f'{v}_reason') or 'none given'})")


def show(path: str) -> None:
    """Print every test and label for one leaf, e.g. show("Vidal/R3/V5")."""
    r = DF.set_index("path").loc[path]
    print(f"{path}  (CCOVI {r.ccovi}, sheet row {r.sheet_row}, image: {'yes' if r.image else 'no'})")
    for v in COLUMNS:
        print(f"  {VIRUS[v]}:")
        for f in FIELDS:
            print(f"    {HEADERS[f'{v}_{f}']:28s} {r[f'{v}_{f}'] or '-'}")
        print(f"    {'reason (FINAL cell colour)':28s} {r[f'{v}_reason'] or '-'}")
        print(f"    {'interim sheet label':28s} {r[f'{v}_interim_sheet']}")
        if r.image:
            print(f"    {'HDF5 label now / before':28s} {r[f'{v}_h5']} / {r[f'{v}_h5_interim']}")


def look(path: str) -> None:
    """Show one leaf's preview image and mean spectrum next to its variety's mean, e.g. look("Vidal/R3/V5")."""
    import matplotlib
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    g = H5[path]
    wavelengths = np.asarray(H5["wavelengths"][:])
    # Stored as (lines, bands, samples); background pixels are 0 in every band.
    cube = np.nan_to_num(np.asarray(g["cube"], dtype=np.float32)).transpose(0, 2, 1)
    leaf = cube[(cube != 0).any(axis=2)]
    fig, (ax_img, ax_sp) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 2]})
    ax_img.imshow(mpimg.imread(io.BytesIO(np.asarray(g["preview_png"]).tobytes()), format="png"))
    ax_img.set_axis_off()
    ax_sp.plot(wavelengths, leaf.mean(0), color="C0", label=f"{path} ({len(leaf)} leaf pixels)")
    if (CACHE / "leaf_means.npz").exists():
        means = np.load(CACHE / "leaf_means.npz", allow_pickle=True)
        same = np.array([p.startswith(path.split("/")[0] + "/") for p in means["path"]])
        ax_sp.plot(wavelengths, means["mean"][same].mean(0), color="0.6", label=f"{path.split('/')[0]} mean")
    a = g.attrs
    ax_sp.set(xlabel="wavelength (nm)", ylabel="reflectance",
              title=f"red blotch {a['red_blotch']} (before {a['red_blotch_interim']}), "
                    f"leafroll {a['leafroll3']} (before {a['leafroll3_interim']})")
    ax_sp.legend()
    fig.tight_layout()
    if matplotlib.get_backend().lower() == "agg":   # no screen: save instead
        out = Path(tempfile.gettempdir()) / f"look_{path.replace('/', '_')}.png"
        fig.savefig(out, dpi=120)
        print(f"saved {out}")
    else:
        plt.show()   # close the window to get back to the prompt
    plt.close(fig)


def main() -> None:
    """Checks and summary, then a breakpoint with df, show() and look() ready to use."""
    global DF, HEADERS, H5
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", type=Path, default=H5_COPIES[0], help="HDF5 to review (default: the one the analysis reads)")
    ap.add_argument("--no-break", action="store_true", help="print the checks and summary, then exit")
    args = ap.parse_args()
    with h5py.File(args.h5, "r") as H5:
        print(f"HDF5: {args.h5}\n  label_source: {H5.attrs.get('label_source', '(not set)')}\n")
        DF, HEADERS = build_table(H5)
        df = DF
        changed = df[df.changed]
        print_summary(df, run_checks(df, args.h5))
        if not args.no_break:
            print("\nStopped at a breakpoint. Try:  df[df.changed]   show('Vidal/R3/V5')   look('Vidal/R3/V5')\n"
                  "Type `interact` for a Python prompt (Ctrl-D to leave it), `c` to finish.")
            breakpoint()
            pass   # <- stopped here: df, changed, show(), look() and H5 are ready


DF: pd.DataFrame
HEADERS: dict[str, str]
H5: h5py.File

if __name__ == "__main__":
    main()
