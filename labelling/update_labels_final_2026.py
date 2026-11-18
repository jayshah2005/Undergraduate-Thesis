"""Write the FINAL 2026 virus labels from "New Testing Results.xlsx" into an existing vine_hsi.h5.

Usage (from labelling/): ../.venv/bin/python update_labels_final_2026.py --h5 ../vine_hsi.h5 [--apply]

Without --apply it only prints what would change. The cubes are not touched: only the per-vine
attributes red_blotch, leafroll3 and label (= [red_blotch, leafroll3]) change. The values they
replace stay as red_blotch_interim / leafroll3_interim (the labels from "Virus testing data for
interim report.xlsx"), and the file gets label_source / label_note.

Sheet layout ("Sheet1"): column B = Sample Name (e.g. Baco/R1/V1, Chard/..., CF/...), column E =
Red Blotch "FINAL 2026", column M = Leafroll-3 "FINAL 2026". Row 254 repeats the header. Columns F-I and
N-Q are the record of the original 2026 test and the retests (Baco and Vidal rows R1-R3 only); the colour of a
FINAL cell gives the reason (row 1 legend: positive on retest, negative on retest, decided positive from
previous years and flanking vines). Side tables in columns S-U are not read: rows 255-304 are an extra Vidal
row with no images, rows 307-370 a retest list (it differs from FINAL only where the lab decided positive,
plus Vidal/R1/V39, where it disagrees with column I).
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import openpyxl

ROOT = Path(__file__).parent
XLSX = ROOT / "New Testing Results.xlsx"
PREFIX = {"Baco": "Baco", "Vidal": "Vidal", "Chard": "Chardonnay", "CF": "Cab Franc"}
COL_RB, COL_LR = 5, 13   # E and M: FINAL 2026
SOURCE = "New Testing Results.xlsx, FINAL 2026 columns (Red Blotch col E, Leafroll-3 col M)"


def to_bin(value, where):
    v = str(value).strip().lower()
    if v not in ("positive", "negative"):
        raise ValueError(f"unexpected label {value!r} at {where}")
    return int(v == "positive")


def read_final(path):
    """{(variety, row, vine): (ccovi_code, red_blotch, leafroll3)} from the FINAL 2026 columns."""
    ws = openpyxl.load_workbook(path, data_only=True)["Sheet1"]
    for header in (3, 254):   # the header is repeated above the Vidal rows
        for col in (COL_RB, COL_LR):
            assert "FINAL" in str(ws.cell(header, col).value), (header, col, ws.cell(header, col).value)
    assert ws.cell(2, 3).value == "Red Blotch" and ws.cell(2, 11).value == "Leafroll-3"
    labels = {}
    for r in range(4, ws.max_row + 1):
        name = ws.cell(r, 2).value
        if not name or name == "Sample Name":
            continue
        pfx, row, vine = name.strip().split("/")
        key = (PREFIX[pfx], row, int(vine.lstrip("V")))
        assert key not in labels, f"duplicate {name} at row {r}"
        labels[key] = (str(ws.cell(r, 1).value).strip(), to_bin(ws.cell(r, COL_RB).value, f"E{r}"),
                       to_bin(ws.cell(r, COL_LR).value, f"M{r}"))
    assert len(labels) == 1000 and all(n == 250 for n in
                                       np.unique([k[0] for k in labels], return_counts=True)[1]), len(labels)
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    args = ap.parse_args()

    final = read_final(XLSX)
    changes, n = [], 0
    with h5py.File(args.h5, "r+" if args.apply else "r") as f:
        for variety in f:
            if variety == "wavelengths":
                continue
            for row in f[variety]:
                for vine in f[variety][row]:
                    g = f[variety][row][vine]
                    key = (variety, row, int(vine.lstrip("V")))
                    code, rb, lr = final[key]
                    assert code == str(g.attrs["ccovi_code"]).strip(), (key, code, g.attrs["ccovi_code"])
                    old = (int(g.attrs["red_blotch"]), int(g.attrs["leafroll3"]))
                    assert list(g.attrs["label"]) == list(old), key
                    n += 1
                    for name, a, b in (("red_blotch", old[0], rb), ("leafroll3", old[1], lr)):
                        if a != b:
                            changes.append(f"{variety}/{row}/{vine} {name} {a} -> {b}")
                    if args.apply:
                        # Keep the interim value once; a second run must not overwrite it with FINAL.
                        if "red_blotch_interim" not in g.attrs:
                            g.attrs["red_blotch_interim"], g.attrs["leafroll3_interim"] = old
                        g.attrs["red_blotch"], g.attrs["leafroll3"] = rb, lr
                        g.attrs["label"] = np.array([rb, lr], dtype=np.int8)
        if args.apply:
            f.attrs["label_source"] = SOURCE
            f.attrs["label_note"] = ("red_blotch_interim / leafroll3_interim on each vine keep the labels from "
                                     "'Virus testing data for interim report.xlsx'")
    print(f"{n} leaves checked; {len(changes)} label changes{' written' if args.apply else ' (dry run)'}:")
    print("\n".join(changes))


if __name__ == "__main__":
    main()
