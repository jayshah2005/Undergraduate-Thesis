"""Embed 2026 virus labels into ONE HDF5 file: /<Variety>/<Row>/V<n>/{cube,preview_png}.

Usage (from labelling/): ../.venv/bin/python build_h5_single.py [--out vine_hsi.h5]

Label mapping (virus 1 / virus 2):
  label[0] = red_blotch (GRBV)  — Excel: Baco/Vidal col E (2026), Cab Franc/Chardonnay col C
  label[1] = leafroll3 (LR3)    — Excel: Baco/Vidal col H (2026), Cab Franc/Chardonnay col D
  Positive -> 1, Negative -> 0

Vines are keyed by folder path Data/<Variety>/<Row>/<n>/ (Excel Sample Name
e.g. Baco/R1/V1), not by the .hdr filename stem.

/wavelengths is the Specim IQ factory calibration read from the leaf .hdr files (every leaf
carries the same list). Each vine also keeps its capture's camera settings from the header.

Labels: this reads the interim sheet. After a rebuild, run update_labels_final_2026.py --h5 <file> --apply
to apply the FINAL 2026 retest labels from "New Testing Results.xlsx" (7 Vidal leaves differ).
"""
import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import openpyxl

ROOT = Path(__file__).parent
DATA = ROOT / "Data"
XLSX = ROOT / "Virus testing data for interim report.xlsx"

# Excel sheet -> (data folder name, sample-name prefix, 2026 column letters for [virus1, virus2])
VARIETIES = {
    "Baco": ("Baco", "Baco", ("E", "H")),
    "Vidal": ("Vidal", "Vidal", ("E", "H")),
    "Cab Franc": ("Cab Franc", "CF", ("C", "D")),
    "Chardonnay": ("Chardonnay", "Chard", ("C", "D")),
}
CAPTURE_DATES = {
    "Baco": "2026-08-12",
    "Chardonnay": "2026-08-26",
    "Cab Franc": "2026-09-03",
    "Vidal": "2026-09-03",
}
LABEL_NAMES = ["red_blotch", "leafroll3"]  # virus 1 = Red Blotch (GRBV), virus 2 = Leafroll-3 (LR3)


def to_bin(value, where):
    if value not in ("Positive", "Negative"):
        raise ValueError(f"unexpected label {value!r} at {where}")
    return int(value == "Positive")


def read_labels(wb):
    """{(variety, row, vine_number): (ccovi_code, [v1, v2])}"""
    labels = {}
    for variety, (_, prefix, cols) in VARIETIES.items():
        ws = wb[variety]
        for r in range(3, ws.max_row + 1):
            name = ws[f"B{r}"].value
            if not name:
                continue
            pfx, row, vine = name.split("/")
            assert pfx == prefix, f"{variety}!B{r}: {name}"
            where = f"{variety}!{r}"
            labels[(variety, row, int(vine.lstrip("V")))] = (
                ws[f"A{r}"].value,
                [to_bin(ws[f"{c}{r}"].value, where) for c in cols],
            )
    return labels


def read_hdr(path):
    """ENVI header -> {field: value}; braced lists become lists of strings (description stays text)."""
    fields = {}
    for key, val in re.findall(r"^[ \t]*([^=\n{}]+?)[ \t]*=[ \t]*(\{[^}]*\}|[^\n]*)", path.read_text(), re.M):
        key = key.strip().lower()
        if val.startswith("{"):
            inner = val[1:-1].strip()
            val = inner if key == "description" else [s.strip() for s in inner.split(",") if s.strip()]
        fields[key] = val if isinstance(val, list) else val.strip()
    return fields


def parse_hdr(path):
    """(lines, bands, samples) and the header fields; leaves are float32 BIL little-endian."""
    h = read_hdr(path)
    assert h["data type"] == "4" and h["interleave"].lower() == "bil" and h["byte order"] == "0", path
    assert len(h.get("wavelength", [])) == int(h["bands"]), f"{path}: no wavelength list (run restore_leaf_headers.py)"
    return int(h["lines"]), int(h["bands"]), int(h["samples"]), h


def camera_attrs(h):
    """Capture settings from the Specim IQ header. The camera's clock is not reliable (some dates
    are weeks after the files existed), so the field date stays in capture_date."""
    out = {}
    if "acquisition date" in h:
        d, m, y = h["acquisition date"].split("-")
        out["camera_acquisition_date"] = f"{y}-{m}-{d}"
    if "tint" in h:
        out["tint_ms"] = float(h["tint"])
    if "fps" in h:
        out["fps"] = float(h["fps"])
    return out


def band_info(hdrs):
    """Wavelengths (nm) and default RGB bands (1-based, ENVI convention) shared by every leaf."""
    first = read_hdr(hdrs[0])
    for hdr in hdrs[1:]:
        h = read_hdr(hdr)
        assert h["wavelength"] == first["wavelength"] and h["default bands"] == first["default bands"], hdr
    return np.array(first["wavelength"], dtype=np.float32), np.array(first["default bands"], dtype=np.int16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "vine_hsi.h5"))
    args = ap.parse_args()

    labels = read_labels(openpyxl.load_workbook(XLSX, data_only=True))
    used, no_label, stem_mismatch = set(), [], []
    wavelengths, default_bands = band_info(sorted(DATA.glob("*/R*/*/*.hdr")))

    with h5py.File(args.out, "w") as f:
        f.attrs.update(
            label_names=LABEL_NAMES,
            label_year=2026,
            class_names=["Negative", "Positive"],
            sensor="Specim IQ",
            sensor_docs="docs/Specim-IQ-Datasheet.pdf",
            layout="cube is (lines, bands, samples) as stored in the ENVI BIL files",
            default_bands=default_bands,
            default_bands_note="R, G, B band numbers from the Specim IQ header, 1-based (ENVI convention)",
        )
        wl = f.create_dataset("wavelengths", data=wavelengths)
        wl.attrs.update(
            units="nm", sensor="Specim IQ", wavelengths_source="factory",
            note="Specim IQ factory calibration, from the capture .hdr files (identical for every leaf)",
        )

        for variety, (folder, _, _) in VARIETIES.items():
            vg = f.create_group(variety)
            vg.attrs["capture_date"] = CAPTURE_DATES[variety]
            for hdr in sorted((DATA / folder).glob("R*/*/*.hdr"), key=lambda p: (p.parts[-3], int(p.parent.name))):
                row, n = hdr.parts[-3], int(hdr.parent.name)
                key = (variety, row, n)
                if key not in labels:
                    no_label.append(key)
                    continue
                if hdr.stem != str(n):
                    stem_mismatch.append((key, hdr.name))
                used.add(key)
                code, lab = labels[key]
                lines, bands, samples, h = parse_hdr(hdr)
                cube = np.fromfile(hdr.with_suffix(".img"), dtype="<f4").reshape(lines, bands, samples)
                g = vg.require_group(f"{row}/V{n}")
                d = g.create_dataset(
                    "cube", data=cube, compression="gzip", compression_opts=4,
                    chunks=(min(lines, 32), bands, min(samples, 32)),
                )
                d.attrs["interleave"] = "bil"
                d.attrs["data_ignore_value"] = float(h.get("data ignore value", 0))  # background outside the leaf
                g.attrs.update(
                    label=np.array(lab, dtype=np.int8),
                    red_blotch=lab[0], leafroll3=lab[1],
                    ccovi_code=code, variety=variety, row=row, vine=n,
                    capture_date=CAPTURE_DATES[variety],
                    source_path=str(hdr.with_suffix(".img").relative_to(ROOT)),
                    hdr_stem=hdr.stem,
                    **camera_attrs(h),
                )
                png = hdr.with_suffix(".png")
                if png.exists():
                    g.create_dataset("preview_png", data=np.frombuffer(png.read_bytes(), dtype=np.uint8))

    no_cube = sorted(set(labels) - used)
    print(f"wrote {len(used)} leaves to {args.out}")
    print(f"cubes without a label ({len(no_label)}): {no_label}")
    print(f"labels without a cube ({len(no_cube)}): {no_cube}")
    if stem_mismatch:
        print(f"warning: {len(stem_mismatch)} hdr stems != vine folder "
              f"(labels still keyed by folder): {stem_mismatch}")


if __name__ == "__main__":
    main()
