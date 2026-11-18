"""Restore the Specim IQ header information in Data/<Variety>/<Row>/<n>/<n>.hdr.

The leaf cubes were cut out of the REFLECTANCE_<id>.dat captures in VINO33/leaves_indoor by
tests/Indoor Segmentation/manual_ndvi_sam_segmentation.py (rotated 90 degrees, cropped, background
set to 0). That script used to write headers with only the layout (size / data type), dropping the
wavelengths, default RGB bands, acquisition date, camera settings, ...

This script finds every leaf's source capture by matching pixel values exactly, then writes a header
made from the capture's header + the leaf's layout -- the same header the fixed segmentation tools
now write (built with CODE/Segmentation's own `cutout_envi_metadata`). Only .hdr files are written.

Usage (needs numpy + spectral; CODE/Segmentation/.venv has both):
    ../../Segmentation/.venv/bin/python restore_leaf_headers.py --stage /tmp/staged        # match + stage
    ../../Segmentation/.venv/bin/python restore_leaf_headers.py --stage /tmp/staged \\
        --apply Data ../../Data ~/Desktop/Data                                              # copy into place

Writes ../helpers/leaf_sources.csv (leaf -> capture, capture id, orientation, leaf pixel count, mismatched values).
"""
import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from spectral.io import envi

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parents[1] / "Segmentation"))   # CODE/Segmentation
from backend.pipeline.export import cutout_envi_metadata  # noqa: E402  (same header as the app writes)

CAPTURES = Path.home() / "Desktop" / "VINO33" / "leaves_indoor"
VARIETIES = {  # Data folder -> capture folder (as in manual_ndvi_sam_segmentation.py)
    "Baco": "Aug12_2026_HSI_Baco_Row_18_17_16_15_14_leaves",
    "Chardonnay": "Aug262026_HSI_Chardonnay_leaves",
    "Cab Franc": "Sept03042026_HSI_CabFranc_leaves",
    "Vidal": "Sept03042026_HSI_Vidal_leaves",
}
SIDE, BANDS, KEY_BAND = 512, 204, 50
# 8 rotations/flips, as maps of a leaf offset (dy, dx) to a capture offset
ORIENTATIONS = {
    "identity": lambda a, b: (a, b), "flip_rows": lambda a, b: (-a, b),
    "flip_cols": lambda a, b: (a, -b), "rot180": lambda a, b: (-a, -b),
    "transpose": lambda a, b: (b, a), "rot90_cw": lambda a, b: (-b, a),
    "rot90_ccw": lambda a, b: (b, -a), "anti_transpose": lambda a, b: (-b, -a),
}
LAYOUT = ("samples", "lines", "bands", "header offset", "data type", "interleave", "byte order")


def capture_cube(path):
    return np.memmap(path, dtype="<f4", mode="r", shape=(SIDE, BANDS, SIDE))  # BIL: (y, band, x)


def leaf_cube(hdr):
    h = envi.read_envi_header(str(hdr))
    assert h["data type"] == "4" and h["interleave"].lower() == "bil" and h["byte order"] == "0", hdr
    shape = (int(h["lines"]), int(h["bands"]), int(h["samples"]))
    return h, np.memmap(hdr.with_suffix(".img"), dtype="<f4", mode="r", shape=shape)


def build_index(capture_dir):
    """Sorted band-50 values of every pixel of every capture -> (capture, y, x)."""
    caps = sorted(capture_dir.glob("*/results/REFLECTANCE_*.dat"), key=lambda p: int(p.parent.parent.name))
    keys, cap_idx, pos = [], [], []
    for i, p in enumerate(caps):
        plane = np.ascontiguousarray(capture_cube(p)[:, KEY_BAND, :]).view(np.uint32).ravel()
        keys.append(plane)
        cap_idx.append(np.full(plane.size, i, np.int16))
        pos.append(np.arange(plane.size, dtype=np.int32))
    keys, cap_idx, pos = np.concatenate(keys), np.concatenate(cap_idx), np.concatenate(pos)
    order = np.argsort(keys, kind="stable")
    return caps, keys[order], cap_idx[order], pos[order]


def match_leaf(hdr, caps, keys, cap_idx, pos):
    _, leaf = leaf_cube(hdr)
    ly, lx = np.nonzero(leaf[:, KEY_BAND, :] != 0)  # leaf pixels; background is 0
    if ly.size == 0:
        return None
    k = ly.size // 2
    ky, kx = ly[k], lx[k]
    key = np.float32(leaf[ky, KEY_BAND, kx]).view(np.uint32)
    lo, hi = np.searchsorted(keys, key, "left"), np.searchsorted(keys, key, "right")
    sample = np.linspace(0, ly.size - 1, min(64, ly.size)).astype(int)
    for c, p in zip(cap_idx[lo:hi], pos[lo:hi]):
        cy, cx = divmod(int(p), SIDE)
        src = capture_cube(caps[c])
        for name, t in ORIENTATIONS.items():
            sy, sx = t(ly - ky, lx - kx)
            sy, sx = sy + cy, sx + cx
            if sy.min() < 0 or sx.min() < 0 or sy.max() >= SIDE or sx.max() >= SIDE:
                continue
            if not np.array_equal(src[sy[sample], :, sx[sample]], leaf[ly[sample], :, lx[sample]]):
                continue
            mismatches = int((src[sy, :, sx] != leaf[ly, :, lx]).sum())  # every leaf pixel, every band
            return {"capture": caps[c], "orientation": name, "key_xy": (cx, cy),
                    "leaf_pixels": int(ly.size), "mismatched_values": mismatches}
    return None


def new_header(leaf_hdr, capture_hdr):
    leaf = envi.read_envi_header(str(leaf_hdr))
    meta = cutout_envi_metadata(envi.read_envi_header(str(capture_hdr)), None)
    meta.update({k: leaf[k] for k in LAYOUT})
    meta["file type"] = "ENVI Standard"
    return meta


def stage(data, stage_dir):
    rows, failed = [], []
    for variety, folder in VARIETIES.items():
        hdrs = sorted((data / variety).glob("R*/*/*.hdr"), key=lambda p: (p.parts[-3], int(p.parent.name)))
        if not hdrs:
            continue
        caps, keys, cap_idx, pos = build_index(CAPTURES / folder)
        print(f"{variety}: {len(hdrs)} leaves, {len(caps)} captures indexed", flush=True)
        for hdr in hdrs:
            rel = hdr.relative_to(data)
            m = match_leaf(hdr, caps, keys, cap_idx, pos)
            if m is None or m["mismatched_values"]:
                failed.append((str(rel), m))
                continue
            cap_hdr = m["capture"].with_suffix(".hdr")
            out = stage_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            envi.write_envi_header(str(out), new_header(hdr, cap_hdr))
            rows.append({
                "leaf": str(rel.with_suffix(".img")),
                "capture": str(m["capture"].relative_to(CAPTURES)),
                "capture_id": m["capture"].parent.parent.name,
                "orientation": m["orientation"],
                "leaf_pixels": m["leaf_pixels"],
                "mismatched_values": m["mismatched_values"],
            })
    with open(ROOT.parent / "helpers" / "leaf_sources.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"matched {len(rows)} leaves exactly; unmatched {len(failed)}: {failed}")
    return failed


def check_staged(data, stage_dir):
    """Every staged header opens, describes its .img exactly and parses like the HDF5 builders expect."""
    staged = sorted(stage_dir.rglob("*.hdr"))
    for s in staged:
        rel = s.relative_to(stage_dir)
        img = data / rel.with_suffix(".img")
        h = envi.read_envi_header(str(s))
        assert int(h["lines"]) * int(h["bands"]) * int(h["samples"]) * 4 == img.stat().st_size, rel
        assert len(h["wavelength"]) == int(h["bands"]) and h["default bands"], rel
        text = s.read_text()
        get = lambda k: re.search(rf"^{k}\s*=\s*(\S+)", text, re.M).group(1)  # build_h5_*.parse_hdr
        assert get("data type") == "4" and get("interleave").lower() == "bil" and get("byte order") == "0", rel
        px = envi.open(str(s), str(img)).read_pixel(int(h["lines"]) // 2, int(h["samples"]) // 2)
        _, raw = leaf_cube(data / rel)
        assert np.array_equal(px, raw[int(h["lines"]) // 2, :, int(h["samples"]) // 2]), rel
    print(f"checked {len(staged)} staged headers")
    return [s.relative_to(stage_dir) for s in staged]


def snapshot(root):
    """(size, mtime) of every file except headers, to prove nothing else is touched."""
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".hdr"):
                st = os.stat(os.path.join(dp, f))
                out[os.path.join(dp, f)] = (st.st_size, st.st_mtime_ns)
    return out


def apply(stage_dir, rels, roots):
    for root in roots:
        before = snapshot(root)
        for rel in rels:
            assert (root / rel).exists() and (root / rel).with_suffix(".img").exists(), root / rel
            shutil.copyfile(stage_dir / rel, root / rel)
        assert snapshot(root) == before, f"non-header files changed under {root}"
        assert all((root / r).read_bytes() == (stage_dir / r).read_bytes() for r in rels)
        print(f"updated {len(rels)} headers in {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "Data"), help="Data folder used for matching")
    ap.add_argument("--stage", required=True, help="folder for the new headers (written first, checked, then copied)")
    ap.add_argument("--apply", nargs="*", metavar="DATA", help="Data folders to copy the staged headers into")
    args = ap.parse_args()
    data, stage_dir = Path(args.data), Path(args.stage)
    if args.apply is None:
        failed = stage(data, stage_dir)
        check_staged(data, stage_dir)
        sys.exit(1 if failed else 0)
    rels = check_staged(data, stage_dir)
    apply(stage_dir, rels, [Path(r).expanduser() for r in args.apply])


if __name__ == "__main__":
    main()
