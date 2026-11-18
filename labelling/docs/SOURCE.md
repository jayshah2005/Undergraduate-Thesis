# Sensor documentation

Hyperspectral data were captured with a Specim IQ (portable hyperspectral camera, discontinued).

- `Specim-IQ-Datasheet.pdf` — retrieved 2026-10-11 from
  https://www.specim.com/wp-content/uploads/2024/11/Specim-IQ-Datasheet-Long-08.pdf
  (backup: https://atecorp.com/getmedia/bc972f54-becd-4514-ad27-0887f60dfcde/specim-iq-datasheet_1.pdf)
- Datasheet specs: 400-1000 nm, 204 bands (2x bin: 102, 3x bin: 68), 7 nm mean spectral resolution, 512 x 512 px.
- Only the datasheet was found; the full user manual was not located.
- Wavelengths: `/wavelengths` in the HDF5 is the camera's **factory calibration** (`wavelengths_source="factory"`),
  397.32–1003.58 nm in uneven ~2.9–3.1 nm steps, read from the leaf `.hdr` files. Every capture in
  `VINO33/leaves_indoor` has the same list and `default bands = {70, 53, 19}` (R/G/B, 1-based).
- History: the leaf `.hdr` files in `Data/` were first written without the capture header (no wavelengths, default bands
  or camera settings), and the HDF5 used a nominal `linspace(400, 1000, 204)` grid instead — off by up to 5.3 nm
  (mean 3.6 nm). On 2026-10-29 `restore_leaf_headers.py` matched every leaf to its source capture by exact pixel values
  (998/998 matched, all rotated 90° clockwise; see `leaf_sources.csv`) and rewrote the headers from the capture's header.
  The original headers are in `CODE/header_backups/Data_hdr_original_2026-10-29.tar.gz`.
- Camera dates: the header `acquisition date` (kept as `camera_acquisition_date`) does not always match the field dates in
  `CAPTURE_DATES`: Cab Franc and Vidal say 26–27 Sep, although the files were already on this computer on 11 Sep
  (their inode change time), so the camera clock was wrong for those; Chardonnay says 25 Aug (vs 26 Aug) and 82 of the
  248 Baco leaves say 13 Aug (vs 12 Aug). `capture_date` stays the field date.
