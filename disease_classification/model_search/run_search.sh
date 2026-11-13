#!/bin/sh
# Every step of the disease model search, in the order it ran. Each family skips candidates already saved in
# cache/disease_search/preds/ and every feature cache is built only once, so the script can be rerun after a stop.
# See README.md for the environments.
set -e
cd "$(dirname "$0")"
PY=../../.venv/bin/python
TABPFN_PY=${TABPFN_PY:-$PY}   # an interpreter with tabpfn 2.0.9 and scikit-learn < 1.7

# 1. Per-leaf feature caches (cache/disease_search/*.npz), built only when missing: the saved predictions were made
#    from these files, so rebuilding one (e.g. prospect.npz with prosail) would no longer match them.
C=../../cache/disease_search
[ -f $C/features.npz ] || $PY build_features.py
[ -f $C/texture.npz ] || $PY build_texture.py
[ -f $C/pixels.npz ] || $PY build_pixels.py
[ -f $C/dino_small_rgb.npz ] || $PY build_dino.py vit_small_patch14_dinov2.lvd142m
[ -f $C/morph.npz ] || $PY build_shape.py
[ -f $C/prospect.npz ] || $PY build_prospect.py

# 2. Untuned families (2026-11-12): within one variety and pooled
$PY check_baseline.py
$PY check_pooled.py
$PY family_spectral.py
$PY family_dino.py small
$PY family_rownorm_texture.py
$PY family_pixel_sil.py
$PY family_photo_centring.py
$PY permutation_null.py family_spectral svm:snv_vis within 50
$PY permutation_null.py family_rownorm_texture lda:rownorm_snv_vis within 50

# 3. CNNs with a clean validation row (GPU): backbone pooling bands mode
for cfg in "small_cnn mean all within" "small_cnn mean all pooled" "small_resnet mean all within" \
           "small_resnet mean all pooled" "small_cnn mean vis within" "small_cnn mean vis pooled" \
           "small_cnn mean rgb within" "small_cnn mean rgb pooled" "resnet18_imagenet mean rgb pooled" \
           "conv3d mean vis pooled"; do
  $PY family_cnn.py $cfg
done
# The repo's own small-ResNet runs (early-stopped on the validation row), for reference only: prefix ES_
for b in "" "_rgb"; do
  for v in Baco Chardonnay Vidal; do
    lv=$(echo $v | tr 'A-Z' 'a-z')
    $PY import_repo_cnn.py ES_cnn_small_resnet$b disease_classification/within_variety_disease_cnn/results/${lv}${b}_small_resnet $v
  done
  $PY import_repo_cnn.py ES_cnn_small_resnet$b disease_classification/cnn/results/baco_chardonnay_vidal${b}_small_resnet baco_chardonnay_vidal
done

# 4. Pooled target (2026-11-13): multi-task, FiLM CNN, leaf shape, TabPFN
$PY family_multitask.py
for cfg in "film_small_cnn mean all pooled" "film_small_cnn attention all pooled" "film_small_cnn mean vis pooled"; do
  $PY family_cnn.py $cfg
done
$PY family_shape.py
$TABPFN_PY family_tabpfn.py

# 5. Wide hyperparameter tuning (prefix tuned:)
for algo in lr lda pls svm knn rf et hgb; do
  $PY family_tuned.py $algo
done
$TABPFN_PY family_tuned_tabpfn.py
for hp in "3e-4 1e-2 0.5" "3e-3 1e-2 0.5" "1e-3 1e-1 0.5" "1e-3 1e-4 0.2" "3e-4 1e-1 0.2" "3e-3 1e-4 0.2"; do
  $PY family_cnn.py small_cnn mean vis within 3 $hp
done

# 6. Literature methods (prefix lit:) and the split comparison
for part in prospect cars spa joint4; do
  $PY family_literature.py $part
done
$PY sawyer_protocol.py

# 7. Reports
$PY report.py
