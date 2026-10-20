# Task: Implement controlled PMMFNet experiments for grapevine disease classification

I want to improve my PMMFNet model for classifying hyperspectral grapevine leaf images into four classes: healthy, red blotch only, leafroll only, and both diseases.

My existing implementation is in `experiments/11_pmmfnet.py`, with the network in `pmmfnet.py` and shared data/fold utilities in `common.py`. Please inspect these files and the existing results before making changes.

I have already completed the original five-fold experiment. Do not overwrite its source files, metrics, predictions, or checkpoints.

## General requirements

Keep the existing five folds, leaf assignments, and group-based splitting exactly the same. Do not generate new folds. Keep the same dataset, labels, model architecture, Adam optimizer, learning rate (`1e-5`), original patch size (`5x5`), batch size (`256` patch inputs), and other settings unless a change is explicitly specified below.

Preserve the current output metrics and evaluate all models at the leaf level. Each leaf must contribute exactly one final prediction to the evaluation.

Use **four-class leaf-level balanced accuracy as both the primary reported evaluation metric and the primary validation checkpoint-selection metric** for all new controlled experiments. Track validation loss as a secondary metric. Save the checkpoint with the highest validation balanced accuracy; use lower validation loss to break ties. Stop after 5 epochs without improvement in the primary checkpoint metric, and restore the best checkpoint.

This changes the checkpoint-selection rule from the original experiment, which selected checkpoints using validation loss. Handle that difference explicitly in the experiment design below.

Continue reporting macro-F1, per-class precision and recall, the confusion matrix, one-vs-rest ROC AUC, within-variety AUC, and the existing per-disease and per-variety metrics. Calculate these metrics on held-out test leaves, not on individual patches.

Keep validation samples fixed across epochs so changes in metrics reflect model changes rather than different sampled patches. Do not use test-fold results to choose checkpoints, hyperparameters, preprocessing, or experiment configurations.

Save every new experiment to a separate output directory. Preserve reproducibility with explicit random seeds and document all settings.

## Stage 0: Existing original baseline — reference only

Use the results already available from my original experiment.

The original configuration uses:
- Random spatial shuffle during training.
- Original spatial ordering during validation and testing.
- Patch-level training loss.
- Scaling fitted using training and validation leaves, excluding test leaves.
- Checkpoint selection based on minimum validation loss.

Do not rerun this experiment unless needed for a specific debugging reason. Preserve its results as the historical reference.

## Stage 1: Isolate the effect of checkpoint selection

### Experiment A1 — Balanced-accuracy checkpoint baseline

Re-run the original configuration with exactly the same preprocessing, train/validation/test splits, patch sampling, spatial shuffling, architecture, and hyperparameters.

Change only the checkpoint-selection criterion:
- Select the checkpoint with the highest four-class leaf-level validation balanced accuracy.
- Continue using patch-level training loss.
- Keep the existing scaling procedure for this experiment, including fitting the global minimum and maximum using training plus validation leaves.
- Keep the existing train-shuffled, validation/test-original ordering.

This experiment isolates the effect of changing checkpoint selection from validation loss to validation balanced accuracy.

## Stage 2: Correct the scaling procedure

### Experiment A2 — Training-only scaler

Start from A1 and change only the scaling procedure.

Currently, the scaler's minimum and maximum are calculated across the non-test leaves, meaning training plus validation leaves. Instead:

1. Calculate the global minimum and maximum using only the training leaves that contribute to gradient updates.
2. Apply those exact same fitted values to transform training, validation, and test data.
3. Preserve the current global min-max scaling definition across bands and pixels.
4. Preserve the existing zero-padding convention and its ordering relative to scaling.
5. Do not fit or refit any scaling parameters using validation or test data.
6. Do not clip transformed validation or test values unless clipping is already part of the existing pipeline.

Keep all other settings identical to A1.

The goal is to measure the effect of correcting scaling while holding checkpoint selection and everything else constant.

## Stage 3: Spatial-shuffling ablations

Use the A2 configuration as the baseline for these experiments: training-only scaling, patch-level supervision, and validation balanced-accuracy checkpoint selection.

Run all three configurations using the same folds, architecture, patch sampler, optimizer, learning rate, training budget, and other settings.

### Experiment S1 — No shuffle

- Training: preserve the original spatial arrangement of every `5x5` patch.
- Validation and test: preserve the original spatial arrangement.

### Experiment S2 — Shuffle during training, validation, and testing

- Training: randomly permute the 24 non-centre pixels of each `5x5` patch, leaving the centre pixel fixed. Continue using fresh permutations during training as in the current implementation.
- Validation and test: apply the same spatial-shuffle transformation, but make it deterministic. Generate and cache a reproducible permutation for each evaluation patch so the same inputs are used across validation epochs and repeated evaluations.

Do not change the order of spectral bands. This experiment shuffles spatial positions only.

### Experiment S3 — Shuffle during training only

- Training: randomly permute the 24 non-centre pixels, keeping the centre pixel fixed.
- Validation and test: preserve the original spatial arrangement.

This is the spatial-shuffle policy used by my existing implementation.

For all three experiments, change only the specified shuffle policy relative to the A2 setup. Report all metrics for each configuration.

## Stage 4: Leaf-level training versus patch-level training

I want to test whether training against the actual leaf-level label improves generalization.

### Experiment L1 — Patch-level training reference

Use A2's corrected scaler and S3's shuffle policy. Calculate the training loss for each individual patch using its leaf's class label, as in my current implementation.

### Experiment L2 — Leaf-level multiple-instance learning

Start with the same corrected scaler, S3 shuffle policy, architecture, folds, and optimizer.

Change the training objective so that the model is optimized against one disease label per leaf.

For each training leaf:
1. Sample 32 interior `5x5` patches from that leaf.
2. Pass each patch through PMMFNet to obtain four-class logits.
3. Apply softmax to obtain patch-level probabilities.
4. Average the patch-level probabilities to create one four-class probability vector for the leaf.
5. Calculate the cross-entropy loss for that leaf using its actual class label.

Backpropagate using the leaf-level loss rather than independent patch-level losses.

To preserve a comparable training budget, use balanced batches of 8 leaves, with 2 leaves from each of the four classes. Sample leaves with replacement when necessary. Each leaf contributes 32 patches per batch, giving 256 patch inputs per batch. Aim for approximately the same total number of patch inputs per epoch as the existing 400,000-patch training budget, and document the actual number of optimizer steps and patch inputs.

Because the leaf-level batches are already class-balanced, do not add inverse-frequency class weights on top of that sampling unless running a separate, explicitly documented experiment.

Use the same fixed set of up to 256 validation patches per leaf as the existing pipeline. Aggregate the probabilities to one prediction per leaf before calculating validation loss, balanced accuracy, and other metrics. For test evaluation, retain the existing approach of averaging patch probabilities across the leaf's interior pixels.

Compare L1 and L2 using the same scaler, shuffle policy, folds, and checkpoint-selection rule. The principal change should be the level at which the training loss is calculated.

Do not introduce a second, duplicate leaf-level training experiment. L1 and L2 are the patch-level and leaf-level comparison.

## Reporting and outputs

For every new experiment:

- Run all five existing folds.
- Record fold assignments, random seeds, scaling parameters, shuffle policy, selected checkpoint epoch, stopping epoch, best validation balanced accuracy, validation loss, and training duration.
- Save out-of-fold leaf probabilities and predictions, along with the existing metrics in CSV, Markdown, and JSON formats wherever supported by the current pipeline.
- Keep outputs separated by experiment so nothing is overwritten.
- Create a final comparison table with one row per experiment and the following columns: four-class balanced accuracy, macro-F1, per-class recall, per-class precision, mean one-vs-rest AUC, mean within-variety AUC, and runtime.

In the report, clearly distinguish the historical original baseline from A1, which uses the new checkpoint-selection criterion. Use A1 to measure the effect of changing checkpoint selection, A2 to measure the effect of correcting scaling, S1–S3 to compare shuffle policies, and L1–L2 to compare training objectives.

Do not use the test results to select a winning shuffle policy or training objective and then present those same results as an unbiased final test estimate. Treat comparisons as experimental results and document any limitations.

Before launching the full runs, verify the implementation with a small smoke test to ensure the splits, scaling, patch shapes, leaf-level loss, metric calculations, and output paths are correct. Then run the full experiments and summarize which configurations performed best, whether the differences were consistent across folds, and what should be investigated next.