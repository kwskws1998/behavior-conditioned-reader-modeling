# Behavior-Conditioned Reader Modeling

This repository trains Part 1 of the project:

```text
reader calibration behavior profile z_r + passage text
-> behavior-conditioned XLM-RoBERTa
-> word-level log1p(TRT) prediction
```

The goal is to test whether a short behavior-only reader profile can personalize gaze prediction without using continuous gaze features as input.

## Data

The MECO data is not committed to this repository. Download it from the shared Google Drive folder:

https://drive.google.com/drive/folders/1zMmdLosiGM8ZY2LNu22yKS8ZyVZM6Kj5?usp=sharing

Expected local path:

```text
data/version 1.3/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda
```

On a cloud GPU machine:

```bash
python scripts/download_data.py
```

Or run `gdown` directly:

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1zMmdLosiGM8ZY2LNu22yKS8ZyVZM6Kj5?usp=sharing" -O data
```

If the folder layout differs after download, pass the exact RDA path with `--rda-path`.

## Setup

Install PyTorch for the cloud GPU CUDA version first if the provider does not already include it. Then:

```bash
pip install -r requirements.txt
```

## Single Run

Concat behavior-conditioning baseline:

```bash
python scripts/train_part1_behavior_conditioned.py \
  --rda-path "data/version 1.3/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda" \
  --output-dir artifacts/part1_concat_en_seed13 \
  --conditioning-type concat \
  --seed 13 \
  --epochs 5 \
  --batch-size 8
```

Text-only baseline:

```bash
python scripts/train_part1_behavior_conditioned.py \
  --rda-path "data/version 1.3/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda" \
  --output-dir artifacts/part1_text_only_en_seed13 \
  --conditioning-type none \
  --seed 13 \
  --epochs 5 \
  --batch-size 8
```

Profile-gated MoE:

```bash
python scripts/train_part1_behavior_conditioned.py \
  --rda-path "data/version 1.3/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda" \
  --output-dir artifacts/part1_moe4_en_seed13 \
  --conditioning-type moe \
  --num-experts 4 \
  --seed 13 \
  --epochs 5 \
  --batch-size 8
```

## Multi-Seed Runs

```bash
python scripts/run_part1_multiseed.py \
  --rda-path "data/version 1.3/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda" \
  --conditioning-types none concat moe \
  --seeds 13 21 42 \
  --epochs 5 \
  --batch-size 8
```

## High-Variance Analysis

After a run finishes, dump word-level predictions:

```bash
python scripts/dump_part1_predictions.py \
  --run-dir artifacts/multiseed/moe_seed13 \
  --rda-path "data/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda"
```

Then analyze whether actual reader profiles help more on words with high across-reader TRT variance:

```bash
python scripts/analyze_high_variance.py \
  --run-dir artifacts/multiseed/moe_seed13 \
  --rda-path "data/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda" \
  --split test
```

Outputs:

```text
artifacts/multiseed/moe_seed13/predictions/
artifacts/multiseed/moe_seed13/high_variance/
```

The main quantities are:

```text
gain_vs_mean = abs_error_mean - abs_error_actual
gain_vs_shuffled = abs_error_shuffled - abs_error_actual
```

Positive gain means the actual reader profile predicted TRT better than the baseline profile.

For all completed runs under `artifacts/multiseed`:

```bash
python scripts/run_high_variance_multirun.py \
  --run-root artifacts/multiseed \
  --rda-path "data/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda" \
  --splits test \
  --stratify-by word_variance
```

This writes:

```text
artifacts/multiseed/word_variance_summary.csv
```

For the reader-specific deviation analysis:

```bash
python scripts/run_high_variance_multirun.py \
  --run-root artifacts/multiseed \
  --rda-path "data/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda" \
  --splits test \
  --stratify-by reader_deviation
```

This writes:

```text
artifacts/multiseed/reader_deviation_summary.csv
```

## MoE Gate Analysis

After the MoE runs finish, analyze whether the profile-conditioned gate behaves like a latent reader-type variable:

```bash
python scripts/analyze_moe_gates.py \
  --run-root artifacts/multiseed \
  --rda-path "data/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda"
```

If you only want one run:

```bash
python scripts/analyze_moe_gates.py \
  --run-dir artifacts/multiseed/moe_seed13 \
  --rda-path "data/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda"
```

Main outputs:

```text
artifacts/multiseed/gate_analysis/gate_analysis_report.json
artifacts/multiseed/gate_analysis/all_moe_gates.csv
artifacts/multiseed/gate_analysis/all_gate_behavior_correlations.csv
artifacts/multiseed/gate_analysis/all_gate_stability.csv
artifacts/multiseed/gate_analysis/all_gate_pca.csv
artifacts/multiseed/gate_analysis/all_dominant_expert_behavior_profiles.csv
```

If word-level predictions already exist under each run's `predictions/` directory, the script also writes:

```text
artifacts/multiseed/gate_analysis/all_gate_gain_correlations.csv
artifacts/multiseed/gate_analysis/all_dominant_expert_gains.csv
```

Interpretation:

```text
gate_stability:
  same reader trial1-vs-trial2 similarity should be higher than different-reader similarity.

gate_behavior_correlations:
  gate weights should correlate with behavior-only profile features such as skip, reread, or regression rates.

dominant_expert_behavior_profiles:
  each dominant expert should show a different behavior profile.

gate_gain_correlations:
  gate entropy or expert assignment should explain which readers benefit more from MoE personalization.
```

## Part 2 Main: Personalized Gaze-Augmented Comprehension Risk

Part 2 tests whether predicted personalized gaze helps predict whether a reader answers a comprehension question correctly.

The closed-form MECO task is question-level:

```text
reader r + passage/trial p + comprehension question q
-> passage text + question text + predicted gaze summary
-> P(correct_{r,p,q})
```

First inspect the MECO schema to find the comprehension correctness column. In MECO 1.3, the relevant
label file is typically `joint_l1_acc_full_breakdown.rda` with `ACCURACY` and `QUESTIONNUM`, while
the question text comes from `comp-questions.xlsx`.

```bash
python scripts/inspect_meco_schema.py \
  --rda-path "data/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda"
```

Then dump Part 1 predictions including the train trials:

```bash
python scripts/dump_part1_predictions.py \
  --run-dir artifacts/multiseed/moe_seed13 \
  --rda-path "data/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda" \
  --include-train \
  --batch-size 8
```

Build the Part 2 dataset:

```bash
python scripts/build_part2_comprehension_dataset.py \
  --part1-run-dir artifacts/multiseed/moe_seed13 \
  --rda-path "data/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda" \
  --comprehension-path "data/primary data/comprehension data/joint_l1_acc_full_breakdown.rda" \
  --correctness-column ACCURACY \
  --question-column QUESTIONNUM \
  --question-materials-path "data/auxiliary files/reading task materials/comp-questions.xlsx" \
  --text-materials-path "data/auxiliary files/reading task materials/supp texts.xlsx"
```

Train the Part 2 classifiers:

Quick sanity check before the language-model run:

```bash
python scripts/train_part2_comprehension_risk.py \
  --dataset-path artifacts/multiseed/moe_seed13/part2_comprehension/part2_dataset.csv \
  --output-dir artifacts/multiseed/moe_seed13/part2_comprehension/linear_backup
```

The key linear baselines are `item_question_only`, `item_question_plus_predicted_gaze`,
`item_question_plus_mean_gaze`, and `item_question_plus_shuffled_gaze`.

```bash
python scripts/train_part2_original_lm.py \
  --dataset-path artifacts/multiseed/moe_seed13/part2_comprehension/part2_dataset.csv
```

For all MoE seeds:

```bash
python scripts/run_part2_multirun.py \
  --run-root artifacts/multiseed \
  --rda-path "data/primary data/eye tracking data/joint_l1_data_trimmed_version1.3.rda" \
  --comprehension-path "data/primary data/comprehension data/joint_l1_acc_full_breakdown.rda" \
  --correctness-column ACCURACY \
  --question-column QUESTIONNUM \
  --question-materials-path "data/auxiliary files/reading task materials/comp-questions.xlsx" \
  --text-materials-path "data/auxiliary files/reading task materials/supp texts.xlsx" \
  --trainer original_lm \
  --epochs 5 \
  --batch-size 8
```

Part 2 model families:

```text
text_only:
  passage text + comprehension question text.

text_plus_profile:
  passage/question text + behavior-only profile z_r.

text_plus_mean_gaze:
  passage/question text + predicted gaze from the mean reader profile.

text_plus_personalized_gaze:
  passage/question text + predicted gaze from the actual reader profile.

text_plus_shuffled_gaze:
  passage/question text + predicted gaze from a mismatched reader profile.

text_plus_profile_personalized_gaze:
  passage/question text + z_r + predicted personalized gaze.
```

Main expected result:

```text
text_plus_personalized_gaze > text_only
text_plus_personalized_gaze > text_plus_profile
text_plus_personalized_gaze > text_plus_mean_gaze
text_plus_personalized_gaze > text_plus_shuffled_gaze
```

Primary metrics:

```text
balanced_accuracy
AUROC
average_precision
Brier score
```

Part 2 backup, not the main plan:

```text
train_part2_comprehension_risk.py
```

This is a lightweight linear/scikit-learn backup for MF/MB-inspired computational modeling.
Use `--trainer linear_backup` in `run_part2_multirun.py` only when explicitly testing that backup story.

## What Enters the Model

Input:

```text
text tokens
+ behavior-only reader profile z_r
```

The current `z_r` features are calibration-trial rates of:

```text
skip
firstrun.skip
reread
refix
reg.in
reg.out
firstrun.refix
firstrun.reg.in
firstrun.reg.out
```

Not input:

```text
dur / TRT
nfix
firstrun.dur
firstrun.nfix
firstfix.dur
gopast
```

Target:

```text
log1p(TRT)
```

Loss:

```text
MSE(predicted log1p(TRT), true log1p(TRT))
```
