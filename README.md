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
