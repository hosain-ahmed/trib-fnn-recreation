# Dataset Setup

The dataset files are intentionally not stored in normal Git because the
processed tensors are large and the raw dataset may have redistribution or
licensing restrictions. The preprocessing and tokenization code is stored in
the repository so every teammate can recreate the same outputs.

## Required files

Obtain the raw dataset from the team's shared storage or the approved original
source. The expected filename is:

```text
bangla_online_comments_dataset.xlsx
```

Place it here:

```text
data/raw/bangla_online_comments_dataset.xlsx
```

Do not rename the file unless you also update `data/data_pipeline.py`.

## Generate the processed data

From the repository root, create and activate a virtual environment if needed,
then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the cleaning, balancing, and stratified splitting pipeline:

```powershell
python data/data_pipeline.py
```

This creates:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

The pipeline performs the sprint-plan preprocessing steps: removes unwanted
English characters, punctuation and pictographic symbols, removes isolated
Bangla characters, collapses whitespace, removes missing/empty rows, creates a
stratified 70/15/15 split, and randomly oversamples only the training split.

## Generate token tensors

Run:

```powershell
python data/tokenizer.py
```

The tokenizer downloads `csebuetnlp/banglabert` the first time it is used and
creates:

```text
data/processed/train_tensors.pt
data/processed/val_tensors.pt
data/processed/test_tensors.pt
```

These files contain `input_ids`, `attention_mask`, and `labels`. They are
ignored by Git and should be regenerated rather than committed to the normal
repository history.

## Verify the setup

Check that the expected files exist:

```powershell
Get-ChildItem data/raw
Get-ChildItem data/processed
```

Then run the Pair 1 shape tests:

```powershell
python model/shape_test.py
```

Run the quick Day 3 baseline comparison:

```powershell
python -m train.train_baselines --epochs 1 --max-batches 50 --batch-size 64 --output eval/baseline_results.json
```

For a complete training pass, omit `--max-batches`. The processed tensor
files must exist before running the baseline training script.

## Team-sharing rules

1. Store the raw workbook and generated tensors in the team's approved shared
   storage location.
2. Share the storage link privately with project members; do not commit a
   private link or credentials to this repository.
3. Everyone should use the same raw dataset version, tokenizer model, and
   random seed (`42`) when reproducing the sprint results.
4. If the dataset changes, record the change and its date in `DECISIONS.md`.
5. Do not upload the dataset if its license does not permit redistribution.

Git LFS is an alternative for teams that explicitly want the tensors versioned
and have sufficient LFS storage and bandwidth. It must be configured before
adding the files; ordinary Git is not suitable for the large `.pt` files.
