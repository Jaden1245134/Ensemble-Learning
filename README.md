# Random Forest for Network Attack Classification

Machine Learning assignment 3 (chose option 1, ensemble learning). A random forest is
implemented from scratch in NumPy (no scikit-learn) and compared with a decision
tree and a k-nearest-neighbour classifier on the UNSW-NB15-style
`networkTraffic.csv` data set (10 classes: Normal and nine attack types).

## What the script does

1. Cleans the data (drops `id`, removes exact duplicates, collapses rare nominal levels)
   and draws a stratified working set of 60,000 rows.
2. Tunes the forest on out-of-bag (OOB) macro F1: number of trees, features per
   split, depth, leaf size and class-weight exponent.
3. Compares six models (tree, k-NN and forest, each with and without an imbalance
   treatment) on 30 stratified 70/30 train/test partitions (seeds 42 to 71).
4. Plots the results and prints the summary tables and paired comparisons.
5. Fits the forest with fully balanced class weights (gamma = 1) as an extra check.

## Setup

Used the recent Python 3 (3.14.3) (the pinned versions in `requirements.txt` need a modern
interpreter, 3.11 or newer).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `networkTraffic.csv` in the same folder as the script.

## Running

```bash
python script.py                   # full run
NB_QUICK=1 python script.py        # fast smoke test on a small subsample
NB_RUNS=5 python script.py         # fewer partitions than the default 30
```

On Windows PowerShell, set the variable first: `$env:NB_QUICK = "1"`.

The full run fits many forests and takes a long time (about an hour or more,
depending on the number of CPU cores, which are all used). Results of each
partition are cached in `runs_cache/`, so a repeated run skips finished
partitions. Delete the folder to start again.

The script was exported from a notebook. Each `plt.show()` opens a window and pauses
the script until it is closed. The notebook (`ensemble_learning.ipynb`) shows the
figures inline and is easier to read.

## Files

| File | Purpose |
|---|---|
| `script.py` | Whole experiment, from data cleaning to results |
| `requirements.txt` | Pinned package versions |
| `networkTraffic.csv` | Data set (not included, supply your own copy) |
| `runs_cache/` | Cached per-partition results, created on the first run |

## Notes

- Random seeds are fixed, so results reproduce exactly on the same data.
- `scikit-learn` is not needed. If it is installed, the script uses it only to
  check the own tree and k-NN against reference implementations.