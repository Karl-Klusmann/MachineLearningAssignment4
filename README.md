# Random forests for network intrusion classification (ML741 Assignment 4)

Implementations from first principles of a classification tree, a random forest and a k-nearest-neighbour (k-NN) classifier, compared on attack-category classification of the UNSW-NB15 network-traffic subset (the Assignment 2 data). scikit-learn is used only as a reference to verify the implementations.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place `networkTraffic.csv` in `data/` (not tracked by Git).

## Running

All commands run from the repository root with `python -m`.

```bash
# Stage 1: data layer
python -m tests.test_stage1      # checks for loading, splitting, encoding
python -m scripts.prepare_data   # data summary + fixed split -> results/

# Stage 2: classification tree and metrics
python -m tests.test_stage2      # checks for the tree and the metrics
python -m scripts.check_tree     # tree vs scikit-learn, nominal handling, cost of binning
python -m scripts.check_tree_size  # why scikit-learn's tree has more leaves (tie-breaking)

# Stage 3: random forest
python -m tests.test_stage3      # checks for sampling, voting, OOB, strength/correlation
python -m scripts.check_forest   # speed, forest vs scikit-learn, OOB vs held-out

# Stage 4: tuning on the development set (each script reads the winners of the previous ones)
python -m scripts.tune_tree        # pruning alpha of the single tree, 5-fold CV
python -m scripts.e1_forest_size   # E1: OOB macro-F1 vs number of trees, hard vs soft voting
python -m scripts.e2_features      # E2: features per node m, strength and correlation
python -m scripts.e3_tree_size     # E3: leaf size x bag fraction, pruned trees in the forest
python -m scripts.e4_imbalance     # E4: class weights vs balanced/capped sampling

# Stage 5: analysis and final evaluation
python -m tests.test_stage4        # checks for bias-variance, k-NN and the statistics
python -m scripts.e5_bias_variance # E5: bias and variance of trees, bagging and forests
python -m scripts.tune_knn         # k-NN: k, distance weighting, class-weighted voting (5-fold CV)
python -m scripts.e6_final         # E6: the one evaluation on the generalisation set, with tests
```

Add `--quick` to any Stage 4 script for a one-seed, 20-tree run that only checks it works
(results go to `results/quick/`).

## Layout

| Path | Purpose |
|---|---|
| `ids/config.py` | Paths, feature schema, class names, master seed and seeded random streams |
| `ids/data.py` | Load and validate the CSV, remove exact duplicates, flag conflicting-label groups |
| `ids/splits.py` | Stratified hold-out split and stratified k-fold cross-validation |
| `ids/encoding.py` | Training-fitted, lossless encoding of features into integer codes |
| `ids/tree.py` | Classification tree: Gini splits, class weights, multiplicities, cost-complexity pruning |
| `ids/forest.py` | Random forest: bootstrap/balanced/capped samples, m features per node, OOB, voting, strength and correlation |
| `ids/experiments.py` | Shared tuning machinery: development data, seeds, OOB summaries, selection rule |
| `ids/knn.py` | k-NN: A2 preprocessing, parallel Manhattan neighbour search, uniform/distance/class-weighted voting |
| `ids/stats.py` | McNemar's test, paired stratified bootstrap of macro-F1 differences, Holm correction |
| `ids/bias_variance.py` | Bias-variance decomposition of the zero-one loss (Domingos, 2000) |
| `ids/metrics.py` | Confusion matrix, per-class and macro measures, binary normal-vs-attack view |
| `scripts/` | Entry points that run experiments and write to `results/` |
| `tests/` | Checks for each stage |
