"""Check the single classification tree against scikit-learn and measure what each design choice costs.
Four fully grown, class-weighted trees are compared with stratified five-fold cross-validation (CV) on the development set. They are A: scikit-learn on raw values, B: the tree of this project on lossless codes with nominal codes as ordered numbers, C: the same tree with one-against-the-rest nominal splits, and D: as C with 255 bins per feature.
The paired differences A - B, C - B and C - D in macro-F1 isolate the implementation, the nominal splits and the cost of binning. Macro-F1 is the mean over the classes of the F1 score, which combines precision and recall.
A second part makes one binned feature at a time lossless to find the features that cause the cost of binning.
The verification part of the implementation section of the report uses these results.
Run: python -m scripts.check_tree. The script writes results/check_tree.json.
"""

import json
import time

import numpy as np

from ids import config
from ids.data import load_dataset
from ids.encoding import TreeEncoder
from ids.metrics import macro_f1
from ids.splits import development_generalisation_split, stratified_kfold
from ids.tree import ClassificationTree, class_weights

try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    raise SystemExit("scikit-learn is required for this check: pip install scikit-learn")

SEEDS = 3

LABELS = {
    "A": "scikit-learn, raw values, ordered nominal",
    "B": "own, lossless, ordered nominal",
    "C": "own, lossless, one-vs-rest nominal",
    "D": "own, 255 bins, one-vs-rest nominal",
}


def fold_inputs(X, tr, va):
    """
    Prepare the three kinds of input for one fold, all fitted on the training part.

    Args:
        X: Table of development features.
        tr: Indices of the training part.
        va: Indices of the validation part.

    Returns:
        A pair: the lossless encoder and a dictionary. The keys "exact" and "binned" hold (training codes, validation codes, number of codes, code values). The key "raw" holds the training and validation matrices of raw numeric values with nominal features as codes.
    """
    exact = TreeEncoder().fit(X.iloc[tr])
    binned = TreeEncoder(255).fit(X.iloc[tr])
    out = {
        "exact": (exact.transform(X.iloc[tr]), exact.transform(X.iloc[va]), exact.n_codes,
                  exact.code_values()),
        "binned": (binned.transform(X.iloc[tr]), binned.transform(X.iloc[va]), binned.n_codes,
                   binned.code_values()),
    }
    raw_tr, raw_va = out["exact"][0].astype(float), out["exact"][1].astype(float)
    for j, name in enumerate(exact.feature_names):
        if not exact.is_nominal[j]:
            raw_tr[:, j], raw_va[:, j] = X[name].iloc[tr], X[name].iloc[va]
    out["raw"] = (raw_tr, raw_va)
    return exact, out


def mixed_code_values(binned, exact, j):
    """
    Build code values for the binned encoding, but with the values of feature j taken from the lossless encoding.

    Args:
        binned: Pair (values, offsets) of the binned encoding.
        exact: Pair (values, offsets) of the lossless encoding.
        j: Index of the feature that stays lossless.

    Returns:
        A pair (values, offsets) in the same layout as TreeEncoder.code_values.
    """
    (bv, bo), (ev, eo) = binned, exact
    parts = [ev[eo[i]:eo[i + 1]] if i == j else bv[bo[i]:bo[i + 1]] for i in range(len(bo) - 1)]
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    return np.concatenate(parts), offsets


def main():
    """Compare the four configurations on five folds, print the tables and write the results to results/check_tree.json."""
    data = load_dataset()
    dev, _ = development_generalisation_split(data.y)
    X, y = data.X.iloc[dev], data.y[dev]
    folds = stratified_kfold(y, 5, config.rng("folds", 0))

    scores = {k: [] for k in LABELS}
    seconds = {k: [] for k in LABELS}
    leaves = {k: [] for k in LABELS}
    binned_features = None
    single_feature_gain = {}

    for f, (tr, va) in enumerate(folds):
        encoder, inp = fold_inputs(X, tr, va)
        y_tr, y_va = y[tr], y[va]
        w = class_weights(y_tr, config.N_CLASSES, beta=1.0)
        ordered = np.zeros(len(encoder.feature_names), bool)
        one_vs_rest = encoder.is_nominal
        (ex_tr, ex_va, ex_k, ex_v), (bn_tr, bn_va, bn_k, bn_v) = inp["exact"], inp["binned"]

        def own_tree(codes, n_codes, nominal, values, seed):
            """
            Fit the tree of this project on one fold.

            Args:
                codes: Matrix of training codes.
                n_codes: Number of codes of every feature.
                nominal: True for each feature that is split one against the rest.
                values: Code values of the encoding.
                seed: Seed of the tree.

            Returns:
                The fitted ClassificationTree.
            """
            return ClassificationTree(seed=seed).fit(codes, y_tr, n_codes, nominal, config.N_CLASSES,
                                                     class_weight=w, code_values=values)

        runs = {
            "A": (lambda s: DecisionTreeClassifier(class_weight=dict(enumerate(w)), random_state=s)
                  .fit(inp["raw"][0], y_tr), inp["raw"][1]),
            "B": (lambda s: own_tree(ex_tr, ex_k, ordered, ex_v, s), ex_va),
            "C": (lambda s: own_tree(ex_tr, ex_k, one_vs_rest, ex_v, s), ex_va),
            "D": (lambda s: own_tree(bn_tr, bn_k, one_vs_rest, bn_v, s), bn_va),
        }
        d_first_seed = None
        for key, (run, X_va) in runs.items():
            fold_scores = []
            for seed in range(SEEDS):
                t = time.perf_counter()
                model = run(seed)
                seconds[key].append(time.perf_counter() - t)
                fold_scores.append(macro_f1(y_va, model.predict(X_va)))
                leaves[key].append(int(model.get_n_leaves() if key == "A" else model.n_leaves))
            scores[key].append(float(np.mean(fold_scores)))
            if key == "D":
                d_first_seed = fold_scores[0]

        # Make one binned feature at a time lossless, to find which features carry the cost of binning.
        if binned_features is None:
            binned_features = [n for j, n in enumerate(encoder.feature_names)
                               if not encoder.is_nominal[j] and bn_k[j] < ex_k[j]]
        for name in binned_features:
            j = encoder.feature_names.index(name)
            mix_tr, mix_va, mix_k = bn_tr.astype(ex_tr.dtype), bn_va.astype(ex_tr.dtype), bn_k.copy()
            mix_tr[:, j], mix_va[:, j], mix_k[j] = ex_tr[:, j], ex_va[:, j], ex_k[j]
            mix_v = mixed_code_values(bn_v, ex_v, j)
            mixed = own_tree(mix_tr, mix_k, one_vs_rest, mix_v, 0)
            gain = macro_f1(y_va, mixed.predict(mix_va)) - d_first_seed
            single_feature_gain.setdefault(name, []).append(gain)
        print(f"fold {f + 1}/5 done")

    print(f"\n{'configuration':<42}{'macro-F1 (mean ± sd)':>22}{'leaves':>9}{'fit s':>8}")
    for key, label in LABELS.items():
        s = np.array(scores[key])
        print(f"{key}  {label:<39}{s.mean():>13.4f} ± {s.std(ddof=1):.4f}"
              f"{np.mean(leaves[key]):>9.0f}{np.mean(seconds[key]):>8.2f}")

    diffs = {
        "implementation (A - B)": np.subtract(scores["A"], scores["B"]),
        "one-vs-rest nominal (C - B)": np.subtract(scores["C"], scores["B"]),
        "255-bin cost (C - D)": np.subtract(scores["C"], scores["D"]),
    }
    print(f"\n{'paired difference in macro-F1':<30}{'mean ± sd':>18}   per fold")
    for label, d in diffs.items():
        print(f"{label:<30}{d.mean():>+10.4f} ± {d.std(ddof=1):.4f}   " + " ".join(f"{x:+.4f}" for x in d))

    ranked = sorted(single_feature_gain.items(), key=lambda kv: -np.mean(kv[1]))
    print(f"\nmacro-F1 gained by making one binned feature lossless (of {len(ranked)} binned features)")
    for name, g in ranked[:5]:
        print(f"  {name:<20}{np.mean(g):+.4f} ± {np.std(g, ddof=1):.4f}   positive in "
              f"{np.sum(np.array(g) > 0)}/5 folds")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "check_tree.json").write_text(json.dumps({
        "configurations": LABELS, "macro_f1_per_fold": scores, "fit_seconds": seconds,
        "leaves": leaves, "paired_differences": {k: v.tolist() for k, v in diffs.items()},
        "single_feature_lossless_gain": single_feature_gain,
    }, indent=2))
    print("\nwrote results/check_tree.json")


if __name__ == "__main__":
    main()
