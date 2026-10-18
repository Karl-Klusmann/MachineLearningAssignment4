"""Explain why the fully grown tree of scikit-learn has more leaves than the tree of this project.
The script check_tree.py found equal macro-F1 but about 9% more leaves for scikit-learn. Macro-F1 is the mean over the classes of the F1 score, which combines precision and recall.
Both trees choose an optimal split at every node, so they can only differ in which of several equally good splits they pick.
The script grows both trees on the same data with three kinds of class weights: none, whole numbers and fractions.
If the leaf counts agree for the first two and differ only for fractions, rounding of floating-point numbers causes the gap.
Run: python -m scripts.check_tree_size. The script prints a table and writes no file.
"""

import numpy as np

from ids import config
from ids.data import load_dataset
from ids.encoding import TreeEncoder
from ids.splits import development_generalisation_split, stratified_kfold
from ids.tree import ClassificationTree, class_weights

try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    raise SystemExit("scikit-learn is required for this check: pip install scikit-learn")


def main():
    """Grow both trees on every fold with each kind of class weights and print the mean number of leaves."""
    data = load_dataset()
    dev, _ = development_generalisation_split(data.y)
    X, y = data.X.iloc[dev], data.y[dev]
    folds = stratified_kfold(y, 5, config.rng("folds", 0))
    leaves = {(w, impl): [] for w in ("none", "integer", "fraction") for impl in ("sklearn", "own")}

    for f, (tr, _) in enumerate(folds):
        enc = TreeEncoder().fit(X.iloc[tr])
        codes, y_tr = enc.transform(X.iloc[tr]), y[tr]
        ordered = np.zeros(len(enc.feature_names), bool)
        n_c = np.bincount(y_tr, minlength=config.N_CLASSES)
        weights = {
            "none": np.ones(config.N_CLASSES),
            # Whole-number weights keep sums of weights exact, so tied splits stay exactly tied.
            "integer": np.round(n_c.max() / n_c),
            # Fractional weights carry rounding errors, which can break a tie between splits.
            "fraction": class_weights(y_tr, config.N_CLASSES, beta=1.0),
        }
        for name, w in weights.items():
            sk = DecisionTreeClassifier(class_weight=dict(enumerate(w)), random_state=f)
            leaves[name, "sklearn"].append(int(sk.fit(codes.astype(np.float32), y_tr).get_n_leaves()))
            own = ClassificationTree(seed=f).fit(codes, y_tr, enc.n_codes, ordered,
                                                  config.N_CLASSES, class_weight=w)
            leaves[name, "own"].append(own.n_leaves)
        print(f"fold {f + 1}/5 done")

    print(f"\n{'class weights':<15}{'scikit-learn':>14}{'own':>10}{'difference':>13}   (mean leaves over 5 folds)")
    for name in ("none", "integer", "fraction"):
        a, b = np.mean(leaves[name, "sklearn"]), np.mean(leaves[name, "own"])
        print(f"{name:<15}{a:>14,.0f}{b:>10,.0f}{a - b:>+13,.0f}")


if __name__ == "__main__":
    main()
