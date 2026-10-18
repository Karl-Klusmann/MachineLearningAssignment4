"""Tests for the classification tree and the performance measures.
They check ids/tree.py and ids/metrics.py on small hand-made examples and, where scikit-learn is installed, against scikit-learn.
The tests also check by brute force that every split of the tree is the best available one. One measure is macro-F1, the mean over the classes of the F1 score, which combines precision and recall.
The verification part of the implementation section of the report relies on them.
Run: python -m tests.test_stage2. Tests that need scikit-learn return without checking if it is missing.
"""

import numpy as np

from ids import config
from ids.data import load_dataset
from ids.encoding import TreeEncoder
from ids.metrics import confusion_matrix, macro_f1, per_class_scores, summarise
from ids.splits import development_generalisation_split, stratified_holdout
from ids.tree import LEAF, ClassificationTree, class_weights

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


def fit(X, y, nominal=None, n_classes=None, **kwargs):
    """
    Fit a tree on a small matrix of whole numbers.

    Args:
        X: Matrix of codes, one row per instance.
        y: Class label of every instance.
        nominal: True for each nominal feature, or None for all numeric.
        n_classes: Number of classes, or None to use the largest label plus one.
        **kwargs: Tree settings (max_features, min_samples_leaf, max_depth, seed) and the other arguments of ClassificationTree.fit, such as multiplicity.

    Returns:
        The fitted ClassificationTree.
    """
    X = np.asarray(X, np.uint8)
    n_codes = X.max(axis=0).astype(np.int64) + 1
    nominal = np.zeros(X.shape[1], bool) if nominal is None else np.asarray(nominal)
    tree_args = {k: kwargs.pop(k) for k in ("max_features", "min_samples_leaf", "max_depth", "seed")
                 if k in kwargs}
    n_classes = n_classes or int(np.max(y)) + 1
    return ClassificationTree(**tree_args).fit(X, np.asarray(y), n_codes, nominal, n_classes, **kwargs)


def test_metrics_on_hand_computed_example():
    """Checks the confusion matrix, precision, recall, F1, detection rate and false-alarm rate on a six-instance example."""
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    precision, recall, f1 = per_class_scores(cm)
    assert np.allclose(precision, [1.0, 0.5, 0.5])
    assert np.allclose(recall, [2 / 3, 0.5, 1.0])
    assert np.allclose(f1, [0.8, 0.5, 2 / 3])
    s = summarise(y_true, y_pred, 3)
    assert np.isclose(s["detection_rate"], 1.0)
    assert np.isclose(s["false_alarm_rate"], 1 / 3)


def test_metrics_match_sklearn():
    """Checks that accuracy, macro-F1, weighted F1 and balanced accuracy equal the values of scikit-learn."""
    if not HAVE_SKLEARN:
        return
    from sklearn import metrics as skm
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 10, 5000)
    y_pred = np.where(rng.random(5000) < 0.6, y_true, rng.integers(0, 9, 5000))
    s = summarise(y_true, y_pred, 10)
    assert np.isclose(s["accuracy"], skm.accuracy_score(y_true, y_pred))
    assert np.isclose(s["macro_f1"], skm.f1_score(y_true, y_pred, average="macro", zero_division=0))
    assert np.isclose(s["weighted_f1"], skm.f1_score(y_true, y_pred, average="weighted", zero_division=0))
    assert np.isclose(s["balanced_accuracy"], skm.balanced_accuracy_score(y_true, y_pred))


def test_learns_a_threshold():
    """Checks that the tree finds the single threshold that separates two classes."""
    X = [[0], [1], [2], [3], [4], [5]]
    y = [0, 0, 0, 1, 1, 1]
    tree = fit(X, y)
    assert tree.n_leaves == 2 and tree.feature[0] == 0 and tree.code[0] == 2
    assert tree.predict(np.array(X, np.uint8)).tolist() == y


def test_threshold_is_placed_midway_in_the_gap():
    """Checks that a threshold lies midway between the neighbouring training values, also when the values are not evenly spaced."""
    X = np.array([[0], [0], [10], [10]], np.uint8)
    y = np.array([0, 0, 1, 1])
    n_codes, nominal = np.array([11]), np.array([False])
    tree = ClassificationTree().fit(X, y, n_codes, nominal, 2)
    # Codes 1 to 9 are unseen values. The threshold lies midway between codes 0 and 10, so codes 0 to 5 go left.
    assert tree.code[0] == 5
    values = (np.array(list(range(10)) + [100.0]), np.array([0, 11]))
    tree = ClassificationTree().fit(X, y, n_codes, nominal, 2, code_values=values)
    # Midpoint of the values 0 and 100 is 50. The largest code with a value up to 50 is code 9.
    assert tree.code[0] == 9


def test_learns_xor_which_needs_a_zero_gain_first_split():
    """Checks that the tree fits the XOR pattern, where the first split gives no gain in impurity."""
    X = [[0, 0], [0, 1], [1, 0], [1, 1]] * 5
    y = [0, 1, 1, 0] * 5
    tree = fit(X, y)
    assert (tree.predict(np.array(X, np.uint8)) == np.array(y)).all()


def test_nominal_split_is_one_category_against_the_rest():
    """Checks that a nominal test isolates one category, while the same data treated as ordered needs two thresholds."""
    X = [[0], [1], [2], [3], [4]] * 4
    y = [0, 0, 1, 0, 0] * 4
    tree = fit(X, y, nominal=[True])
    assert tree.n_leaves == 2 and tree.code[0] == 2
    numeric_tree = fit(X, y, nominal=[False])
    assert numeric_tree.n_leaves == 3


def test_conflicting_identical_rows_end_in_a_mixed_leaf():
    """Checks that identical rows with different labels end in one leaf that predicts the more frequent label."""
    X = [[1, 1]] * 3 + [[0, 0]]
    y = [0, 1, 1, 0]
    tree = fit(X, y)
    leaf = tree.apply(np.array([[1, 1]], np.uint8))[0]
    assert tree.left[leaf] == LEAF
    assert np.allclose(tree.value[leaf], [1, 2])
    assert tree.predict(np.array([[1, 1]], np.uint8))[0] == 1


def test_min_samples_leaf_and_max_depth_are_respected():
    """Checks that no leaf is smaller than the minimum leaf size and that the tree is not deeper than the depth limit."""
    rng = np.random.default_rng(2)
    X = rng.integers(0, 20, (500, 4))
    y = rng.integers(0, 3, 500)
    tree = fit(X, y, min_samples_leaf=7)
    leaves = np.flatnonzero(tree.left == LEAF)
    assert tree.count[leaves].min() >= 7
    assert fit(X, y, max_depth=3).depth <= 3


def test_multiplicity_equals_duplicating_rows():
    """Checks that a tree grown with multiplicities equals a tree grown on the repeated rows."""
    rng = np.random.default_rng(3)
    X = rng.integers(0, 10, (200, 3))
    y = rng.integers(0, 3, 200)
    # A multiplicity of 0 leaves the instance out, like an out-of-bag instance.
    mult = rng.integers(0, 4, 200)
    a = fit(X, y, multiplicity=mult, min_samples_leaf=3)
    b = fit(np.repeat(X, mult, axis=0), np.repeat(y, mult), min_samples_leaf=3)
    probe = np.array(rng.integers(0, 10, (300, 3)), np.uint8)
    assert np.array_equal(a.predict_proba(probe), b.predict_proba(probe))


def test_class_weights():
    """Checks that beta = 0 gives equal weights, beta = 1 gives equal total weight per class and the total weight equals the sample size."""
    y = np.array([0] * 90 + [1] * 10)
    assert np.allclose(class_weights(y, 2, beta=0.0), [1, 1])
    w = class_weights(y, 2, beta=1.0)
    assert np.isclose(90 * w[0], 10 * w[1])
    assert np.isclose(90 * w[0] + 10 * w[1], 100)


def test_feature_sampling_is_seeded():
    """Checks that the random choice of features repeats for the same seed and changes for another seed."""
    rng = np.random.default_rng(4)
    X = rng.integers(0, 10, (400, 8))
    y = (X[:, 0] + X[:, 3] > 9).astype(int)
    same = [fit(X, y, max_features=2, seed=5) for _ in range(2)]
    other = fit(X, y, max_features=2, seed=6)
    assert np.array_equal(same[0].feature, same[1].feature)
    assert not np.array_equal(same[0].feature, other.feature)


def test_pruning_behaviour():
    """Checks that pruning keeps the training error at alpha = 0, shrinks the tree as alpha grows, keeps the true split and returns a copy."""
    rng = np.random.default_rng(5)
    X = np.array(rng.integers(0, 30, (2000, 5)), np.uint8)
    y = (X[:, 0] > 15).astype(int)
    flip = rng.random(2000) < 0.1
    y[flip] = 1 - y[flip]
    tree = fit(X, y)

    def train_error(t):
        """
        Compute the share of training instances that the tree gets wrong.

        Args:
            t: A fitted tree.

        Returns:
            The training error, a number between 0 and 1.
        """
        return np.mean(t.predict(X) != y)

    assert np.isclose(train_error(tree.prune(0.0)), train_error(tree))
    leaves = [tree.prune(a).n_leaves for a in (0.0, 1e-4, 1e-3, 1e-2)]
    assert leaves == sorted(leaves, reverse=True) and leaves[0] <= tree.n_leaves
    assert tree.prune(1e-2).n_leaves == 2
    assert tree.prune(1.0).n_leaves == 1
    assert tree.n_leaves == fit(X, y).n_leaves


def test_every_split_is_the_best_available():
    """
    Checks by brute force that no candidate split has a lower weighted Gini impurity than the chosen one at any internal node.

    The check uses a sample of the real data with class weights, multiplicities and nominal features.
    """
    d = load_dataset()
    rng = np.random.default_rng(6)
    sample = rng.choice(len(d.y), 600, replace=False)
    enc = TreeEncoder().fit(d.X.iloc[sample])
    X, y = enc.transform(d.X.iloc[sample]).astype(np.int64), d.y[sample]
    mult = rng.integers(0, 3, len(y))
    w = class_weights(y, config.N_CLASSES, beta=1.0)
    tree = ClassificationTree(seed=1).fit(X, y, enc.n_codes, enc.is_nominal, config.N_CLASSES,
                                          class_weight=w, multiplicity=mult)
    weight = mult * w[y]

    def impurity(goes_left, idx):
        """
        Compute the weighted Gini impurity of a split.

        Args:
            goes_left: True for each instance of the node that goes to the left child.
            idx: Indices of the instances in the node.

        Returns:
            The summed impurity of the two children.
        """
        total = 0.0
        for side in (goes_left, ~goes_left):
            dist = np.bincount(y[idx][side], weights=weight[idx][side], minlength=config.N_CLASSES)
            if dist.sum() > 0:
                total += dist.sum() - np.sum(dist ** 2) / dist.sum()
        return total

    def test(f, c, idx):
        """
        Evaluate a candidate test on the instances of a node.

        Args:
            f: Index of the feature.
            c: Code of the test.
            idx: Indices of the instances in the node.

        Returns:
            True for each instance that passes the test.
        """
        return X[idx, f] == c if enc.is_nominal[f] else X[idx, f] <= c

    stack, checked = [(0, np.flatnonzero(mult > 0))], 0
    while stack:
        k, idx = stack.pop()
        if tree.left[k] == LEAF:
            continue
        chosen = impurity(test(tree.feature[k], tree.code[k], idx), idx)
        for f in range(X.shape[1]):
            codes = np.unique(X[idx, f])
            for c in (codes if enc.is_nominal[f] else codes[:-1]):
                assert chosen <= impurity(test(f, c, idx), idx) + 1e-9
        goes = test(tree.feature[k], tree.code[k], idx)
        stack += [(tree.left[k], idx[goes]), (tree.right[k], idx[~goes])]
        checked += 1
    assert checked > 20


def test_matches_sklearn_on_real_data():
    """
    Checks that the tree of this project matches the tree of scikit-learn on real data.

    Both trees get the nominal features as ordered codes. They must choose the same root split at the same threshold and reach the same training accuracy. Their accuracy on unseen data must be close.
    Only the development set is used: 80% to train and 20% as unseen data.
    """
    if not HAVE_SKLEARN:
        return
    from sklearn.tree import DecisionTreeClassifier
    d = load_dataset()
    development, _ = development_generalisation_split(d.y)
    part_train, part_unseen = stratified_holdout(d.y[development], 0.2, config.rng("test_holdout"))
    dev, gen = development[part_train], development[part_unseen]
    enc = TreeEncoder().fit(d.X.iloc[dev])
    Xd, Xg = enc.transform(d.X.iloc[dev]), enc.transform(d.X.iloc[gen])
    yd, yg = d.y[dev], d.y[gen]
    raw_d, raw_g = Xd.astype(float), Xg.astype(float)
    for j, name in enumerate(enc.feature_names):
        if not enc.is_nominal[j]:
            raw_d[:, j], raw_g[:, j] = d.X[name].iloc[dev], d.X[name].iloc[gen]
    # scikit-learn treats every code as an ordered number.
    all_numeric = np.zeros(Xd.shape[1], bool)

    own = ClassificationTree().fit(Xd, yd, enc.n_codes, all_numeric, config.N_CLASSES,
                                    code_values=enc.code_values())
    ref = DecisionTreeClassifier(criterion="gini", random_state=0).fit(raw_d, yd)

    root = own.feature[0]
    assert root == ref.tree_.feature[0]
    assert np.isclose(enc.thresholds[enc.feature_names[root]][own.code[0]], ref.tree_.threshold[0])
    assert np.isclose(np.mean(own.predict(Xd) == yd), ref.score(raw_d, yd))
    assert abs(np.mean(own.predict(Xg) == yg) - ref.score(raw_g, yg)) < 0.01


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_")]
    for name, f in tests:
        f()
        skipped = not HAVE_SKLEARN and "sklearn" in name
        print(f"{'skipped' if skipped else 'passed '}  {name}")
    print(f"\nall {len(tests)} tests passed" + ("" if HAVE_SKLEARN else " (sklearn tests skipped)"))
