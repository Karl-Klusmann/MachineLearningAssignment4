"""Tests for the random forest.
They check ids/forest.py and the selection rule of ids/experiments.py on small synthetic data, so they run in seconds.
The tests cover the sampling, the voting, the out-of-bag (OOB) estimate, the strength and correlation, and the result that does not depend on the number of threads.
The implementation section of the report relies on them.
Run: python -m tests.test_stage3.
"""

import numpy as np

from ids.forest import (RandomForest, breiman_strength_correlation, draw_multiplicities,
                        resolve_max_features, vote)
from ids.metrics import macro_f1
from ids.tree import ClassificationTree, _apply, class_weights


def noisy_data(n=3000, n_features=6, noise=0.2, seed=0, n_classes=3):
    """
    Make integer-coded data whose class depends on two features, plus label noise.

    Args:
        n: Number of instances.
        n_features: Number of features.
        noise: Share of labels that are replaced by a random class.
        seed: Seed of the random generator.
        n_classes: Number of classes.

    Returns:
        A pair: the matrix of codes and the labels.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 20, size=(n, n_features)).astype(np.uint8)
    y = ((X[:, 0] >= 10).astype(int) + (X[:, 1] >= 14)) % n_classes
    flip = rng.random(n) < noise
    y[flip] = rng.integers(0, n_classes, flip.sum())
    return X, y


def fit_forest(X, y, n_classes=3, **kwargs):
    """
    Fit a forest on integer-coded data in which every feature has 20 codes and none is nominal.

    Args:
        X: Matrix of codes.
        y: Class label of every instance.
        n_classes: Number of classes.
        **kwargs: Settings for RandomForest. The keys eval_sets, keep_trees and diagnostics go to fit.

    Returns:
        The fitted RandomForest.
    """
    fit_args = {k: kwargs.pop(k) for k in ("eval_sets", "keep_trees", "diagnostics") if k in kwargs}
    n_codes = np.full(X.shape[1], 20)
    nominal = np.zeros(X.shape[1], bool)
    return RandomForest(**kwargs).fit(X, y, n_codes, nominal, n_classes, **fit_args)


def members_of(y, n_classes):
    """
    List the indices of the instances of every class.

    Args:
        y: Class label of every instance.
        n_classes: Number of classes.

    Returns:
        List with one index array per class.
    """
    return [np.flatnonzero(y == c) for c in range(n_classes)]


def test_bootstrap_multiplicities():
    """Checks that a bootstrap sample has the right size and that the chance of never drawing an instance is about 1/e, or exp(-0.5) for half-size samples."""
    y = np.zeros(100_000, int)
    rng = np.random.default_rng(0)
    full = draw_multiplicities(y, members_of(y, 1), "bootstrap", 1.0, rng)
    assert full.sum() == len(y)
    assert abs(np.mean(full == 0) - np.exp(-1)) < 0.005
    half = draw_multiplicities(y, members_of(y, 1), "bootstrap", 0.5, rng)
    assert half.sum() == len(y) // 2
    assert abs(np.mean(half == 0) - np.exp(-0.5)) < 0.005


def test_balanced_and_capped_sampling():
    """Checks that balanced sampling draws the size of the smallest class from every class and that capped sampling reduces only the large classes."""
    y = np.repeat([0, 1, 2], [5000, 300, 40])
    members = members_of(y, 3)
    rng = np.random.default_rng(0)
    balanced = draw_multiplicities(y, members, "balanced", None, rng)
    assert np.bincount(y, weights=balanced).tolist() == [40, 40, 40]
    capped = draw_multiplicities(y, members, "capped", 1000, rng)
    assert np.bincount(y, weights=capped).tolist() == [1000, 300, 40]


def test_max_features():
    """Checks the conversion of the setting for features per node into a number."""
    assert resolve_max_features("sqrt", 42) == 6
    assert resolve_max_features(None, 42) == 42
    assert resolve_max_features(100, 42) == 42
    assert resolve_max_features(0, 42) == 1


def test_hard_vote_ties_are_broken_by_soft_scores():
    """Checks that a tie in hard voting goes to the class with the larger soft score."""
    soft = np.array([[1.4, 1.6, 0.0], [2.0, 0.5, 0.5]])
    hard = np.array([[1, 1, 0], [2, 0, 1]], float)
    count = np.array([2.0, 3.0])
    soft_pred, hard_pred = vote(soft, hard, count)
    assert soft_pred.tolist() == [1, 0]
    assert hard_pred.tolist() == [1, 0]


def test_each_tree_is_the_tree_its_sample_defines():
    """Checks that tree t of the forest equals a separate tree grown on the sample and seed of tree t."""
    X, y = noisy_data()
    forest = fit_forest(X, y, n_trees=4, max_features=2, seed=7, n_jobs=1)
    members = members_of(y, 3)
    for t in range(4):
        mult, seed = forest.tree_sample(t, y, members)
        w = class_weights(y, 3, 1.0, multiplicity=mult)
        tree = ClassificationTree(max_features=2, seed=seed).fit(
            X, y, np.full(6, 20), np.zeros(6, bool), 3, class_weight=w, multiplicity=mult)
        assert np.allclose(forest.trees_[t].proba[0], tree.value[0] / tree.value[0].sum())
        mine = forest.trees_[t].proba[tree.apply(X)]
        assert np.allclose(mine, tree.predict_proba(X), atol=1e-6)


def test_oob_uses_only_trees_that_did_not_see_the_instance():
    """Checks that the out-of-bag votes of every instance come only from the trees whose sample left the instance out."""
    X, y = noisy_data(n=800)
    forest = fit_forest(X, y, n_trees=15, seed=3, n_jobs=1)
    members = members_of(y, 3)
    soft, count = np.zeros((len(y), 3)), np.zeros(len(y))
    for t, tr in enumerate(forest.trees_):
        out = forest.tree_sample(t, y, members)[0] == 0
        p = tr.proba[_apply(X, tr.feature, tr.code, tr.left, tr.right, forest.is_nominal)]
        soft[out] += p[out]
        count[out] += 1
    assert np.array_equal(count, forest.oob_count_)
    assert np.allclose(soft / np.maximum(count, 1)[:, None], forest.oob_proba_, atol=1e-6)


def test_results_do_not_depend_on_the_number_of_threads():
    """Checks that the forest is identical for one and for three threads and changes with the seed."""
    X, y = noisy_data()
    a = fit_forest(X, y, n_trees=12, seed=1, n_jobs=1, diagnostics=True)
    b = fit_forest(X, y, n_trees=12, seed=1, n_jobs=3, diagnostics=True)
    assert np.array_equal(a.oob_proba_, b.oob_proba_)
    assert a.oob_curve_ == b.oob_curve_
    assert a.diagnostics_ == b.diagnostics_
    c = fit_forest(X, y, n_trees=12, seed=2, n_jobs=3)
    assert not np.array_equal(a.oob_proba_, c.oob_proba_)


def test_eval_sets_match_predictions_of_kept_trees():
    """Checks that predictions made while growing agree with the predictions of the kept trees."""
    X, y = noisy_data()
    X_new, y_new = noisy_data(n=500, seed=9)
    forest = fit_forest(X, y, n_trees=10, seed=0, eval_sets={"new": (X_new, y_new)})
    assert np.allclose(forest.eval_proba_["new"], forest.predict_proba(X_new))
    for voting in ("soft", "hard"):
        assert np.array_equal(forest.eval_predict("new", voting), forest.predict(X_new, voting))
    last = forest.eval_curves_["new"]["soft"][-1]
    assert np.isclose(last, macro_f1(y_new, forest.predict(X_new), 3))
    assert forest.oob_curve_["trees"][-1] == 10


def test_forest_beats_a_single_tree_on_noisy_data():
    """Checks that the forest is clearly more accurate than one tree on noisy data and that the OOB accuracy is close to the accuracy on new data."""
    X, y = noisy_data(n=4000, noise=0.3)
    X_new, y_new = noisy_data(n=4000, noise=0.3, seed=5)
    tree = ClassificationTree(seed=0).fit(X, y, np.full(6, 20), np.zeros(6, bool), 3)
    forest = fit_forest(X, y, n_trees=60, max_features=2, seed=0, class_beta=0.0)
    tree_acc = np.mean(tree.predict(X_new) == y_new)
    forest_acc = np.mean(forest.predict(X_new) == y_new)
    oob = forest.oob_predict()
    oob_acc = np.mean(oob == y)
    assert forest_acc > tree_acc + 0.05
    assert abs(oob_acc - forest_acc) < 0.03


def test_feature_budget_rules():
    """
    Checks the two rules for the number of features per node.

    The class depends only on feature 0, feature 1 is noise and features 2 to 5 are constant.
    With m = 2 varying features, every node sees both useful candidates and one split is enough.
    Under the scikit-learn rule, the two examined features are often constant ones, so splits on the noise feature appear.
    """
    rng = np.random.default_rng(0)
    X = np.zeros((400, 6), np.uint8)
    X[:, 0] = rng.integers(0, 20, 400)
    X[:, 1] = rng.integers(0, 20, 400)
    y = (X[:, 0] >= 10).astype(int)
    args = (X, y, np.full(6, 20), np.zeros(6, bool), 2)
    own = [ClassificationTree(max_features=2, seed=s).fit(*args).n_leaves for s in range(10)]
    sk_rule = [ClassificationTree(max_features=2, seed=s, count_constant=True).fit(*args).n_leaves
               for s in range(10)]
    # One split is enough, so every tree has two leaves.
    assert own == [2] * 10
    assert max(sk_rule) > 2


def test_pruning_inside_the_forest():
    """Checks that pruned trees have far fewer leaves and that alpha = 0 means no pruning."""
    X, y = noisy_data()
    grown = fit_forest(X, y, n_trees=5, seed=0)
    pruned = fit_forest(X, y, n_trees=5, seed=0, prune_alpha=0.01)
    assert np.mean(pruned.tree_leaves_) < np.mean(grown.tree_leaves_) / 5
    zero = fit_forest(X, y, n_trees=5, seed=0, prune_alpha=0.0)
    assert zero.tree_leaves_ == grown.tree_leaves_


def test_sample_size_is_checked():
    """Checks that invalid sample sizes raise an error and that a missing size for a bootstrap sample means the full size."""
    y = np.repeat([0, 1], [50, 10])
    members, rng = members_of(y, 2), np.random.default_rng(0)
    assert draw_multiplicities(y, members, "bootstrap", None, rng).sum() == 60
    for sampling, size in (("bootstrap", 1.5), ("balanced", 1.0), ("capped", None), ("capped", 2.5)):
        try:
            draw_multiplicities(y, members, sampling, size, rng)
        except ValueError:
            continue
        raise AssertionError(f"{sampling} accepted sample_size={size}")


def test_selection_rule_is_deterministic():
    """Checks that the selection rule picks the simplest configuration among those within one standard deviation of the best."""
    from ids.experiments import by_leaves, by_m, select
    results = [{"macro_f1_mean": 0.600, "macro_f1_sd": 0.002, "leaves_mean": 900, "params": {"max_features": 9}},
               {"macro_f1_mean": 0.601, "macro_f1_sd": 0.002, "leaves_mean": 500, "params": {"max_features": 13}},
               {"macro_f1_mean": 0.595, "macro_f1_sd": 0.002, "leaves_mean": 100, "params": {"max_features": 4}}]
    assert select(results, cost=by_leaves) == 1
    assert select(results, cost=by_m) == 0


def test_strength_and_correlation_match_a_direct_computation():
    """Checks strength, correlation, bound and OOB error against a slow direct computation with loops."""
    rng = np.random.default_rng(0)
    n_trees, n, C = 7, 60, 3
    y = rng.integers(0, C, n)
    pred = np.where(rng.random((n_trees, n)) < 0.6, y, rng.integers(0, C, (n_trees, n)))
    # A value of -1 means that the instance was in the sample of the tree.
    pred[rng.random((n_trees, n)) < 0.4] = -1
    got = breiman_strength_correlation(pred.astype(np.int8), y, C)

    margins, j_star, covered = [], [], []
    for i in range(n):
        votes = [np.sum(pred[:, i] == c) for c in range(C)]
        total = np.sum(pred[:, i] >= 0)
        if total == 0:
            j_star.append(-1)
            continue
        Q = np.array(votes) / total
        wrong = [c for c in range(C) if c != y[i]]
        j = wrong[int(np.argmax(Q[wrong]))]
        j_star.append(j)
        covered.append(i)
        margins.append(Q[y[i]] - Q[j])
    margins = np.array(margins)
    s = margins.mean()
    sds = []
    for t in range(n_trees):
        idx = [i for i in covered if pred[t, i] >= 0]
        p1 = np.mean([pred[t, i] == y[i] for i in idx])
        p2 = np.mean([pred[t, i] == j_star[i] for i in idx])
        sds.append(np.sqrt(p1 + p2 - (p1 - p2) ** 2))
    rho = margins.var() / np.mean(sds) ** 2
    assert np.isclose(got["strength"], s)
    assert np.isclose(got["correlation"], rho)
    assert np.isclose(got["bound"], rho * (1 - s ** 2) / s ** 2)
    assert np.isclose(got["oob_error"], np.mean(margins < 0))


def test_undefined_bound_is_none():
    """Checks that the bound is None when the strength is not positive."""
    y = np.repeat([0, 1], [5, 5])
    pred = np.tile(1 - y, (3, 1)).astype(np.int8)
    d = breiman_strength_correlation(pred, y, 2)
    assert d["strength"] == -1.0 and d["bound"] is None


def test_balanced_strength_weights_classes_equally():
    """Checks that the class-balanced strength gives every class the same weight."""
    y = np.repeat([0, 1], [90, 10])
    pred = np.tile(np.where(y == 0, 0, 0), (5, 1)).astype(np.int8)
    plain = breiman_strength_correlation(pred, y, 2)
    balanced = breiman_strength_correlation(pred, y, 2, balanced=True)
    assert np.isclose(plain["strength"], 0.9 * 1 + 0.1 * -1)
    assert np.isclose(balanced["strength"], 0.0)


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_")]
    for name, f in tests:
        f()
        print(f"passed  {name}")
    print(f"\nall {len(tests)} tests passed")
