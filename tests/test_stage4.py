"""Tests for the bias-variance decomposition, the k-nearest-neighbour (k-NN) classifier and the significance tests.
They check ids/bias_variance.py, ids/knn.py and ids/stats.py against hand-computed values and brute-force computations.
Experiments E5 and E6 and the k-NN tuning of the report rely on these functions.
Run: python -m tests.test_stage4.
"""

import numpy as np

from ids import config
from ids.bias_variance import decompose
from ids.knn import KNNEncoder, nearest_neighbours, vote
from ids.stats import bootstrap_macro_f1_difference, holm, mcnemar


def test_bias_variance_on_hand_computed_example():
    """Checks the loss, bias and variance terms on an example with four models and two test instances."""
    y = np.array([0, 1])
    preds = np.array([[0, 0],
                      [0, 0],
                      [1, 1],
                      [0, 2]])
    d = decompose(preds, y, 3)
    assert np.isclose(d["loss"], 0.5)
    assert np.isclose(d["bias"], 0.5)
    assert np.isclose(d["unbiased_variance"], 0.125)
    assert np.isclose(d["biased_variance"], 0.125)
    assert np.isclose(d["net_variance"], 0.0)
    assert d["per_class"]["bias"] == [0.0, 1.0, None]


def test_loss_equals_bias_plus_net_variance():
    """Checks that the loss equals the bias plus the net variance, and that the plain loss equals the error rate."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500)
    preds = np.where(rng.random((7, 500)) < 0.6, y, rng.integers(0, 4, (7, 500)))
    for balanced in (False, True):
        d = decompose(preds, y, 4, balanced)
        assert np.isclose(d["loss"], d["bias"] + d["net_variance"])
    plain = decompose(preds, y, 4)
    assert np.isclose(plain["loss"], np.mean(preds != y))


def test_balanced_loss_is_one_minus_mean_balanced_accuracy():
    """Checks that the class-balanced loss equals one minus the mean balanced accuracy."""
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1, 2], [300, 50, 10])
    preds = np.where(rng.random((5, len(y))) < 0.7, y, 0)
    d = decompose(preds, y, 3, balanced=True)
    bal_acc = [np.mean([np.mean(p[y == c] == c) for c in range(3)]) for p in preds]
    assert np.isclose(d["loss"], 1 - np.mean(bal_acc))


def test_neighbours_match_brute_force():
    """Checks that the neighbour search returns the same neighbours and distances as a full sort, and that a tie in distance keeps the earlier training instance."""
    rng = np.random.default_rng(0)
    X = rng.random((300, 5)).astype(np.float32)
    X[50] = X[10]
    Q = rng.random((40, 5)).astype(np.float32)
    Q[0] = X[10]
    idx, dist = nearest_neighbours(Q, X, K=7)
    full = np.abs(Q[:, None, :] - X[None, :, :]).sum(axis=2)
    expected = np.argsort(full, axis=1, kind="stable")[:, :7]
    assert np.array_equal(idx, expected)
    assert np.allclose(dist, np.take_along_axis(full, expected, axis=1), atol=1e-5)
    assert idx[0, 0] == 10 and idx[0, 1] == 50 and dist[0, 0] == 0


def test_voting_rules():
    """Checks uniform, distance-weighted and class-weighted voting and the tie-break on small examples."""
    labels = np.array([[1, 0, 0, 2]])
    dist = np.array([[0.0, 1.0, 1.0, 3.0]])
    assert vote(labels, dist, 3, 3)[0] == 0
    assert vote(labels, dist, 1, 3)[0] == 1
    assert vote(labels, dist, 4, 3, "distance")[0] == 0
    weights = np.array([0.1, 1.0, 1.0])
    assert vote(labels, dist, 3, 3, class_weight=weights)[0] == 1
    tie = np.array([[2, 1, 1, 2]])
    assert vote(tie, dist, 4, 3)[0] == 2


def test_knn_encoder_follows_the_a2_pipeline():
    """Checks that the preprocessing of Assignment 2 drops the listed feature, log-transforms the skewed feature and pools rare and unseen categories."""
    import pandas as pd
    X = pd.DataFrame({
        # Skewness 2.67, above the limit of 2, so this feature is log-transformed.
        "dur": ([0.0] * 9 + [1000.0]) * 20,
        "rate": [1.0, 2.0, 3.0, 4.0] * 50,
        "dloss": [5.0] * 200,
        "proto": ["tcp"] * 199 + ["rare"],
        "service": [None, "http"] * 100,
        "state": ["FIN"] * 200,
    })
    enc = KNNEncoder().fit(X)
    assert "dloss" not in enc.numeric and enc.log_features == ["dur"]
    assert enc.levels["proto"] == ["rare", "tcp", "infrequent"]
    assert enc.levels["service"] == ["http", config.MISSING_CATEGORY, "infrequent"]
    Z = enc.transform(pd.DataFrame({**{c: X[c].iloc[:1] for c in X.columns}, "proto": ["unseen"]}))
    assert Z.shape[1] == len(enc.feature_names) and Z[0].min() >= 0
    assert Z[0, enc.feature_names.index("proto=infrequent")] == 1.0


def test_mcnemar():
    """Checks McNemar's test against a hand-computed statistic and p-value."""
    y = np.zeros(30, int)
    a = np.zeros(30, int)
    b = np.zeros(30, int)
    b[:10] = 1
    r = mcnemar(y, a, b)
    assert (r["b"], r["c"]) == (10, 0)
    assert np.isclose(r["chi2"], 81 / 10)
    # The tail of a chi-squared distribution with one degree of freedom at 8.1.
    assert np.isclose(r["p"], 0.004427, atol=1e-5)


def test_bootstrap_and_holm():
    """Checks that the bootstrap interval and p-value agree with a clear difference, that equal models give a zero difference and that Holm's correction gives the expected values."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    good = np.where(rng.random(600) < 0.9, y, (y + 1) % 3)
    bad = np.where(rng.random(600) < 0.6, y, (y + 1) % 3)
    r = bootstrap_macro_f1_difference(y, good, bad, np.random.default_rng(1), 300, 3)
    assert r["ci_low"] < r["difference"] < r["ci_high"] and r["ci_low"] > 0 and r["p"] < 0.01
    same = bootstrap_macro_f1_difference(y, good, good, np.random.default_rng(1), 100, 3)
    assert same["difference"] == 0 and same["p"] == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_")]
    for name, f in tests:
        f()
        print(f"passed  {name}")
    print(f"\nall {len(tests)} tests passed")
