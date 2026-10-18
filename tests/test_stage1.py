"""Tests for the data layer: loading, splitting and encoding.
They check ids/data.py, ids/splits.py and ids/encoding.py against known counts and small hand-made examples.
The data section of the report relies on the counts that these tests check.
Run: python -m tests.test_stage1. The test functions also run under pytest if it is installed.
"""

import numpy as np
import pandas as pd

from ids import config
from ids.data import identical_feature_groups, load_dataset
from ids.encoding import TreeEncoder, equal_frequency_thresholds
from ids.splits import development_generalisation_split, stratified_holdout, stratified_kfold

_DATA = None


def dataset():
    """
    Load the data set on the first call and return the same object on later calls.

    Returns:
        The Dataset of ids/data.py.
    """
    global _DATA
    if _DATA is None:
        _DATA = load_dataset()
    return _DATA


def test_counts_match_known_values():
    """Checks that the row counts after loading and de-duplication match the known values of the data set."""
    d = dataset()
    assert d.n_raw == 257_673
    assert d.n_duplicates == 94_928
    assert len(d.y) == 162_745
    assert d.X.shape[1] == 42 and "id" not in d.X.columns
    assert d.ambiguous.sum() == 11_506


def test_no_duplicate_rows_remain():
    """Checks that no two rows of the cleaned data are identical in features and label."""
    d = dataset()
    table = d.X.assign(label=d.y)
    assert not table.duplicated().any()


def test_identical_feature_groups_on_toy_data():
    """Checks the groups, the ambiguous flags and the majority labels on a five-row example."""
    X = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "x", "y", "y", np.nan]})
    y = np.array([0, 1, 2, 2, 0])
    group, ambiguous, majority = identical_feature_groups(X, y)
    assert group[0] == group[1] and group[2] == group[3] and len(set(group)) == 3
    assert ambiguous.tolist() == [True, True, False, False, False]
    # In the first group the labels 0 and 1 tie, and the lower label wins.
    assert majority.tolist() == [0, 0, 2, 2, 0]


def test_holdout_is_disjoint_complete_and_stratified():
    """Checks that the development and generalisation parts do not overlap, cover all instances and hold a fifth of every class in the generalisation part."""
    y = dataset().y
    dev, gen = development_generalisation_split(y)
    assert len(np.intersect1d(dev, gen)) == 0
    assert len(dev) + len(gen) == len(y)
    for c in range(config.N_CLASSES):
        n_c = np.sum(y == c)
        assert abs(np.sum(y[gen] == c) - config.GENERALISATION_FRACTION * n_c) <= 0.5


def test_holdout_is_reproducible():
    """Checks that the split is the same on every call and differs for another random stream."""
    y = dataset().y
    a = development_generalisation_split(y)[1]
    b = development_generalisation_split(y)[1]
    assert np.array_equal(a, b)
    other = stratified_holdout(y, 0.2, config.rng("something else"))[1]
    assert not np.array_equal(a, other)


def test_kfold_partitions_and_stratifies():
    """Checks that every instance is validated once, the folds differ in size by at most one and every class is spread evenly over the folds."""
    y = dataset().y
    folds = stratified_kfold(y, 5, config.rng("folds"))
    validation = np.concatenate([v for _, v in folds])
    # Together, the validation parts hold every instance exactly once.
    assert np.array_equal(np.sort(validation), np.arange(len(y)))
    sizes = [len(v) for _, v in folds]
    assert max(sizes) - min(sizes) <= 1
    for train, val in folds:
        assert len(np.intersect1d(train, val)) == 0
        for c in range(config.N_CLASSES):
            assert abs(np.sum(y[val] == c) - np.sum(y == c) / 5) <= 1


def test_few_distinct_values_are_binned_losslessly():
    """Checks that a feature with few distinct values gets one threshold midway between each pair of neighbouring values."""
    values = np.array([0, 0, 1, 5, 5, 9], dtype=float)
    assert np.allclose(equal_frequency_thresholds(values, 255), [0.5, 3.0, 7.0])


def test_many_distinct_values_respect_max_bins_and_isolate_a_spike():
    """Checks that binning respects the bin limit, gives a spike of zeros its own bin and fills the other bins about equally."""
    rng = np.random.default_rng(0)
    values = np.concatenate([np.zeros(5000), rng.exponential(size=5000)])
    thresholds = equal_frequency_thresholds(values, 16)
    assert len(thresholds) + 1 <= 16
    # The zeros get a bin of their own, so the first threshold lies between 0 and the smallest positive value.
    assert 0 < thresholds[0] < values[values > 0].min()
    codes = np.searchsorted(thresholds, values[values > 0])
    shares = np.bincount(codes)[1:] / codes.size
    assert shares.max() < 2.5 * shares.min()


def test_encoding_preserves_order_and_split_equivalence():
    """Checks that codes keep the order of the values and that a test on a code gives the same split as the test on the value, for the lossless and the 255-bin encoding."""
    d = dataset()
    dev, gen = development_generalisation_split(d.y)
    for max_bins in (None, 255):
        encoder = TreeEncoder(max_bins).fit(d.X.iloc[dev])
        codes = encoder.transform(d.X.iloc[gen])
        for j, name in enumerate(encoder.feature_names):
            if encoder.is_nominal[j]:
                continue
            raw = d.X[name].iloc[gen].to_numpy()
            order = np.argsort(raw, kind="stable")
            assert np.all(np.diff(codes[order, j].astype(np.int64)) >= 0)
            b = len(encoder.thresholds[name]) // 2
            if len(encoder.thresholds[name]):
                assert np.array_equal(codes[:, j] <= b, raw <= encoder.thresholds[name][b])


def test_default_encoding_is_lossless():
    """Checks that every numeric feature gets one code per distinct training value."""
    d = dataset()
    dev, _ = development_generalisation_split(d.y)
    encoder = TreeEncoder().fit(d.X.iloc[dev])
    for name in d.numeric_features:
        assert len(encoder.thresholds[name]) + 1 == d.X[name].iloc[dev].nunique()


def test_nominal_missing_and_unseen_categories():
    """Checks that unseen categories, missing values and values beyond the training range get the intended codes."""
    train = pd.DataFrame({"proto": ["tcp", "udp", "tcp"], "service": ["http", np.nan, "dns"],
                          "state": ["FIN", "INT", "FIN"], "dur": [0.1, 0.2, 0.3]})
    test = pd.DataFrame({"proto": ["sctp"], "service": [np.nan], "state": ["FIN"], "dur": [9.0]})
    encoder = TreeEncoder().fit(train)
    codes = encoder.transform(test)[0]
    # An unseen category gets the extra code that follows the known categories.
    assert codes[0] == len(encoder.categories["proto"])
    assert encoder.categories["service"][codes[1]] == config.MISSING_CATEGORY
    assert codes[3] == len(encoder.thresholds["dur"])
    assert encoder.describe_test(0, 0) == "proto = tcp"


def test_encoder_uses_training_data_only():
    """Checks that the thresholds come from the training data and that outside values get the first or the last code."""
    train = pd.DataFrame({"dur": [1.0, 2.0, 3.0]})
    encoder = TreeEncoder().fit(train)
    assert np.allclose(encoder.thresholds["dur"], [1.5, 2.5])
    assert encoder.transform(pd.DataFrame({"dur": [100.0, -5.0]})).ravel().tolist() == [2, 0]


def test_code_values_cover_every_code():
    """
    Checks that every feature has one value per code, nominal features included.

    A tree that treats nominal codes as ordered numbers then never reads the values of another feature.
    """
    X = pd.DataFrame({"proto": ["tcp", "udp", "tcp", "arp"], "dur": [0.5, 0.1, 0.5, 2.0],
                      "service": ["http", None, "dns", "http"]})
    for enc in (TreeEncoder().fit(X), TreeEncoder(2).fit(X)):
        values, offsets = enc.code_values()
        assert np.array_equal(np.diff(offsets), enc.n_codes)
    exact = TreeEncoder().fit(X)
    values, offsets = exact.code_values()
    j = exact.feature_names.index("dur")
    assert np.array_equal(values[offsets[j]:offsets[j + 1]], [0.1, 0.5, 2.0])


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_")]
    for name, f in tests:
        f()
        print(f"passed  {name}")
    print(f"\nall {len(tests)} tests passed")
