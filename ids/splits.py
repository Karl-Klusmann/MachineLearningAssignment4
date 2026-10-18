"""Stratified hold-out split and stratified cross-validation (CV) folds.
Stratified means that every class keeps the same share in every subset.
This matters because the class Worms has only 171 instances, so an unstratified split could leave a fold with almost none.
The file creates the development set and the generalisation set described in the data section of the report.
The tuning scripts use the folds, and the final script uses the generalisation set once.
"""

import numpy as np

from . import config


def stratified_holdout(y: np.ndarray, test_fraction: float, rng: np.random.Generator):
    """
    Split instance indices into a training part and a test part with equal class shares.

    Each class is shuffled on its own. The first round(test_fraction * class size) instances of each class go to the test part.

    Args:
        y: Class label of every instance.
        test_fraction: Share of every class that goes to the test part, between 0 and 1.
        rng: NumPy random generator that does the shuffling.

    Returns:
        A pair of sorted index arrays: the training indices and the test indices.
    """
    test = []
    for c in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == c))
        test.append(members[: int(round(test_fraction * len(members)))])
    test = np.sort(np.concatenate(test))
    train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
    return train, test


def stratified_kfold(y: np.ndarray, k: int, rng: np.random.Generator):
    """
    Split instance indices into k folds for stratified cross-validation (CV).

    The shuffled instances of each class are dealt to the folds in turn, like cards.
    The dealing position carries over from one class to the next, so the fold sizes differ by at most one.
    Every instance is in the validation part of exactly one fold.

    Args:
        y: Class label of every instance.
        k: Number of folds.
        rng: NumPy random generator that does the shuffling.

    Returns:
        A list of k pairs of index arrays. Each pair holds the training indices and the validation indices of one fold.
    """
    fold_of = np.empty(len(y), dtype=np.int64)
    position = 0
    for c in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == c))
        fold_of[members] = (position + np.arange(len(members))) % k
        position += len(members)
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def development_generalisation_split(y: np.ndarray):
    """
    Make the one fixed split of the project into a development set and a generalisation set.

    All tuning uses the development set. The generalisation set is used once, for the final comparison.
    The split is the same on every call because it uses the stream rng("holdout").

    Args:
        y: Class label of every instance.

    Returns:
        A pair of index arrays: the development indices and the generalisation indices.
    """
    return stratified_holdout(y, config.GENERALISATION_FRACTION, config.rng("holdout"))
