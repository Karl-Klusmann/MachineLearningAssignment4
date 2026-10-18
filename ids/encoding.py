"""Turns the feature table into the whole-number codes that the classification tree uses.
A numeric feature becomes the rank of its value among the distinct training values. A nominal feature becomes a category number.
Codes keep the order of the values, so a test "code <= b" is the same as a test "value <= threshold" on the original values.
The report describes this in the section on tree induction.
The encoder is fitted on training data only. The out-of-bag (OOB) experiments fit it on the whole development set, which uses no labels.
"""

import numpy as np
import pandas as pd

from . import config


class TreeEncoder:
    """
    Finds the codes of every feature from training data and applies them to any data.

    With max_bins=None, every distinct training value gets its own code. No split that an exact tree could choose is lost.
    An integer limits each numeric feature to that many bins. This is only used to measure what binning would cost.
    """

    def __init__(self, max_bins: int | None = None):
        """
        Store the bin limit.

        Args:
            max_bins: Largest number of bins per numeric feature, or None for one code per distinct value.

        Raises:
            ValueError: If max_bins is smaller than 2.
        """
        if max_bins is not None and max_bins < 2:
            raise ValueError("max_bins must be at least 2")
        self.max_bins = max_bins

    def fit(self, X: pd.DataFrame) -> "TreeEncoder":
        """
        Find the thresholds of the numeric features and the categories of the nominal features in the training data.

        A nominal feature gets one extra code for categories that never occur in the training data.
        The smallest whole-number type that holds the largest code is used for the encoded matrix.

        Args:
            X: Table of training features. Columns named in config.NOMINAL_FEATURES are treated as nominal.

        Returns:
            The fitted encoder.
        """
        self.feature_names = list(X.columns)
        self.is_nominal = np.array([c in config.NOMINAL_FEATURES for c in self.feature_names])
        self.thresholds = {}
        self.categories = {}
        self._distinct = {}

        for name, nominal in zip(self.feature_names, self.is_nominal):
            if nominal:
                values = X[name].fillna(config.MISSING_CATEGORY).astype(str)
                self.categories[name] = sorted(values.unique())
            else:
                values = X[name].to_numpy(float)
                self._distinct[name] = np.unique(values)
                self.thresholds[name] = equal_frequency_thresholds(values, self.max_bins)

        self.n_codes = np.array([
            len(self.categories[n]) + 1 if nom else len(self.thresholds[n]) + 1
            for n, nom in zip(self.feature_names, self.is_nominal)])
        largest = self.n_codes.max()
        self.dtype = np.uint8 if largest <= 2**8 else np.uint16 if largest <= 2**16 else np.uint32
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Encode a table of features with the codes found in fit.

        A numeric value gets the number of thresholds strictly below it. A missing or unseen nominal value gets the reserved code.

        Args:
            X: Table with the same columns as the table given to fit.

        Returns:
            Matrix of codes with one row per instance and one column per feature.

        Raises:
            ValueError: If the columns differ from those of fit or a numeric feature has a missing value.
        """
        if list(X.columns) != self.feature_names:
            raise ValueError("columns differ from those seen during fit")
        codes = np.empty((len(X), len(self.feature_names)), dtype=self.dtype)

        for j, (name, nominal) in enumerate(zip(self.feature_names, self.is_nominal)):
            if nominal:
                values = X[name].fillna(config.MISSING_CATEGORY).astype(str)
                index = pd.Categorical(values, categories=self.categories[name]).codes
                unseen = len(self.categories[name])
                codes[:, j] = np.where(index < 0, unseen, index)
            else:
                values = X[name].to_numpy(float)
                if np.isnan(values).any():
                    raise ValueError(f"missing value in numeric feature '{name}'")
                codes[:, j] = np.searchsorted(self.thresholds[name], values, side="left")
        return codes

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the encoder on a table and encode the same table.

        Args:
            X: Table of training features.

        Returns:
            Matrix of codes with one row per instance and one column per feature.
        """
        return self.fit(X).transform(X)

    def code_values(self):
        """
        Give the value that each code stands for, packed for all features.

        Code c of feature j stands for values[offsets[j] + c].
        For a lossless numeric feature this is the c-th distinct training value. The tree uses these values to place a threshold midway between two values.
        A nominal feature or a binned feature uses the code number itself.

        Returns:
            A pair: the array of all values joined end to end, and the array of start positions of each feature in it.
        """
        values, offsets = [], [0]
        for j, name in enumerate(self.feature_names):
            if self.is_nominal[j] or self.max_bins is not None:
                v = np.arange(self.n_codes[j], dtype=float)
            else:
                v = self._distinct[name]
            values.append(v)
            offsets.append(offsets[-1] + len(v))
        return np.concatenate(values), np.array(offsets, dtype=np.int64)

    def describe_test(self, j: int, code: int) -> str:
        """
        Write the test that sends an instance to the left child in words.

        Args:
            j: Index of the feature.
            code: Code used in the test.

        Returns:
            Text such as "proto = tcp" for a nominal feature or "dur <= 0.5" for a numeric feature.
        """
        name = self.feature_names[j]
        if self.is_nominal[j]:
            return f"{name} = {self.categories[name][code]}"
        return f"{name} <= {self.thresholds[name][code]:.6g}"


def equal_frequency_thresholds(values: np.ndarray, max_bins: int | None) -> np.ndarray:
    """
    Choose the split thresholds of one numeric feature.

    With max_bins=None, or with at most max_bins distinct values, every gap between neighbouring distinct values gets a threshold. No information is lost.
    Otherwise the distinct values are merged into bins that hold about the same number of instances.
    After each bin closes, the target size is computed again from the instances that remain.
    A value that holds a large share of the data, such as many zeros, then gets a bin of its own.
    Each threshold lies midway between the last value of one bin and the first value of the next.

    Args:
        values: Values of the feature in the training data.
        max_bins: Largest number of bins, or None for no limit.

    Returns:
        Sorted array of thresholds. The feature has one more code than it has thresholds.
    """
    distinct, counts = np.unique(values, return_counts=True)
    if max_bins is None or len(distinct) <= max_bins:
        cut_after = np.arange(len(distinct) - 1)
    else:
        cut_after = []
        remaining, bins_left, filled = counts.sum(), max_bins, 0
        for i, count in enumerate(counts[:-1]):
            filled += count
            if filled >= remaining / bins_left:
                cut_after.append(i)
                remaining -= filled
                bins_left -= 1
                filled = 0
                if bins_left == 1:
                    break
        cut_after = np.array(cut_after, dtype=np.int64)

    lower, upper = distinct[cut_after], distinct[cut_after + 1]
    # Rounding can place the midpoint on the upper value, so the next line falls back to the lower value.
    midpoint = lower + (upper - lower) / 2
    return np.where(midpoint < upper, midpoint, lower)
