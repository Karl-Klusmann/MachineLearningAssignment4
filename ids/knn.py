"""The k-nearest-neighbour (k-NN) classifier, written from first principles, with the preprocessing of Assignment 2.
A k-NN classifier gives a new instance the majority class among the k training instances that lie closest to it.
The file has the preprocessing, a parallel search for the nearest neighbours and the voting rules: uniform, distance-weighted and class-weighted.
The report describes the classifier in the section on baseline models. The script tune_knn.py tunes it, and e6_final.py compares it with the forest.
"""

import numpy as np
import pandas as pd
from numba import njit, prange

from . import config

A2_REDUNDANT = (
    "is_ftp_login",
    "dloss",
    "sloss",
    "dwin",
    "ct_srv_dst",
    "dpkts",
    "spkts",
    "ct_src_dport_ltm",
    "ct_dst_src_ltm",
    "tcprtt",
    "is_sm_ips_ports",
    "ct_dst_ltm",
)
SKEW_LIMIT = 2.0
RARE_SHARE = 0.005
INFREQUENT = "infrequent"


def skewness(x):
    """
    Compute the skewness of a sample. Skewness measures how lopsided the distribution is. A long tail to the right gives a large positive value.

    Args:
        x: Values of one feature.

    Returns:
        The skewness, or 0 if all values are equal.
    """
    x = np.asarray(x, float)
    sd = x.std()
    return 0.0 if sd == 0 else float(np.mean((x - x.mean()) ** 3) / sd ** 3)


class KNNEncoder:
    """
    The preprocessing of Assignment 2 as a fit and transform object. It is fitted on training data only.

    It drops 12 features that are almost perfectly correlated with a retained feature.
    Numeric features with no negative value and a skewness above 2 are transformed with log(1 + x), so that a few huge values do not squeeze the rest.
    Numeric features are scaled to the range 0 to 1 with the training minimum and range.
    A missing service becomes its own category. Categories held by fewer than 0.5% of the training instances are pooled into one "infrequent" category, which also takes categories that the training data never showed.
    The nominal features are then one-hot encoded: one column per category with 1 for the category of the instance and 0 otherwise.
    """

    def __init__(self, drop=A2_REDUNDANT, skew_limit=SKEW_LIMIT, rare_share=RARE_SHARE):
        """
        Store the settings of the preprocessing.

        Args:
            drop: Names of the features to remove.
            skew_limit: Features with a skewness above this limit get the logarithm.
            rare_share: Categories with a smaller share of the training data are pooled.
        """
        self.drop = tuple(drop)
        self.skew_limit = skew_limit
        self.rare_share = rare_share

    def fit(self, X: pd.DataFrame) -> "KNNEncoder":
        """
        Find from the training data which features to log-transform, the scaling ranges and the categories to keep.

        Args:
            X: Table of training features.

        Returns:
            The fitted encoder.
        """
        kept = [c for c in X.columns if c not in self.drop]
        self.nominal = [c for c in kept if c in config.NOMINAL_FEATURES]
        self.numeric = [c for c in kept if c not in config.NOMINAL_FEATURES]
        self.log_features = []
        self.low, self.span = {}, {}
        for name in self.numeric:
            x = X[name].to_numpy(float)
            if x.min() >= 0 and skewness(x) > self.skew_limit:
                self.log_features.append(name)
                x = np.log1p(x)
            self.low[name] = x.min()
            self.span[name] = x.max() - x.min()
        self.levels = {}
        for name in self.nominal:
            share = self._categories(X[name]).value_counts(normalize=True)
            self.levels[name] = sorted(share.index[share >= self.rare_share]) + [INFREQUENT]
        self.feature_names = list(self.numeric) + [f"{n}={v}" for n in self.nominal
                                                   for v in self.levels[n]]
        return self

    @staticmethod
    def _categories(column):
        """
        Turn a nominal column into text, with a missing value as its own category.

        Args:
            column: Column of a nominal feature.

        Returns:
            Column of text values.
        """
        return column.fillna(config.MISSING_CATEGORY).astype(str)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply the preprocessing to a table of features.

        A value outside the training range is not clipped, so a scaled value can lie outside 0 to 1.

        Args:
            X: Table with the same columns as the table given to fit.

        Returns:
            Matrix of 32-bit floats with one row per instance and one column per numeric feature or one-hot category.
        """
        parts = []
        for name in self.numeric:
            x = X[name].to_numpy(float)
            if name in self.log_features:
                x = np.log1p(np.maximum(x, 0.0))
            span = self.span[name]
            parts.append(((x - self.low[name]) / span if span > 0 else np.zeros_like(x))[:, None])
        for name in self.nominal:
            cats = self._categories(X[name])
            levels = self.levels[name]
            cats = cats.where(cats.isin(levels), INFREQUENT)
            index = pd.Categorical(cats, categories=levels).codes
            onehot = np.zeros((len(X), len(levels)))
            onehot[np.arange(len(X)), index] = 1.0
            parts.append(onehot)
        return np.ascontiguousarray(np.hstack(parts), dtype=np.float32)


@njit(parallel=True, cache=True)
def _neighbours(Q, X, K, order):
    """
    Find the K nearest training instances of every query with the Manhattan distance, in parallel over the queries.

    The Manhattan distance is the sum of the absolute differences of the features.
    Each query keeps a sorted list of its K best candidates. The sum for a candidate stops as soon as it reaches the K-th smallest distance found so far.
    Features are added in the order of decreasing spread, so the sum stops early. A candidate at an equal distance does not replace an earlier one.

    Args:
        Q: Query matrix with one row per instance.
        X: Training matrix.
        K: Number of neighbours to find.
        order: Order in which the features are added.

    Returns:
        A pair of arrays with one row per query: the indices of the K nearest training instances and their distances, both sorted by distance.
    """
    nq, n, d = Q.shape[0], X.shape[0], X.shape[1]
    out_i = np.empty((nq, K), np.int32)
    out_d = np.empty((nq, K), np.float32)
    for q in prange(nq):
        best_d = np.full(K, np.inf)
        best_i = np.full(K, -1, np.int32)
        worst = np.inf
        for i in range(n):
            acc = 0.0
            for jj in range(d):
                j = order[jj]
                acc += abs(Q[q, j] - X[i, j])
                if acc >= worst:
                    break
            if acc < worst:
                pos = K - 1
                while pos > 0 and best_d[pos - 1] > acc:
                    best_d[pos] = best_d[pos - 1]
                    best_i[pos] = best_i[pos - 1]
                    pos -= 1
                best_d[pos] = acc
                best_i[pos] = i
                worst = best_d[K - 1]
        out_i[q] = best_i
        out_d[q] = best_d.astype(np.float32)
    return out_i, out_d


def nearest_neighbours(X_query, X_train, K=51):
    """
    Find the K nearest training instances of every query instance.

    Args:
        X_query: Query matrix with one row per instance.
        X_train: Training matrix with the same columns.
        K: Number of neighbours to find. One search with the largest K serves every smaller k.

    Returns:
        A pair of arrays with one row per query: the indices of the neighbours and the distances, sorted from nearest to farthest.
    """
    X_query = np.ascontiguousarray(X_query, dtype=np.float32)
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    order = np.argsort(-X_train.std(axis=0)).astype(np.int64)
    return _neighbours(X_query, X_train, int(K), order)


def class_vote_weights(y_train, n_classes, beta):
    """
    Compute the weight of a vote for each class, proportional to n_c^(-beta).

    Here n_c is the number of training instances of class c. With beta = 0 all weights are 1. With beta = 1 every class gets the same total vote.
    The weights are scaled so that the largest weight is 1.

    Args:
        y_train: Class label of every training instance.
        n_classes: Number of classes.
        beta: Exponent of the weighting.

    Returns:
        Array with one weight per class.
    """
    n_c = np.bincount(y_train, minlength=n_classes).astype(float)
    w = np.zeros(n_classes)
    w[n_c > 0] = n_c[n_c > 0] ** (-beta)
    return w / w.max()


def vote(neighbour_labels, neighbour_dist, k, n_classes, weighting="uniform", class_weight=None):
    """
    Predict the class of every query from its k nearest neighbours.

    With uniform voting, each neighbour has one vote. With distance voting, neighbour i at distance d_i gets the weight (d_k - d_i) / (d_k - d_1) of Dudani (1976), or 1 if d_k equals d_1.
    Class weights multiply the votes for a class. A remaining tie goes to the tied class whose neighbours are closest.

    Args:
        neighbour_labels: Labels of the neighbours, one row per query, sorted from nearest to farthest. It needs at least k columns.
        neighbour_dist: Distances of the neighbours in the same layout.
        k: Number of neighbours that vote.
        n_classes: Number of classes.
        weighting: "uniform" or "distance".
        class_weight: Weight of a vote for each class, or None.

    Returns:
        Array with the predicted class of every query.

    Raises:
        ValueError: If weighting is not "uniform" or "distance".
    """
    labels = neighbour_labels[:, :k]
    nq = labels.shape[0]
    if weighting == "uniform":
        w = np.ones((nq, k))
    elif weighting == "distance":
        d = neighbour_dist[:, :k].astype(float)
        d1, dk = d[:, :1], d[:, k - 1:k]
        spread = dk - d1
        w = np.where(spread > 0, (dk - d) / np.where(spread > 0, spread, 1.0), 1.0)
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    # Each pair of query and class gets one cell number, so one bincount adds up the votes of all queries.
    cell = (np.arange(nq)[:, None] * n_classes + labels).ravel()
    size = nq * n_classes
    scores = np.bincount(cell, weights=w.ravel(), minlength=size).reshape(nq, n_classes)
    closeness = np.bincount(cell, weights=np.tile(1.0 / np.arange(1, k + 1), nq),
                            minlength=size).reshape(nq, n_classes)
    if class_weight is not None:
        scores *= class_weight
    # The tiny closeness term only breaks exact ties. A neighbour at rank r adds 1 / r.
    return (scores + 1e-9 * closeness).argmax(axis=1)
