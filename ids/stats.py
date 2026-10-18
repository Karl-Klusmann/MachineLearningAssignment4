"""Significance tests for comparing two classifiers on the same test set.
The file has McNemar's test, a paired stratified bootstrap of the macro-F1 difference and Holm's correction for several tests. Macro-F1 is the mean over the classes of the F1 score, which combines precision and recall.
The empirical process section of the report describes the tests, and experiment E6 applies them to the final comparison.
The file tests/test_stage4.py checks each function.
"""

import math

import numpy as np

from . import config
from .metrics import macro_f1


def mcnemar(y_true, pred_a, pred_b):
    """
    Run McNemar's test on the accuracy of two classifiers.

    The test looks only at instances where exactly one classifier is correct.
    If both classifiers have the same error rate, the two kinds of disagreement are equally likely.
    The statistic (|b - c| - 1)^2 / (b + c) then follows a chi-squared distribution with one degree of freedom. The 1 is a continuity correction.

    Args:
        y_true: True class of every instance.
        pred_a: Predictions of the first classifier.
        pred_b: Predictions of the second classifier.

    Returns:
        Dictionary with b (only the first is correct), c (only the second is correct), chi2 and the p-value p.
    """
    a_right = np.asarray(pred_a) == y_true
    b_right = np.asarray(pred_b) == y_true
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(chi2), "p": float(math.erfc(math.sqrt(chi2 / 2)))}


def bootstrap_macro_f1_difference(y_true, pred_a, pred_b, rng, n_resamples=2000,
                                  n_classes=config.N_CLASSES):
    """
    Estimate the macro-F1 difference of two classifiers with a paired stratified bootstrap.

    A bootstrap resample draws instances at random with replacement. Here the draw happens inside each class, so every resample keeps the class sizes.
    Both classifiers are scored on the same resample. The 2.5th and 97.5th percentiles of the differences give a 95% confidence interval.
    The two-sided p-value is twice the share of resampled differences on the far side of zero. It is never below 1 divided by the number of resamples.

    Args:
        y_true: True class of every instance.
        pred_a: Predictions of the first classifier.
        pred_b: Predictions of the second classifier.
        rng: NumPy random generator for the resampling.
        n_resamples: Number of bootstrap resamples.
        n_classes: Number of classes.

    Returns:
        Dictionary with the observed difference (first minus second), the interval limits ci_low and ci_high, and the p-value p.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    members = [np.flatnonzero(y_true == c) for c in range(n_classes)]
    members = [m for m in members if len(m) > 0]
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = np.concatenate([m[rng.integers(0, len(m), len(m))] for m in members])
        diffs[r] = (macro_f1(y_true[idx], pred_a[idx], n_classes)
                    - macro_f1(y_true[idx], pred_b[idx], n_classes))
    observed = macro_f1(y_true, pred_a, n_classes) - macro_f1(y_true, pred_b, n_classes)
    far_side = np.mean(diffs <= 0) if observed > 0 else np.mean(diffs >= 0)
    p = min(1.0, max(2 * far_side, 1.0 / n_resamples))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(observed), "ci_low": float(low), "ci_high": float(high), "p": float(p)}


def holm(p_values):
    """
    Correct a family of p-values with Holm's step-down procedure.

    The i-th smallest p-value is multiplied by (m - i + 1), where m is the number of tests. The adjusted values never decrease and never exceed 1.

    Args:
        p_values: The p-values of all tests in the family.

    Returns:
        List of adjusted p-values in the original order.
    """
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()
