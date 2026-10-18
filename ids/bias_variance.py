"""Bias-variance decomposition of the zero-one loss, after Domingos (2000).
A model is trained on several training sets and predicts the same test instances.
The error then splits into bias, which is the error of the main prediction, and variance, which is the disagreement between the models.
Experiment E5 and the results section on bias and variance of the report use this file.
The script e5_bias_variance.py calls decompose, and the tests check it against hand-computed examples.
"""

import numpy as np


def decompose(predictions, y, n_classes, balanced=False):
    """
    Split the zero-one loss into bias and variance terms.

    For one test instance, the main prediction is the most frequent prediction of the models. A tie goes to the lowest class index.
    The bias is 1 if the main prediction is wrong and 0 otherwise. The variance is the share of models that differ from the main prediction.
    The observed label counts as the best possible answer, so label noise appears as bias.
    The loss equals the bias plus the unbiased variance minus the biased variance.
    The unbiased variance averages the variance over instances with a correct main prediction. There, variance only adds errors.
    The biased variance averages the share of correct predictions over instances with a wrong main prediction. There, variance removes errors.

    Args:
        predictions: Array with one row per model and one column per test instance. Each entry is the predicted class.
        y: True class of every test instance.
        n_classes: Number of classes.
        balanced: If True, every class carries the same total weight, and the loss equals one minus the balanced accuracy. If False, every instance carries the same weight.

    Returns:
        Dictionary with the keys loss, bias, unbiased_variance, biased_variance, net_variance and per_class. The entry per_class holds the mean per class of loss, bias, unbiased variance and biased variance. A class without test instances gives None.
    """
    predictions = np.asarray(predictions)
    y = np.asarray(y)
    R, n = predictions.shape
    counts = np.zeros((n, n_classes))
    for c in range(n_classes):
        counts[:, c] = (predictions == c).sum(axis=0)
    main = counts.argmax(axis=1)
    rows = np.arange(n)
    p_correct = counts[rows, y] / R
    loss = 1.0 - p_correct
    bias = (main != y).astype(float)
    variance = 1.0 - counts[rows, main] / R

    if balanced:
        n_c = np.bincount(y, minlength=n_classes)
        weight = 1.0 / (n_classes * n_c[y])
        weight /= weight.sum()
    else:
        weight = np.full(n, 1.0 / n)

    out = {
        "loss": float(np.sum(weight * loss)),
        "bias": float(np.sum(weight * bias)),
        "unbiased_variance": float(np.sum(weight * (1 - bias) * variance)),
        "biased_variance": float(np.sum(weight * bias * p_correct)),
    }
    out["net_variance"] = out["unbiased_variance"] - out["biased_variance"]
    out["per_class"] = {
        key: [float(np.mean(values[y == c])) if np.any(y == c) else None for c in range(n_classes)]
        for key, values in (("loss", loss), ("bias", bias),
                            ("unbiased_variance", (1 - bias) * variance),
                            ("biased_variance", bias * p_correct))
    }
    return out
