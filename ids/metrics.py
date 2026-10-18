"""Performance measures computed from the confusion matrix.
The main measure of the report is macro-F1, the mean over the classes of the F1 score, which combines precision and recall.
The file also gives accuracy, balanced accuracy and a view that merges the nine attack classes into one.
It is used in every experiment and in the final comparison, and the tests check it against scikit-learn.
"""

import numpy as np

from . import config


def confusion_matrix(y_true, y_pred, n_classes=config.N_CLASSES):
    """
    Count how often each true class is predicted as each class.

    Args:
        y_true: True class of every instance.
        y_pred: Predicted class of every instance.
        n_classes: Number of classes.

    Returns:
        Array of shape (n_classes, n_classes). Entry [i, j] counts instances of class i that were predicted as class j.
    """
    return np.bincount(np.asarray(y_true) * n_classes + np.asarray(y_pred),
                       minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _ratio(numerator, denominator):
    """
    Divide two arrays and return 0 where the denominator is 0.

    Args:
        numerator: Array of numerators.
        denominator: Array of denominators with the same shape.

    Returns:
        Array of quotients as floats.
    """
    numerator = np.asarray(numerator, float)
    denominator = np.asarray(denominator, float)
    out = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def per_class_scores(cm):
    """
    Compute precision, recall and F1 for every class.

    Precision is the share of predictions of a class that are correct. Recall is the share of instances of a class that are found.
    The F1 score is the harmonic mean of the two. A class that is never predicted gets precision 0.

    Args:
        cm: Confusion matrix with true classes in the rows and predicted classes in the columns.

    Returns:
        Three arrays with one entry per class: precision, recall and F1.
    """
    tp = np.diag(cm)
    precision = _ratio(tp, cm.sum(axis=0))
    recall = _ratio(tp, cm.sum(axis=1))
    f1 = _ratio(2 * precision * recall, precision + recall)
    return precision, recall, f1


def macro_f1(y_true, y_pred, n_classes=config.N_CLASSES):
    """
    Compute the macro-F1 score, the mean of the F1 scores of all classes.

    Every class counts equally, whatever its size.

    Args:
        y_true: True class of every instance.
        y_pred: Predicted class of every instance.
        n_classes: Number of classes.

    Returns:
        The macro-F1 score, a number between 0 and 1.
    """
    return float(per_class_scores(confusion_matrix(y_true, y_pred, n_classes))[2].mean())


def summarise(y_true, y_pred, n_classes=config.N_CLASSES, normal_class=0):
    """
    Compute every measure used in the report for one set of predictions.

    The binary view merges all attack classes. The detection rate is the share of attacks that are predicted as any attack class.
    The false-alarm rate is the share of normal instances that are predicted as an attack.

    Args:
        y_true: True class of every instance.
        y_pred: Predicted class of every instance.
        n_classes: Number of classes.
        normal_class: Index of the class for normal traffic.

    Returns:
        Dictionary with accuracy, macro_f1, weighted_f1, balanced_accuracy, detection_rate, false_alarm_rate, the per-class lists precision, recall and f1, and the confusion_matrix as nested lists. The weighted F1 weights each class by its size.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precision, recall, f1 = per_class_scores(cm)
    support = cm.sum(axis=1)

    attack = np.arange(n_classes) != normal_class
    attacks_detected = cm[np.ix_(attack, attack)].sum()
    false_alarms = cm[normal_class, attack].sum()

    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.sum(f1 * support) / support.sum()),
        "balanced_accuracy": float(recall.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "detection_rate": float(attacks_detected / cm[attack].sum()),
        "false_alarm_rate": float(false_alarms / cm[normal_class].sum()),
        "confusion_matrix": cm.tolist(),
    }
