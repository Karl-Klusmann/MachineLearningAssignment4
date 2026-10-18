"""Loading, checking and de-duplicating the UNSW-NB15 network traffic data.
The file reads the CSV file, drops the identifier column and removes rows that are exact duplicates, label included.
It also finds instances that share one feature vector but carry different labels.
These instances are called ambiguous. They are used for analysis only and never for training.
The data section of the report and the error analysis in the results section use these numbers.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class Dataset:
    """
    The cleaned data set and the information about identical feature vectors.

    Attributes:
        X: Table with one row per instance and the 42 descriptive features in their original units. A missing nominal value is NaN.
        y: Class label of every instance, an integer from 0 to 9.
        group: Identifier of the group of instances with the same feature vector.
        ambiguous: True if the group of the instance holds more than one label.
        group_majority: Most frequent label in the group of the instance. A tie goes to the lower label.
        n_raw: Number of rows in the file.
        n_duplicates: Number of exact duplicate rows that were removed.
    """

    X: pd.DataFrame
    y: np.ndarray
    group: np.ndarray
    ambiguous: np.ndarray
    group_majority: np.ndarray
    n_raw: int
    n_duplicates: int

    @property
    def numeric_features(self) -> list[str]:
        """Return the names of the numeric features."""
        return [c for c in self.X.columns if c not in config.NOMINAL_FEATURES]

    @property
    def nominal_features(self) -> list[str]:
        """Return the names of the nominal features. A nominal feature holds categories without an order."""
        return [c for c in self.X.columns if c in config.NOMINAL_FEATURES]


def load_raw(path: Path = config.DATA_CSV) -> pd.DataFrame:
    """
    Read the CSV file and check it against the expected schema.

    The marker "?" is read as a missing value. Only nominal features may have missing values.

    Args:
        path: Location of the CSV file.

    Returns:
        The table exactly as read, including the identifier column.

    Raises:
        ValueError: If an expected column is missing, a numeric feature holds text or missing values, or a label lies outside 0 to 9.
    """
    df = pd.read_csv(path, na_values=[config.MISSING_TOKEN], keep_default_na=False)

    for column in (config.ID_COLUMN, config.TARGET, *config.NOMINAL_FEATURES):
        if column not in df.columns:
            raise ValueError(f"expected column '{column}' is missing")

    numeric = [c for c in df.columns
               if c not in (config.ID_COLUMN, config.TARGET, *config.NOMINAL_FEATURES)]
    non_numeric = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"numeric features with non-numeric values: {non_numeric}")
    if df[numeric].isna().any().any():
        raise ValueError("missing values in a numeric feature; only nominal features may be missing")

    labels = df[config.TARGET]
    if labels.isna().any() or not labels.between(0, config.N_CLASSES - 1).all():
        raise ValueError("target contains values outside 0..N_CLASSES-1")
    return df


def load_dataset(path: Path = config.DATA_CSV) -> Dataset:
    """
    Load the data, drop the identifier and remove exact duplicate rows.

    Two rows are duplicates only if they match in every feature and in the label.
    Rows with the same features but different labels stay in the data. They are the ambiguous instances.

    Args:
        path: Location of the CSV file.

    Returns:
        A Dataset with the cleaned features, the labels and the information about identical feature vectors.
    """
    raw = load_raw(path)
    table = raw.drop(columns=[config.ID_COLUMN])
    deduplicated = table.drop_duplicates(keep="first").reset_index(drop=True)

    features = [c for c in deduplicated.columns if c != config.TARGET]
    X = deduplicated[features]
    y = deduplicated[config.TARGET].to_numpy(dtype=np.int64)

    group, ambiguous, majority = identical_feature_groups(X, y)
    return Dataset(X=X, y=y, group=group, ambiguous=ambiguous, group_majority=majority,
                   n_raw=len(raw), n_duplicates=len(table) - len(deduplicated))


def identical_feature_groups(X: pd.DataFrame, y: np.ndarray):
    """
    Group instances that share one feature vector and summarise the labels of each group.

    Args:
        X: Table of descriptive features, one row per instance.
        y: Class label of every instance.

    Returns:
        Three arrays with one entry per instance: the group identifier, whether the group holds more than one label, and the most frequent label of the group. A tie goes to the lower label.
    """
    group = X.groupby(list(X.columns), dropna=False, sort=False).ngroup().to_numpy()
    n_groups = group.max() + 1

    # counts[g, c] is the number of instances of group g with label c.
    counts = np.bincount(group * config.N_CLASSES + y,
                         minlength=n_groups * config.N_CLASSES).reshape(n_groups, config.N_CLASSES)
    labels_per_group = (counts > 0).sum(axis=1)
    majority_per_group = counts.argmax(axis=1)

    return group, labels_per_group[group] > 1, majority_per_group[group]
