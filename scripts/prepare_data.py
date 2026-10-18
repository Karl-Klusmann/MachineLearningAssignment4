"""Load the data, make the fixed development and generalisation split and summarise both.
The summary holds the counts after de-duplication, the class sizes, the share of ambiguous instances and the accuracy ceiling.
The data section of the report (Table I and the text around it) quotes these numbers.
Run: python -m scripts.prepare_data. The script writes results/data_summary.json and results/split_indices.npz.
"""

import json
import time

import numpy as np

from ids import config
from ids.data import load_dataset
from ids.encoding import TreeEncoder
from ids.splits import development_generalisation_split


def class_counts(y):
    """
    Count the instances of every class.

    Args:
        y: Class label of every instance.

    Returns:
        Array with one count per class.
    """
    return np.bincount(y, minlength=config.N_CLASSES)


def main():
    """Print the data summary and write the summary file and the split indices."""
    start = time.perf_counter()
    data = load_dataset()
    n = len(data.y)
    counts = class_counts(data.y)

    print(f"rows in file            {data.n_raw:>9,}")
    print(f"exact duplicates removed {data.n_duplicates:>8,}")
    print(f"instances kept          {n:>9,}")
    print(f"descriptive features    {data.X.shape[1]:>9}  "
          f"({len(data.numeric_features)} numeric, {len(data.nominal_features)} nominal)")
    print(f"imbalance ratio         {counts.max() / counts.min():>9.1f} : 1")

    # If every group of identical feature vectors received its most frequent label, this share of instances would be correct.
    ceiling = np.mean(data.group_majority == data.y)
    print(f"conflicting-label rows  {data.ambiguous.sum():>9,}  ({data.ambiguous.mean():.2%})")
    print(f"accuracy ceiling        {ceiling:>9.2%}  (each group labelled with its majority class)")

    dev, gen = development_generalisation_split(data.y)
    print(f"\ndevelopment set {len(dev):,}   generalisation set {len(gen):,}\n")

    header = f"{'class':<15}{'total':>8}{'dev':>8}{'gen':>7}{'% ambiguous':>13}{'max recall':>12}"
    print(header)
    print("-" * len(header))
    per_class = []
    for c, name in enumerate(config.CLASS_NAMES):
        members = data.y == c
        row = {
            "class": name,
            "total": int(counts[c]),
            "development": int(np.sum(data.y[dev] == c)),
            "generalisation": int(np.sum(data.y[gen] == c)),
            "share_ambiguous": float(data.ambiguous[members].mean()),
            "max_recall_majority_labelling": float(np.mean(data.group_majority[members] == c)),
        }
        per_class.append(row)
        print(f"{name:<15}{row['total']:>8,}{row['development']:>8,}{row['generalisation']:>7,}"
              f"{row['share_ambiguous']:>12.1%}{row['max_recall_majority_labelling']:>12.1%}")

    encoder = TreeEncoder().fit(data.X.iloc[dev])
    numeric = ~encoder.is_nominal
    distinct = encoder.n_codes[numeric]
    print(f"\nencoder fitted on the development set (lossless)")
    print(f"distinct values per numeric feature: min {distinct.min()}, "
          f"median {int(np.median(distinct))}, max {distinct.max():,}; "
          f"{np.sum(distinct > 255)} of {numeric.sum()} have more than 255")
    print("categories per nominal feature (+1 code for unseen): "
          + ", ".join(f"{f} {k - 1}" for f, k in zip(np.array(encoder.feature_names)[~numeric],
                                                    encoder.n_codes[~numeric])))

    config.RESULTS_DIR.mkdir(exist_ok=True)
    summary = {
        "rows_in_file": data.n_raw,
        "duplicates_removed": data.n_duplicates,
        "instances": n,
        "numeric_features": data.numeric_features,
        "nominal_features": data.nominal_features,
        "imbalance_ratio": float(counts.max() / counts.min()),
        "conflicting_label_instances": int(data.ambiguous.sum()),
        "accuracy_ceiling_majority_labelling": float(ceiling),
        "development_size": int(len(dev)),
        "generalisation_size": int(len(gen)),
        "per_class": per_class,
        "codes_per_feature": dict(zip(encoder.feature_names, map(int, encoder.n_codes))),
    }
    (config.RESULTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(config.RESULTS_DIR / "split_indices.npz", development=dev, generalisation=gen)
    print(f"\nwrote results/data_summary.json and results/split_indices.npz "
          f"in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
