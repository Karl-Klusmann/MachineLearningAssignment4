"""Experiment E4: how to handle the class imbalance of 501 to 1 inside the forest.
Starting from the E3 winner, only the treatment changes: no correction, class weights with beta 0.5 and 1, a balanced bootstrap, and capped samples of 2,000 or 8,000 instances per class with and without weights.
A balanced bootstrap draws the same number of instances from every class. A capped sample limits the number of draws per class.
The out-of-bag (OOB) precision, recall and F1 of every class show what each treatment gains and costs. Macro-F1 is the mean over the classes of the F1 score. It is the selection measure.
A pre-declared follow-up re-checks the neighbouring values of m with the winning treatment. The final forest configuration is saved in the result file.
The results section on class imbalance of the report and Table IV use the result.
Run: python -m scripts.e4_imbalance. It needs results/e2_features.json and results/e3_tree_size.json. The script took about 25 minutes on 8 cores and writes results/e4_imbalance.json.
"""

import numpy as np

from ids import config
from ids.experiments import (M_GRID, Setup, by_leaves, by_m, development_data, mean_over_runs,
                             run_forest, select)

SHORT = ("Nor", "Rec", "Bac", "DoS", "Exp", "Ana", "Fuz", "Wor", "She", "Gen")


def treatments(base):
    """
    List the imbalance treatments with the forest settings of each.

    The sampling treatments run with minimum leaf size 1 and, if E3 chose a larger leaf size, also with that one. Their samples are much smaller than a full bootstrap sample, and the minimum leaf size counts drawn instances.

    Args:
        base: Settings of the E3 winner, as a dictionary for RandomForest.

    Returns:
        List of pairs (label, settings).
    """
    boot = {**base, "sampling": "bootstrap"}
    out = [
        ("no correction", {**boot, "class_beta": 0.0}),
        ("weights beta = 0.5", {**boot, "class_beta": 0.5}),
        ("weights beta = 1", {**boot, "class_beta": 1.0}),
    ]
    sampling = [
        ("balanced bootstrap", {"sampling": "balanced", "sample_size": None, "class_beta": 0.0}),
        ("capped 2,000", {"sampling": "capped", "sample_size": 2000, "class_beta": 0.0}),
        ("capped 2,000 + weights", {"sampling": "capped", "sample_size": 2000, "class_beta": 1.0}),
        ("capped 8,000", {"sampling": "capped", "sample_size": 8000, "class_beta": 0.0}),
        ("capped 8,000 + weights", {"sampling": "capped", "sample_size": 8000, "class_beta": 1.0}),
    ]
    leaf_sizes = sorted({1, base["min_samples_leaf"]})
    for label, change in sampling:
        for leaf in leaf_sizes:
            suffix = f", leaf {leaf}" if len(leaf_sizes) > 1 else ""
            out.append((label + suffix, {**base, **change, "min_samples_leaf": leaf}))
    return out


def print_tables(results, best):
    """
    Print the OOB macro-F1, balanced accuracy and accuracy of every treatment, and its recall, precision and F1 for each class.

    Args:
        results: List of result dictionaries from run_forest.
        best: Index of the selected treatment.
    """
    print(f"\n{'treatment':<32}{'macro-F1':>16}{'bal. acc.':>10}{'accuracy':>10}{'leaves':>8}{'fit s':>7}")
    for i, r in enumerate(results):
        mark = "  <- selected" if i == best else ""
        print(f"{r['label']:<32}{r['macro_f1_mean']:>8.4f} ± {r['macro_f1_sd']:.4f}"
              f"{mean_over_runs(r, ('oob', 'balanced_accuracy')):>10.4f}"
              f"{mean_over_runs(r, ('oob', 'accuracy')):>10.4f}{r['leaves_mean']:>8,.0f}"
              f"{r['seconds_mean']:>7.0f}{mark}")
    for measure in ("recall", "precision", "f1"):
        print(f"\nOOB {measure} per class")
        print(f"{'treatment':<32}" + "".join(f"{c:>6}" for c in SHORT))
        for r in results:
            values = mean_over_runs(r, ("oob", measure))
            print(f"{r['label']:<32}" + "".join(f"{v:>6.2f}" for v in values))


def main():
    """Run every treatment, select the winner, re-check m and write the results."""
    setup = Setup(__doc__.splitlines()[0])
    m = setup.load("e2_features")["best_m"]
    base = setup.load("e3_tree_size")["best_params"]
    codes, y, encoder = development_data()
    print(f"base: {base}")
    print("class sizes: " + ", ".join(f"{c} {n:,}" for c, n in zip(SHORT, np.bincount(y))))

    results = [run_forest(codes, y, encoder, params, setup, label)
               for label, params in treatments(base)]
    best = select(results, cost=by_leaves)
    print_tables(results, best)

    winner = results[best]
    i = M_GRID.index(m)
    neighbours = [M_GRID[j] for j in (i - 1, i + 1) if 0 <= j < len(M_GRID)]
    print(f"\nre-checking m = {neighbours} with '{winner['label']}' (m = {m} already run)")
    recheck = [winner] + [run_forest(codes, y, encoder, {**winner["params"], "max_features": k},
                                     setup, f"{winner['label']}, m = {k}")
                          for k in neighbours]
    final = recheck[select(recheck, cost=by_m)]
    print(f"\nfinal forest configuration: {final['label']}  {final['params']}")

    setup.save("e4_imbalance", {"base": base, "results": results, "best_index": best,
                                "recheck": recheck, "final_params": final["params"],
                                "final_label": final["label"], "class_names": list(config.CLASS_NAMES)})


if __name__ == "__main__":
    main()
