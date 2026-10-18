"""Experiment E3: whether averaging replaces pruning. The script varies the leaf size and the bag size and adds a forest of pruned trees.
With the m selected in E2, the forest grows for every combination of minimum leaf size (1, 5, 25) and bag fraction (0.2, 0.5, 1.0).
One more forest prunes every tree with the alpha that cross-validation chose for the single tree.
The out-of-bag (OOB) macro-F1 is the selection measure, with ties going to the forest with fewer leaves per tree. Macro-F1 is the mean over the classes of the F1 score, which combines precision and recall.
The winner goes to experiment E4. The results section on leaf size and bag size of the report and Table III use the result.
Run: python -m scripts.e3_tree_size. It needs results/e2_features.json and results/tune_tree.json. The script took about 20 minutes on 8 cores and writes results/e3_tree_size.json.
"""

import numpy as np

from ids.experiments import BASE, Setup, by_leaves, development_data, run_forest, select

LEAF_SIZES = (1, 5, 25)
BAG_FRACTIONS = (0.2, 0.5, 1.0)
MATCHED_T = {0.2: 200, 0.5: 300, 1.0: 500}


def matched_score(result):
    """
    Read the OOB macro-F1 at the forest size that gives about the same number of OOB trees per instance for every bag fraction.

    An instance is out-of-bag for about exp(-f) of the trees with bag fraction f. The matched sizes are T = 200 for f = 0.2, T = 300 for f = 0.5 and T = 500 for f = 1. They give about 164, 182 and 184 OOB trees per instance.

    Args:
        result: Result dictionary of one configuration from run_forest.

    Returns:
        The mean score over the seeds, or NaN if a quick run stopped before that forest size.
    """
    T = MATCHED_T.get(result["params"]["sample_size"], 500)
    scores = []
    for run in result["runs"]:
        curve = run["oob_curve"]
        if T not in curve["trees"]:
            return float("nan")
        scores.append(curve["soft"][curve["trees"].index(T)])
    return float(np.mean(scores))


def main():
    """Grow the forest for every leaf size and bag fraction and for the pruned trees, select the winner and write the results."""
    setup = Setup(__doc__.splitlines()[0])
    m = setup.load("e2_features")["best_m"]
    alpha = setup.load("tune_tree")["best_alpha"]
    codes, y, encoder = development_data()
    print(f"m = {m} (from E2); {setup.n_trees} trees x {len(setup.seeds)} seed(s) per configuration")

    results = []
    for leaf in LEAF_SIZES:
        for f in BAG_FRACTIONS:
            params = {**BASE, "max_features": m, "min_samples_leaf": leaf, "sample_size": f}
            results.append(run_forest(codes, y, encoder, params, setup,
                                      f"min_leaf = {leaf:>2}, f = {f}"))
    if alpha > 0:
        params = {**BASE, "max_features": m, "prune_alpha": alpha}
        results.append(run_forest(codes, y, encoder, params, setup,
                                  f"pruned trees, alpha = {alpha:.1e}"))
    best = select(results, cost=by_leaves)

    print(f"\n{'configuration':<36}{'macro-F1 (T=500)':>18}{'matched':>9}{'hard':>8}"
          f"{'leaves':>9}{'fit s':>7}")
    for i, r in enumerate(results):
        hard = np.mean([run["oob_hard_macro_f1"] for run in r["runs"]])
        mark = "  <- selected" if i == best else ""
        print(f"{r['label']:<36}{r['macro_f1_mean']:>10.4f} ± {r['macro_f1_sd']:.4f}"
              f"{matched_score(r):>9.4f}{hard:>8.4f}{r['leaves_mean']:>9,.0f}"
              f"{r['seconds_mean']:>7.0f}{mark}")

    setup.save("e3_tree_size", {"m": m, "alpha": alpha, "results": results, "best_index": best,
                                "best_params": results[best]["params"],
                                "matched_trees": {str(k): v for k, v in MATCHED_T.items()}})


if __name__ == "__main__":
    main()
