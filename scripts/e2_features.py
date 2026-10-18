"""Experiment E2: what random feature selection adds beyond bootstrap sampling.
The base forest grows with m features per node for every m of the grid. The value m = 42 uses all features, so it is bagging, which has no random feature selection.
For every m, the out-of-bag (OOB) votes give the strength of Breiman, the mean correlation between trees and the bound on the error, for all instances and with classes weighted equally.
One extra forest repeats m = 6 under the scikit-learn rule for constant features. It is reported but is not a candidate for selection.
The winning m goes to experiments E3 and E4. The results section on features per node of the report and Fig. 2 use the result.
Run: python -m scripts.e2_features. The script took about 30 minutes on 8 cores and writes results/e2_features.json.
"""

import numpy as np

from ids.experiments import BASE, M_GRID, Setup, by_m, development_data, run_forest, select


def main():
    """Grow a forest for every m, print the table of scores, strength and correlation, and write the results."""
    setup = Setup(__doc__.splitlines()[0])
    codes, y, encoder = development_data()
    print(f"{setup.n_trees} trees x {len(setup.seeds)} seed(s) per configuration")
    results = [run_forest(codes, y, encoder, {**BASE, "max_features": m}, setup,
                          f"m = {m}" + (" (bagging)" if m == 42 else ""), diagnostics=True)
               for m in M_GRID]
    sk_rule = run_forest(codes, y, encoder, {**BASE, "count_constant": True}, setup,
                         "m = 6, scikit-learn's feature rule", diagnostics=True)
    best = select(results, cost=by_m)

    print(f"\n{'configuration':<36}{'macro-F1':>16}{'s':>7}{'rho':>7}{'bound':>7}"
          f"{'s bal':>7}{'rho bal':>8}{'leaves':>8}{'fit s':>7}")
    for i, r in enumerate(results + [sk_rule]):
        d = [run["diagnostics"] for run in r["runs"]]
        mean = lambda part, key: np.mean([np.nan if x[part][key] is None else x[part][key] for x in d])
        mark = "  <- selected" if i == best else ""
        print(f"{r['label']:<36}{r['macro_f1_mean']:>8.4f} ± {r['macro_f1_sd']:.4f}"
              f"{mean('all_instances', 'strength'):>7.3f}{mean('all_instances', 'correlation'):>7.3f}"
              f"{mean('all_instances', 'bound'):>7.2f}{mean('class_balanced', 'strength'):>7.3f}"
              f"{mean('class_balanced', 'correlation'):>8.3f}{r['leaves_mean']:>8,.0f}"
              f"{r['seconds_mean']:>7.0f}{mark}")

    setup.save("e2_features", {"m_grid": list(M_GRID), "results": results,
                               "scikit_learn_rule": sk_rule, "best_index": best,
                               "best_m": M_GRID[best]})


if __name__ == "__main__":
    main()
