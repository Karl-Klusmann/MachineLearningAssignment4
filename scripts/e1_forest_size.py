"""Experiment E1: how the forest improves as trees are added, and whether the voting rule matters.
The base forest (m = 6 features per node, bootstrap samples, fully grown class-weighted trees) grows to 1,000 trees with each of three seeds.
The out-of-bag (OOB) macro-F1 is recorded with soft and with hard voting at every checkpoint. Macro-F1 is the mean over the classes of the F1 score, which combines precision and recall.
The curve shows whether 500 trees, the size of the later experiments, lie on the plateau.
The results section on forest size of the report and Fig. 1 use the result.
Run: python -m scripts.e1_forest_size. The script took about 4 minutes and writes results/e1_forest_size.json.
"""

import numpy as np

from ids.experiments import BASE, Setup, development_data, run_forest

TOLERANCE = 0.002


def main():
    """Grow the base forest, print the soft and hard curves and write the results."""
    setup = Setup(__doc__.splitlines()[0], full_trees=1000)
    codes, y, encoder = development_data()
    print(f"growing {setup.n_trees} trees with {len(setup.seeds)} seed(s)")
    result = run_forest(codes, y, encoder, BASE, setup, "base forest")

    curves = [r["oob_curve"] for r in result["runs"]]
    trees = curves[0]["trees"]
    soft = np.array([c["soft"] for c in curves])
    hard = np.array([c["hard"] for c in curves])
    sd = lambda a: a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1])
    print(f"\n{'T':>5}{'soft (mean ± sd)':>20}{'hard - soft (mean ± sd)':>27}")
    for i, T in enumerate(trees):
        d = hard[:, i] - soft[:, i]
        print(f"{T:>5}{soft[:, i].mean():>12.4f} ± {sd(soft)[i]:.4f}"
              f"{d.mean():>+18.4f} ± {d.std(ddof=1) if len(d) > 1 else 0:.4f}")

    final = soft[:, -1].mean()
    # The smallest checkpoint whose mean score is within TOLERANCE of the score at the largest forest.
    enough = next(T for i, T in enumerate(trees) if soft[:, i].mean() >= final - TOLERANCE)
    print(f"\nsmallest recorded T within {TOLERANCE} of the T = {trees[-1]} score: {enough}")
    if 500 in trees:
        i = trees.index(500)
        print(f"T = 500 vs T = {trees[-1]}: {soft[:, i].mean() - final:+.4f}")

    setup.save("e1_forest_size", {"config": result, "tolerance": TOLERANCE,
                                  "trees_within_tolerance": enough})


if __name__ == "__main__":
    main()
