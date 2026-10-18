"""Tune the k-nearest-neighbour (k-NN) classifier by five-fold cross-validation (CV): k, distance weighting and class-weighted voting.
One neighbour search per fold finds the 101 nearest training instances of every held-out instance. That one search serves every candidate.
The choice follows the one-standard-error rule: the best mean macro-F1 wins, and among the configurations within one standard error the simplest is chosen. Macro-F1 is the mean over the classes of the F1 score, which combines precision and recall.
The configuration of Assignment 2 (k = 7, uniform votes, no class weights) is scored under the same protocol.
Table II and Fig. 3 of the report use the result.
Run: python -m scripts.tune_knn. The script writes results/tune_knn.json.
"""

import time

import numpy as np

from ids import config
from ids.data import load_dataset
from ids.experiments import Setup, select
from ids.knn import KNNEncoder, class_vote_weights, nearest_neighbours, vote
from ids.metrics import macro_f1
from ids.splits import development_generalisation_split, stratified_kfold

K_MAX = 101
K_GRID = tuple(range(1, 32, 2)) + (41, 51, 61, 71, 81, 91, 101)
WEIGHTINGS = ("uniform", "distance")
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
A2_CONFIG = {"k": 7, "weighting": "uniform", "beta": 0.0}


def simplicity(result):
    """
    Rank a k-NN configuration by simplicity for the tie rule.

    A smaller tuple is simpler: uniform votes before distance votes, then the smaller beta, then the larger k, which gives the smoother decision boundary.

    Args:
        result: Result dictionary of one configuration.

    Returns:
        Tuple to compare.
    """
    p = result["params"]
    return (p["weighting"] != "uniform", p["beta"], -p["k"])


def main():
    """Score every configuration on the five folds, select the best one and write the results."""
    setup = Setup(__doc__.splitlines()[0])
    data = load_dataset()
    dev, _ = development_generalisation_split(data.y)
    X, y = data.X.iloc[dev], data.y[dev]
    folds = stratified_kfold(y, 5, config.rng("folds", 0))
    if setup.quick:
        folds = [(tr, va[:2000]) for tr, va in folds[:2]]

    configs = [{"k": k, "weighting": wt, "beta": b} for wt in WEIGHTINGS for b in BETAS for k in K_GRID]
    scores = np.zeros((len(configs), len(folds)))
    start = time.perf_counter()
    for f, (tr, va) in enumerate(folds):
        encoder = KNNEncoder().fit(X.iloc[tr])
        A, B = encoder.transform(X.iloc[tr]), encoder.transform(X.iloc[va])
        # One search with K_MAX neighbours gives the neighbours for every smaller k.
        idx, dist = nearest_neighbours(B, A, K_MAX)
        labels = y[tr][idx]
        weights = {b: class_vote_weights(y[tr], config.N_CLASSES, b) for b in BETAS}
        for c, p in enumerate(configs):
            pred = vote(labels, dist, p["k"], config.N_CLASSES, p["weighting"],
                        weights[p["beta"]] if p["beta"] > 0 else None)
            scores[c, f] = macro_f1(y[va], pred)
        print(f"fold {f + 1}/{len(folds)} done ({time.perf_counter() - start:.0f} s; "
              f"{len(encoder.feature_names)} dimensions)", flush=True)

    k_folds = len(folds)
    results = [{"params": p, "macro_f1_mean": float(scores[c].mean()),
                "macro_f1_se": float(scores[c].std(ddof=1) / np.sqrt(k_folds)),
                "macro_f1_per_fold": scores[c].tolist()} for c, p in enumerate(configs)]
    best = select(results, cost=simplicity, spread="macro_f1_se")
    a2 = next(i for i, r in enumerate(results) if r["params"] == A2_CONFIG)
    top = int(np.argmax([r["macro_f1_mean"] for r in results]))

    print(f"\nbest k for each voting rule (macro-F1, mean ± SE over folds)")
    print(f"{'weighting':<10}{'beta':>6}{'best k':>8}{'macro-F1':>18}{'at k = 7':>10}")
    for wt in WEIGHTINGS:
        for b in BETAS:
            rows = [r for r in results if r["params"]["weighting"] == wt and r["params"]["beta"] == b]
            r = max(rows, key=lambda r: r["macro_f1_mean"])
            k7 = next(x for x in rows if x["params"]["k"] == 7)
            print(f"{wt:<10}{b:>6.2f}{r['params']['k']:>8}{r['macro_f1_mean']:>10.4f} ± "
                  f"{r['macro_f1_se']:.4f}{k7['macro_f1_mean']:>10.4f}")
    for label, i in (("highest mean", top), ("selected (1-SE rule)", best), ("A2 configuration", a2)):
        r = results[i]
        print(f"{label:<22}{r['params']}  macro-F1 {r['macro_f1_mean']:.4f} ± {r['macro_f1_se']:.4f}")
    gain = np.subtract(results[best]["macro_f1_per_fold"], results[a2]["macro_f1_per_fold"])
    print("selected - A2, per fold: " + " ".join(f"{g:+.4f}" for g in gain))

    setup.save("tune_knn", {"configs": configs, "results": results, "best_index": best,
                            "best_params": results[best]["params"], "a2_index": a2,
                            "k_max": K_MAX})


if __name__ == "__main__":
    main()
