"""Tune the pruning strength alpha of the single tree.
A single tree has no out-of-bag instances, so the script uses stratified five-fold cross-validation (CV) on the development set.
In every fold one fully grown, class-weighted tree is grown per seed and pruned with every alpha of the grid, and each pruned tree is scored on the held-out fold.
The choice follows the one-standard-error rule: the best mean wins, and the smallest tree within one standard error of the best is chosen. The standard error is that of the mean over the folds.
The report uses the result in Table II and in experiments E3, E5 and E6.
Run: python -m scripts.tune_tree. The script writes results/tune_tree.json.
"""

import numpy as np

from ids import config
from ids.data import load_dataset
from ids.encoding import TreeEncoder
from ids.experiments import Setup, by_leaves, select
from ids.metrics import macro_f1
from ids.splits import development_generalisation_split, stratified_kfold
from ids.tree import ClassificationTree, class_weights

ALPHAS = [0.0] + [float(a) for a in np.logspace(-6, -2, 25)]


def main():
    """Score every alpha on every fold and seed, select alpha, print the table and write the results."""
    setup = Setup(__doc__.splitlines()[0])
    data = load_dataset()
    dev, _ = development_generalisation_split(data.y)
    X, y = data.X.iloc[dev], data.y[dev]
    folds = stratified_kfold(y, 5, config.rng("folds", 0))
    if setup.quick:
        folds = folds[:1]

    # score[a, s, f] and leaves[a, s, f] belong to alpha a, seed s and fold f.
    score = np.zeros((len(ALPHAS), len(setup.seeds), len(folds)))
    leaves = np.zeros_like(score)
    for f, (tr, va) in enumerate(folds):
        enc = TreeEncoder().fit(X.iloc[tr])
        c_tr, c_va = enc.transform(X.iloc[tr]), enc.transform(X.iloc[va])
        w = class_weights(y[tr], config.N_CLASSES, beta=1.0)
        for s, seed in enumerate(setup.seeds):
            tree = ClassificationTree(seed=seed).fit(c_tr, y[tr], enc.n_codes, enc.is_nominal,
                                                     config.N_CLASSES, class_weight=w,
                                                     code_values=enc.code_values())
            for a, alpha in enumerate(ALPHAS):
                pruned = tree.prune(alpha) if alpha > 0 else tree
                score[a, s, f] = macro_f1(y[va], pruned.predict(c_va))
                leaves[a, s, f] = pruned.n_leaves
        print(f"fold {f + 1}/{len(folds)} done", flush=True)

    # per_seed averages over the folds. per_fold averages over the seeds. The standard error uses the fold scores.
    per_seed = score.mean(axis=2)
    per_fold = score.mean(axis=1)
    k = len(folds)
    results = [{"alpha": alpha,
                "macro_f1_mean": float(per_fold[a].mean()),
                "macro_f1_se": float(per_fold[a].std(ddof=1) / np.sqrt(k)) if k > 1 else 0.0,
                "macro_f1_seed_sd": float(per_seed[a].std(ddof=1)) if len(setup.seeds) > 1 else 0.0,
                "leaves_mean": float(leaves[a].mean()),
                "macro_f1_per_fold": per_fold[a].tolist()}
               for a, alpha in enumerate(ALPHAS)]
    best = select(results, cost=by_leaves, spread="macro_f1_se")

    print(f"\n{'alpha':>10}{'leaves':>9}{'macro-F1':>10}{'SE (folds)':>12}{'sd (seeds)':>12}")
    for i, r in enumerate(results):
        mark = "  <- selected" if i == best else ""
        print(f"{r['alpha']:>10.2e}{r['leaves_mean']:>9,.0f}{r['macro_f1_mean']:>10.4f}"
              f"{r['macro_f1_se']:>12.4f}{r['macro_f1_seed_sd']:>12.4f}{mark}")
    gain = np.subtract(results[best]["macro_f1_per_fold"], results[0]["macro_f1_per_fold"])
    print(f"\npruned (alpha = {ALPHAS[best]:.2e}) - fully grown, per fold: "
          + " ".join(f"{g:+.4f}" for g in gain) + f"   mean {gain.mean():+.4f}")

    setup.save("tune_tree", {"alphas": ALPHAS, "results": results, "best_index": best,
                             "best_alpha": ALPHAS[best], "seeds": list(setup.seeds)})


if __name__ == "__main__":
    main()
