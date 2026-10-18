"""Experiment E5: how much of the error of each model is bias and how much is variance.
A fixed stratified 20% of the development set is the test part. Ten training sets are stratified halves of the remaining 80%.
Every model is trained on every training set and predicts the test part, and ids/bias_variance.py splits the errors into bias and variance.
The models are the fully grown tree, the pruned tree, bagging, a forest with m = 6, the final m with minimum leaf size 1, and the final forest with soft and hard voting. The forests have 200 trees.
The results section on bias and variance of the report and Table V use the result.
Run: python -m scripts.e5_bias_variance. It needs results/tune_tree.json and results/e4_imbalance.json. The script took about 10 minutes on 8 cores and writes results/e5_bias_variance.json.
"""

import time

import numpy as np

from ids import config
from ids.bias_variance import decompose
from ids.encoding import TreeEncoder
from ids.experiments import Setup
from ids.data import load_dataset
from ids.forest import RandomForest
from ids.metrics import macro_f1
from ids.splits import development_generalisation_split, stratified_holdout
from ids.tree import ClassificationTree, class_weights

N_TRAINING_SETS = 10
TEST_FRACTION = 0.2
SHORT = ("Nor", "Rec", "Bac", "DoS", "Exp", "Ana", "Fuz", "Wor", "She", "Gen")


def main():
    """Train every model on every training set, decompose the errors, print the tables and write the results."""
    setup = Setup(__doc__.splitlines()[0], full_trees=200)
    n_sets = 3 if setup.quick else N_TRAINING_SETS
    alpha = setup.load("tune_tree")["best_alpha"]
    final = setup.load("e4_imbalance")["final_params"]
    leaf1 = {"max_features": 42, "min_samples_leaf": 1, "sampling": "bootstrap",
             "sample_size": 1.0, "class_beta": 1.0}
    forests = {
        "bagging (m = 42, leaf 1)": leaf1,
        "forest, default (m = 6, leaf 1)": {**leaf1, "max_features": 6},
        f"forest, m = {final['max_features']}, leaf 1": {**leaf1, "max_features": final["max_features"]},
        "final forest": final,
    }
    # With leaf size 1, the final forest and the m-and-leaf-1 forest are the same model, so one entry is dropped.
    if final["min_samples_leaf"] == 1:
        del forests[f"forest, m = {final['max_features']}, leaf 1"]

    data = load_dataset()
    dev, _ = development_generalisation_split(data.y)
    X, y = data.X.iloc[dev], data.y[dev]
    pool, test = stratified_holdout(y, TEST_FRACTION, config.rng("bv_test"))
    encoder = TreeEncoder().fit(X.iloc[pool])
    codes_pool, codes_test = encoder.transform(X.iloc[pool]), encoder.transform(X.iloc[test])
    y_pool, y_test = y[pool], y[test]
    print(f"pool {len(pool):,} instances, test {len(test):,}; {n_sets} training sets of "
          f"about {len(pool) // 2:,}; forests of {setup.n_trees} trees")

    names = ["tree, fully grown", "tree, pruned"] + list(forests) + ["final forest, hard voting"]
    preds = {name: [] for name in names}
    start = time.perf_counter()
    for r in range(n_sets):
        # Training set r is a stratified random half of the pool. The other half is not used in this round.
        _, half = stratified_holdout(y_pool, 0.5, config.rng("bv_train", r))
        Xr, yr = codes_pool[half], y_pool[half]
        w = class_weights(yr, config.N_CLASSES, beta=1.0)
        tree = ClassificationTree(seed=r).fit(Xr, yr, encoder.n_codes, encoder.is_nominal,
                                              config.N_CLASSES, class_weight=w,
                                              code_values=encoder.code_values())
        preds["tree, fully grown"].append(tree.predict(codes_test))
        preds["tree, pruned"].append(tree.prune(alpha).predict(codes_test))
        for name, params in forests.items():
            rf = RandomForest(n_trees=setup.n_trees, seed=r, **params).fit(
                Xr, yr, encoder.n_codes, encoder.is_nominal, config.N_CLASSES,
                code_values=encoder.code_values(), eval_sets={"test": (codes_test, None)},
                keep_trees=False, checkpoints=())
            preds[name].append(rf.eval_predict("test", "soft"))
            if name == "final forest":
                preds["final forest, hard voting"].append(rf.eval_predict("test", "hard"))
        print(f"training set {r + 1}/{n_sets} done ({time.perf_counter() - start:.0f} s)", flush=True)

    results = {}
    for name in names:
        P = np.array(preds[name])
        results[name] = {
            "macro_f1_per_set": [macro_f1(y_test, p) for p in P],
            "all_instances": decompose(P, y_test, config.N_CLASSES),
            "class_balanced": decompose(P, y_test, config.N_CLASSES, balanced=True),
        }

    for part, title in (("all_instances", "every test instance weighted equally (zero-one loss)"),
                        ("class_balanced", "every class weighted equally (1 - balanced accuracy)")):
        print(f"\n{title}")
        print(f"{'model':<34}{'macro-F1':>9}{'loss':>8}{'bias':>8}{'net var':>9}{'unb. var':>10}{'b. var':>8}")
        for name in names:
            r, d = results[name], results[name][part]
            print(f"{name:<34}{np.mean(r['macro_f1_per_set']):>9.4f}{d['loss']:>8.4f}{d['bias']:>8.4f}"
                  f"{d['net_variance']:>9.4f}{d['unbiased_variance']:>10.4f}{d['biased_variance']:>8.4f}")

    print("\nper class: bias / net variance (every test instance of the class weighted equally)")
    print(f"{'model':<34}" + "".join(f"{c:>12}" for c in SHORT))
    for name in ("tree, pruned", "final forest"):
        pc = results[name]["all_instances"]["per_class"]
        cells = [f"{b:.2f}/{u - v:+.2f}" for b, u, v in
                 zip(pc["bias"], pc["unbiased_variance"], pc["biased_variance"])]
        print(f"{name:<34}" + "".join(f"{c:>12}" for c in cells))

    setup.save("e5_bias_variance", {"alpha": alpha, "forests": forests, "n_training_sets": n_sets,
                                    "n_trees": setup.n_trees, "test_size": int(len(test)),
                                    "results": results})


if __name__ == "__main__":
    main()
