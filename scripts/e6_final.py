"""Experiment E6: the final evaluation on the generalisation set. Run it once.
This is the only script that uses the generalisation set. Every model is trained on the whole development set with settings fixed by the earlier experiments and predicts the generalisation set once.
The models are the final forest, the fully grown and pruned trees of the project, the tree of Assignment 2 rebuilt with scikit-learn, and the k-nearest-neighbour (k-NN) classifier with the old and the revised settings.
The script runs the paired tests, splits the errors of the forest by class and by ambiguity, and saves the predictions.
The results sections on the comparison and on the remaining errors of the report, Tables VI to VIII and Fig. 4 use the result.
Run: python -m scripts.e6_final. It needs results/tune_tree.json, results/e4_imbalance.json and results/tune_knn.json. The script took about 16 minutes on 8 cores and writes results/e6_final.json and results/e6_predictions.npz.
"""

import time

import numpy as np

from ids import config
from ids.data import load_dataset
from ids.encoding import TreeEncoder
from ids.experiments import Setup
from ids.forest import RandomForest
from ids.knn import KNNEncoder, class_vote_weights, nearest_neighbours, vote
from ids.metrics import macro_f1, summarise
from ids.splits import development_generalisation_split, stratified_holdout
from ids.stats import bootstrap_macro_f1_difference, holm, mcnemar
from ids.tree import ClassificationTree, class_weights

try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    raise SystemExit("scikit-learn is needed to rebuild the A2 tree: pip install scikit-learn")

N_SEEDS = 10
N_RESAMPLES = 2000
A2_TREE = {"criterion": "entropy", "class_weight": "balanced", "ccp_alpha": 1e-4, "min_samples_leaf": 5}
A2_KNN = {"k": 7, "weighting": "uniform", "beta": 0.0}
SHORT = ("Nor", "Rec", "Bac", "DoS", "Exp", "Ana", "Fuz", "Wor", "She", "Gen")
COMPARISONS = (
    ("final forest", "tree, pruned (own)"),
    ("final forest", "tree, fully grown"),
    ("final forest", "A2 tree (scikit-learn)"),
    ("final forest", "k-NN, revised"),
    ("final forest", "k-NN, A2"),
    ("tree, pruned (own)", "A2 tree (scikit-learn)"),
    ("k-NN, revised", "k-NN, A2"),
)


def forest_voting_rule(e4, final_params):
    """
    Apply the voting rule that was fixed before the final run.

    Hard voting is used only if the out-of-bag (OOB) macro-F1 of hard voting beats that of soft voting by more than one seed standard deviation. Macro-F1 is the mean over the classes of the F1 score.

    Args:
        e4: Result dictionary of experiment E4.
        final_params: Settings of the final forest.

    Returns:
        A pair: "hard" or "soft", and a dictionary with the mean scores, the standard deviation and the rule.
    """
    entries = [r for r in e4["recheck"] + e4["results"] if r["params"] == final_params]
    runs = entries[0]["runs"]
    soft = np.array([run["oob"]["macro_f1"] for run in runs])
    hard = np.array([run["oob_hard_macro_f1"] for run in runs])
    sd = soft.std(ddof=1) if len(soft) > 1 else 0.0
    rule = "hard" if hard.mean() - soft.mean() > sd else "soft"
    return rule, {"oob_soft_mean": float(soft.mean()), "oob_hard_mean": float(hard.mean()),
                  "oob_soft_sd": float(sd), "rule": rule}


def ordinal_codes(X_fit, X):
    """
    Encode the features as the tree of Assignment 2 saw them.

    Numeric features stay unchanged. Nominal features become numbers in the order of the sorted category names. A missing or unseen category becomes NaN.

    Args:
        X_fit: Table of training features. The categories come from this table.
        X: Table to encode.

    Returns:
        Matrix of floats with one row per instance and one column per feature.
    """
    out = X.copy()
    for name in config.NOMINAL_FEATURES:
        categories = sorted(X_fit[name].dropna().astype(str).unique())
        codes = {c: float(i) for i, c in enumerate(categories)}
        out[name] = X[name].map(lambda v: codes.get(str(v), np.nan) if v == v else np.nan)
    return out.to_numpy(float)


def main():
    """
    Train every model, score it once on the generalisation set, run the tests and the error analysis, and write the results.

    With --quick the script trains on 80% of the development set and tests on the other 20%. It never touches the generalisation set.
    """
    setup = Setup(__doc__.splitlines()[0], full_trees=500)
    n_seeds = 2 if setup.quick else N_SEEDS
    alpha = setup.load("tune_tree")["best_alpha"]
    e4 = setup.load("e4_imbalance")
    final = e4["final_params"]
    knn_params = setup.load("tune_knn")["best_params"]
    rule, rule_info = forest_voting_rule(e4, final)
    print(f"final forest {final}, T = {setup.n_trees}; voting: {rule} "
          f"(OOB soft {rule_info['oob_soft_mean']:.4f} ± {rule_info['oob_soft_sd']:.4f}, "
          f"hard {rule_info['oob_hard_mean']:.4f})")

    data = load_dataset()
    dev, gen = development_generalisation_split(data.y)
    if setup.quick:
        part_train, part_test = stratified_holdout(data.y[dev], 0.2, config.rng("quick_holdout"))
        dev, gen = dev[part_train], dev[part_test]
        print("quick mode: testing on 20% of the development set, not the generalisation set")
    X_dev, X_gen, y_dev, y_gen = data.X.iloc[dev], data.X.iloc[gen], data.y[dev], data.y[gen]
    encoder = TreeEncoder().fit(X_dev)
    c_dev, c_gen = encoder.transform(X_dev), encoder.transform(X_gen)
    # preds maps a model name to a list of predictions, one per seed. Seed 0 is the model that the paired tests use.
    preds = {}
    start = time.perf_counter()

    preds["final forest, soft voting"], preds["final forest, hard voting"] = [], []
    for s in range(n_seeds):
        rf = RandomForest(n_trees=setup.n_trees, seed=s, **final).fit(
            c_dev, y_dev, encoder.n_codes, encoder.is_nominal, config.N_CLASSES,
            code_values=encoder.code_values(), eval_sets={"gen": (c_gen, None)},
            keep_trees=False, checkpoints=())
        preds["final forest, soft voting"].append(rf.eval_predict("gen", "soft"))
        preds["final forest, hard voting"].append(rf.eval_predict("gen", "hard"))
        print(f"forest seed {s + 1}/{n_seeds} done ({time.perf_counter() - start:.0f} s)", flush=True)
    # The voting rule that was fixed in advance chooses between the two sets of predictions.
    preds["final forest"] = preds[f"final forest, {rule} voting"]

    w = class_weights(y_dev, config.N_CLASSES, beta=1.0)
    raw_dev, raw_gen = ordinal_codes(X_dev, X_dev), ordinal_codes(X_dev, X_gen)
    for name in ("tree, fully grown", "tree, pruned (own)", "A2 tree (scikit-learn)"):
        preds[name] = []
    for s in range(n_seeds):
        tree = ClassificationTree(seed=s).fit(c_dev, y_dev, encoder.n_codes, encoder.is_nominal,
                                              config.N_CLASSES, class_weight=w,
                                              code_values=encoder.code_values())
        preds["tree, fully grown"].append(tree.predict(c_gen))
        preds["tree, pruned (own)"].append(tree.prune(alpha).predict(c_gen))
        a2 = DecisionTreeClassifier(random_state=s, **A2_TREE).fit(raw_dev, y_dev)
        preds["A2 tree (scikit-learn)"].append(a2.predict(raw_gen))
    a2_leaves = int(a2.get_n_leaves())
    print(f"trees done ({time.perf_counter() - start:.0f} s)", flush=True)

    knn_encoder = KNNEncoder().fit(X_dev)
    k_max = max(A2_KNN["k"], knn_params["k"])
    # One neighbour search serves both k-NN configurations.
    idx, dist = nearest_neighbours(knn_encoder.transform(X_gen), knn_encoder.transform(X_dev), k_max)
    labels = y_dev[idx]
    for name, p in (("k-NN, A2", A2_KNN), ("k-NN, revised", knn_params)):
        cw = class_vote_weights(y_dev, config.N_CLASSES, p["beta"]) if p["beta"] > 0 else None
        preds[name] = [vote(labels, dist, p["k"], config.N_CLASSES, p["weighting"], cw)]
    print(f"k-NN done ({time.perf_counter() - start:.0f} s)", flush=True)

    order = ["final forest", "final forest, soft voting", "final forest, hard voting",
             "tree, fully grown", "tree, pruned (own)", "A2 tree (scikit-learn)",
             "k-NN, A2", "k-NN, revised"]
    scores = {}
    for name in order:
        per_seed = [macro_f1(y_gen, p) for p in preds[name]]
        scores[name] = {"summary": summarise(y_gen, preds[name][0]), "macro_f1_per_seed": per_seed}

    print(f"\n{'model':<28}{'macro-F1':>9}{'10 seeds':>17}{'accuracy':>10}{'bal. acc.':>10}"
          f"{'weighted F1':>12}{'detection':>10}{'false alarm':>12}")
    for name in order:
        s, ps = scores[name]["summary"], scores[name]["macro_f1_per_seed"]
        spread = f"{np.mean(ps):.4f} ± {np.std(ps, ddof=1):.4f}" if len(ps) > 1 else "-"
        print(f"{name:<28}{s['macro_f1']:>9.4f}{spread:>17}{s['accuracy']:>10.4f}"
              f"{s['balanced_accuracy']:>10.4f}{s['weighted_f1']:>12.4f}{s['detection_rate']:>10.4f}"
              f"{s['false_alarm_rate']:>12.4f}")

    print("\nF1 per class (seed 0)")
    models = {"final forest": "forest", "tree, pruned (own)": "pruned tree",
              "A2 tree (scikit-learn)": "A2 tree", "k-NN, revised": "k-NN revised", "k-NN, A2": "k-NN A2"}
    print(f"{'class':<16}{'support':>8}" + "".join(f"{label:>14}" for label in models.values()))
    support = np.bincount(y_gen, minlength=config.N_CLASSES)
    for c, cname in enumerate(config.CLASS_NAMES):
        print(f"{cname:<16}{support[c]:>8,}" + "".join(f"{scores[m]['summary']['f1'][c]:>14.3f}" for m in models))

    tests = []
    for i, (a, b) in enumerate(COMPARISONS):
        pa, pb = preds[a][0], preds[b][0]
        tests.append({"a": a, "b": b, "mcnemar": mcnemar(y_gen, pa, pb),
                      "bootstrap": bootstrap_macro_f1_difference(y_gen, pa, pb, config.rng("bootstrap", i),
                                                                 100 if setup.quick else N_RESAMPLES)})
    for key, adj in (("mcnemar", holm([t["mcnemar"]["p"] for t in tests])),
                     ("bootstrap", holm([t["bootstrap"]["p"] for t in tests]))):
        for t, p in zip(tests, adj):
            t[key]["p_holm"] = p
    print(f"\n{'comparison (A vs B)':<52}{'macro-F1 A - B [95% CI]':>30}{'p (Holm)':>10}"
          f"{'b':>7}{'c':>7}{'McNemar p (Holm)':>18}")
    for t in tests:
        bs, mc = t["bootstrap"], t["mcnemar"]
        print(f"{t['a'] + ' vs ' + t['b']:<52}{bs['difference']:>+12.4f} [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]"
              f"{bs['p_holm']:>10.4f}{mc['b']:>7}{mc['c']:>7}{mc['p_holm']:>18.2e}")

    p = preds["final forest"][0]
    wrong = p != y_gen
    amb = data.ambiguous[gen]
    attack_true, attack_pred = y_gen != 0, p != 0
    ceiling = data.group_majority[gen] == y_gen
    rq4 = {
        "error_rate_ambiguous": float(wrong[amb].mean()),
        "error_rate_unambiguous": float(wrong[~amb].mean()),
        "share_ambiguous": float(amb.mean()),
        "share_of_errors_on_ambiguous": float(amb[wrong].mean()),
        "errors_attack_as_other_attack": int(np.sum(wrong & attack_true & attack_pred)),
        "errors_missed_attack": int(np.sum(wrong & attack_true & ~attack_pred)),
        "errors_false_alarm": int(np.sum(wrong & ~attack_true)),
        "majority_label_accuracy": float(ceiling.mean()),
        "majority_label_recall": [float(ceiling[y_gen == c].mean()) for c in range(config.N_CLASSES)],
    }
    print(f"\nRQ4 (final forest, seed 0): error rate {wrong.mean():.4f}")
    print(f"  ambiguous instances: {rq4['share_ambiguous']:.1%} of the set, error rate "
          f"{rq4['error_rate_ambiguous']:.3f}; unambiguous: {rq4['error_rate_unambiguous']:.3f}; "
          f"{rq4['share_of_errors_on_ambiguous']:.1%} of all errors fall on ambiguous instances")
    n_wrong = wrong.sum()
    print(f"  errors: attack as another attack {rq4['errors_attack_as_other_attack']:,} "
          f"({rq4['errors_attack_as_other_attack'] / n_wrong:.1%}), missed attacks "
          f"{rq4['errors_missed_attack']:,} ({rq4['errors_missed_attack'] / n_wrong:.1%}), false alarms "
          f"{rq4['errors_false_alarm']:,} ({rq4['errors_false_alarm'] / n_wrong:.1%})")
    print(f"  accuracy if every identical-feature group got its majority label: "
          f"{rq4['majority_label_accuracy']:.4f}")
    recall = scores["final forest"]["summary"]["recall"]
    print(f"{'  recall vs ceiling':<20}" + "".join(f"{c:>8}" for c in SHORT))
    print(f"{'  forest':<20}" + "".join(f"{r:>8.2f}" for r in recall))
    print(f"{'  majority label':<20}" + "".join(f"{r:>8.2f}" for r in rq4["majority_label_recall"]))
    cm = np.array(scores["final forest"]["summary"]["confusion_matrix"])
    print("\nconfusion matrix of the final forest (rows: true class, columns: predicted)")
    print(f"{'':>6}" + "".join(f"{c:>7}" for c in SHORT))
    for c, row in zip(SHORT, cm):
        print(f"{c:>6}" + "".join(f"{v:>7,}" for v in row))

    setup.save("e6_final", {"final_params": final, "voting": rule_info, "alpha": alpha,
                            "a2_tree": A2_TREE, "a2_tree_leaves_seed9": a2_leaves,
                            "knn_a2": A2_KNN, "knn_revised": knn_params, "n_seeds": n_seeds,
                            "n_trees": setup.n_trees, "scores": scores, "tests": tests, "rq4": rq4})
    np.savez_compressed(setup.results_dir / "e6_predictions.npz", y=y_gen, ambiguous=amb,
                        **{name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""):
                           preds[name][0] for name in order})


if __name__ == "__main__":
    main()
