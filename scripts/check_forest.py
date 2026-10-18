"""Check the forest against scikit-learn, time it and compare out-of-bag (OOB) and held-out scores.
Part 1 times the growth of one tree with one thread and with all cores. Part 2 uses stratified five-fold cross-validation (CV) on the development set with forests of 100 trees.
The forests are scikit-learn, the forest of this project with the scikit-learn feature rule, the same forest that counts only varying features, and the same forest with one-against-the-rest nominal splits.
Each step changes one design choice, so the paired difference per fold measures its effect. The OOB macro-F1 on the training folds is compared with the macro-F1 on the held-out fold. Macro-F1 is the mean over the classes of the F1 score, which combines precision and recall.
The verification part of the implementation section of the report uses these results.
Run: python -m scripts.check_forest. The script writes results/check_forest.json.
"""

import json
import os
import time

import numpy as np

from ids import config
from ids.data import load_dataset
from ids.encoding import TreeEncoder
from ids.forest import RandomForest
from ids.metrics import macro_f1
from ids.splits import development_generalisation_split, stratified_kfold

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    raise SystemExit("scikit-learn is required for this check: pip install scikit-learn")

N_TREES = 100
SPEED_TREES = 16


def speed(codes, y, encoder):
    """
    Measure the growing time per tree on the whole development set.

    The settings are m = 6 with one thread, m = 6 with all cores and m = 42 (bagging, the slowest setting) with all cores.

    Args:
        codes: Matrix of codes of the development set.
        y: Class label of every instance.
        encoder: The fitted TreeEncoder.

    Returns:
        List of dictionaries with m, the number of threads, the seconds per tree and the mean number of leaves.
    """
    rows = []
    for m, n_jobs in ((6, 1), (6, None), (42, None)):
        rf = RandomForest(n_trees=SPEED_TREES, max_features=m, seed=0, n_jobs=n_jobs).fit(
            codes, y, encoder.n_codes, encoder.is_nominal, config.N_CLASSES,
            code_values=encoder.code_values(), keep_trees=False)
        rows.append({"m": m, "threads": n_jobs or os.cpu_count(),
                     "seconds_per_tree": rf.fit_seconds_ / SPEED_TREES,
                     "leaves": float(np.mean(rf.tree_leaves_))})
    return rows


def main():
    """Run the speed measurement and the five-fold comparison, print the tables and write the results to results/check_forest.json."""
    data = load_dataset()
    dev, _ = development_generalisation_split(data.y)
    X, y = data.X.iloc[dev], data.y[dev]
    folds = stratified_kfold(y, 5, config.rng("folds", 0))

    encoder = TreeEncoder().fit(X)
    codes = encoder.transform(X)
    # The first timed forest may include the one-off Numba compile time.
    codes_speed = speed(codes, y, encoder)
    print(f"speed on the development set ({len(y):,} instances, {SPEED_TREES} trees each)")
    for r in codes_speed:
        print(f"  m = {r['m']:>2}, {r['threads']} thread(s): {r['seconds_per_tree']:.2f} s per tree "
              f"({r['leaves']:,.0f} leaves)")

    keys = ("sklearn", "ordered-sk", "ordered", "ovr")
    val = {k: [] for k in keys}
    oob = {k: [] for k in keys}
    seconds = {k: [] for k in keys}
    curves = []
    for f, (tr, va) in enumerate(folds):
        enc = TreeEncoder().fit(X.iloc[tr])
        c_tr, c_va = enc.transform(X.iloc[tr]), enc.transform(X.iloc[va])
        y_tr, y_va = y[tr], y[va]
        # Marking every feature as not nominal makes the tree treat nominal codes as ordered numbers, as scikit-learn does.
        ordered = np.zeros(len(enc.feature_names), bool)

        # The three variants of the forest of this project. Each changes one design choice of the previous one.
        for key, nominal, sk_rule in (("ordered-sk", ordered, True), ("ordered", ordered, False),
                                      ("ovr", enc.is_nominal, False)):
            rf = RandomForest(n_trees=N_TREES, max_features=6, count_constant=sk_rule, seed=f).fit(
                c_tr, y_tr, enc.n_codes, nominal, config.N_CLASSES, code_values=enc.code_values(),
                eval_sets={"validation": (c_va, y_va)}, keep_trees=False)
            val[key].append(rf.eval_curves_["validation"]["soft"][-1])
            oob[key].append(rf.oob_curve_["soft"][-1])
            seconds[key].append(rf.fit_seconds_)
            if key == "ovr":
                curves.append({"trees": rf.oob_curve_["trees"],
                               "oob_soft": rf.oob_curve_["soft"], "oob_hard": rf.oob_curve_["hard"],
                               "validation_soft": rf.eval_curves_["validation"]["soft"],
                               "validation_hard": rf.eval_curves_["validation"]["hard"]})

        t = time.perf_counter()
        sk = RandomForestClassifier(n_estimators=N_TREES, max_features=6, oob_score=True,
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=f)
        sk.fit(c_tr.astype(np.float32), y_tr)
        seconds["sklearn"].append(time.perf_counter() - t)
        val["sklearn"].append(macro_f1(y_va, sk.predict(c_va.astype(np.float32))))
        oob["sklearn"].append(macro_f1(y_tr, sk.oob_decision_function_.argmax(axis=1)))
        print(f"fold {f + 1}/5 done")

    print(f"\nforests of {N_TREES} trees, m = 6; macro-F1 over 5 folds (mean ± sd)")
    print(f"{'forest':<12}{'held-out fold':>18}{'OOB':>18}{'fit s':>8}")
    for k in keys:
        v, o = np.array(val[k]), np.array(oob[k])
        print(f"{k:<12}{v.mean():>11.4f} ± {v.std(ddof=1):.4f}{o.mean():>11.4f} ± {o.std(ddof=1):.4f}"
              f"{np.mean(seconds[k]):>8.1f}")

    diffs = {
        "implementation (sklearn - ordered-sk)": np.subtract(val["sklearn"], val["ordered-sk"]),
        "m varying features (ordered - ordered-sk)": np.subtract(val["ordered"], val["ordered-sk"]),
        "one-vs-rest nominal (ovr - ordered)": np.subtract(val["ovr"], val["ordered"]),
        "OOB - held-out (ovr, T = 100)": np.subtract(oob["ovr"], val["ovr"]),
    }
    tree_file = config.RESULTS_DIR / "check_tree.json"
    if tree_file.exists():
        # Configuration C of check_tree.py is the single fully grown tree used as the reference.
        single = json.loads(tree_file.read_text())["macro_f1_per_fold"]["C"]
        diffs["forest - single tree (ovr - E0 C)"] = np.subtract(val["ovr"], single)
    print(f"\n{'paired difference in macro-F1':<42}{'mean ± sd':>18}   per fold")
    for label, d in diffs.items():
        print(f"{label:<42}{d.mean():>+10.4f} ± {d.std(ddof=1):.4f}   " + " ".join(f"{x:+.4f}" for x in d))

    trees = curves[0]["trees"]
    print("\nOOB vs held-out macro-F1 as the forest grows (own forest, soft voting, mean of 5 folds)")
    print(f"{'T':>5}{'OOB':>9}{'held-out':>10}{'OOB - held-out':>16}{'hard - soft (held-out)':>24}")
    for i, T in enumerate(trees):
        o = np.mean([c["oob_soft"][i] for c in curves])
        v = np.mean([c["validation_soft"][i] for c in curves])
        h = np.mean([c["validation_hard"][i] - c["validation_soft"][i] for c in curves])
        print(f"{T:>5}{o:>9.4f}{v:>10.4f}{o - v:>+16.4f}{h:>+24.4f}")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "check_forest.json").write_text(json.dumps({
        "speed": codes_speed, "n_trees": N_TREES, "held_out_macro_f1": val, "oob_macro_f1": oob,
        "fit_seconds": seconds, "paired_differences": {k: v.tolist() for k, v in diffs.items()},
        "curves_per_fold": curves,
    }, indent=2))
    print("\nwrote results/check_forest.json")


if __name__ == "__main__":
    main()
