"""Shared machinery of the tuning experiments E1 to E4 on the development set.
Every forest is grown on the whole development set and judged by its out-of-bag (OOB) macro-F1 with soft voting, using the same three seeds.
Macro-F1 is the mean over the classes of the F1 score, which combines precision and recall.
The selection rule was fixed before any experiment: the best mean wins, and configurations within one seed standard deviation count as tied. A tie goes to the simpler configuration.
The scripts tune_tree.py, tune_knn.py and e1 to e6 import this file. The empirical process section of the report describes the rule.
"""

import argparse
import json
import time

import numpy as np

from . import config
from .data import load_dataset
from .encoding import TreeEncoder
from .forest import RandomForest
from .metrics import macro_f1, summarise
from .splits import development_generalisation_split

SEEDS = (0, 1, 2)
N_TREES = 500
M_GRID = (1, 2, 4, 6, 9, 13, 20, 30, 42)

BASE = {"max_features": 6, "min_samples_leaf": 1, "sampling": "bootstrap", "sample_size": 1.0,
        "class_beta": 1.0}


class Setup:
    """
    Command-line options and result files shared by the experiment scripts.

    The option --quick runs a tiny version with one seed and 20 trees. It only checks that the script works, and it writes to results/quick/.
    """

    def __init__(self, description, full_trees=N_TREES):
        """
        Read the command line and prepare the settings and the result folder.

        Args:
            description: Text for the command-line help of the script.
            full_trees: Number of trees of a full run.
        """
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--quick", action="store_true",
                            help="tiny run (1 seed, 20 trees) to check the script; writes to results/quick/")
        args = parser.parse_args()
        self.quick = args.quick
        self.seeds = SEEDS[:1] if self.quick else SEEDS
        self.n_trees = 20 if self.quick else full_trees
        self.results_dir = config.RESULTS_DIR / "quick" if self.quick else config.RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name, payload):
        """
        Write a result to results/<name>.json (or results/quick/ in a quick run) and print the path.

        Args:
            name: Name of the file without the extension.
            payload: Result that JSON can store.
        """
        path = self.results_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {path.relative_to(config.ROOT)}")

    def load(self, name):
        """
        Read a result file that an earlier script wrote.

        Args:
            name: Name of the file without the extension.

        Returns:
            The stored result.

        Raises:
            SystemExit: If the file does not exist. The message names the script that creates it.
        """
        path = self.results_dir / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"{path.relative_to(config.ROOT)} is missing: run "
                             f"python -m scripts.{name}{' --quick' if self.quick else ''} first")
        return json.loads(path.read_text())


def development_data():
    """
    Load the development set and encode it with the lossless tree encoder.

    The encoder is fitted on the whole development set. It uses no labels.

    Returns:
        A tuple: the matrix of codes, the labels and the fitted TreeEncoder.
    """
    data = load_dataset()
    dev, _ = development_generalisation_split(data.y)
    X, y = data.X.iloc[dev], data.y[dev]
    encoder = TreeEncoder().fit(X)
    return encoder.transform(X), y, encoder


def run_forest(codes, y, encoder, params, setup, label, diagnostics=False):
    """
    Grow one forest configuration with every seed and summarise its OOB predictions.

    Args:
        codes: Matrix of codes of the development set.
        y: Class label of every instance.
        encoder: The fitted TreeEncoder.
        params: Settings for RandomForest, as a dictionary.
        setup: The Setup object with the seeds and the number of trees.
        label: Name of the configuration for the printed line.
        diagnostics: If True, also compute strength and correlation.

    Returns:
        Dictionary with the label, params, the mean and the standard deviation of the OOB macro-F1 over the seeds, the mean growing time, the mean number of leaves per tree, and a list with the details of every seed.
    """
    runs = []
    start = time.perf_counter()
    for seed in setup.seeds:
        rf = RandomForest(n_trees=setup.n_trees, seed=seed, **params).fit(
            codes, y, encoder.n_codes, encoder.is_nominal, config.N_CLASSES,
            code_values=encoder.code_values(), keep_trees=False, diagnostics=diagnostics)
        soft, hard = rf.oob_predict("soft"), rf.oob_predict("hard")
        covered = soft >= 0
        runs.append({
            "seed": seed,
            "oob": summarise(y[covered], soft[covered]),
            "oob_hard_macro_f1": macro_f1(y[covered], hard[covered]),
            "oob_coverage": float(covered.mean()),
            "oob_curve": rf.oob_curve_,
            "fit_seconds": rf.fit_seconds_,
            "mean_leaves": float(np.mean(rf.tree_leaves_)),
            "diagnostics": rf.diagnostics_,
        })
    scores = [r["oob"]["macro_f1"] for r in runs]
    result = {
        "label": label,
        "params": params,
        "macro_f1_mean": float(np.mean(scores)),
        "macro_f1_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "seconds_mean": float(np.mean([r["fit_seconds"] for r in runs])),
        "leaves_mean": float(np.mean([r["mean_leaves"] for r in runs])),
        "runs": runs,
    }
    print(f"  {label:<34} macro-F1 {result['macro_f1_mean']:.4f} ± {result['macro_f1_sd']:.4f}"
          f"   ({time.perf_counter() - start:.0f} s)", flush=True)
    return result


def mean_over_runs(result, key):
    """
    Average one number over the seeds of a result.

    Args:
        result: Dictionary from run_forest.
        key: Path to the number inside each run, for example ("oob", "accuracy").

    Returns:
        The mean over the seeds. If the number is a list, the mean is taken per entry.
    """
    values = []
    for r in result["runs"]:
        v = r
        for k in key:
            v = v[k]
        values.append(v)
    return np.mean(values, axis=0)


def by_m(result):
    """Cost of a configuration that is used to break ties: the number of features per node."""
    return result["params"]["max_features"]


def by_leaves(result):
    """Cost of a configuration that is used to break ties: the mean number of leaves per tree."""
    return result["leaves_mean"]


def select(results, cost, spread="macro_f1_sd"):
    """
    Choose the winning configuration with the rule that was fixed in advance.

    The configuration with the best mean macro-F1 wins. Every configuration within spread of the best is tied, and the tied configuration with the lowest cost wins.
    The default spread is the seed standard deviation of the best configuration. A remaining tie keeps the earlier configuration.

    Args:
        results: List of result dictionaries with the key macro_f1_mean.
        cost: Function that gives the cost of a result. A lower cost is simpler.
        spread: Name of the key that holds the width of the tie band.

    Returns:
        Index of the winner in results.
    """
    means = np.array([r["macro_f1_mean"] for r in results])
    best = int(means.argmax())
    tied = [i for i in range(len(results)) if means[i] >= means[best] - results[best][spread]]
    return min(tied, key=lambda i: (cost(results[i]), i))
