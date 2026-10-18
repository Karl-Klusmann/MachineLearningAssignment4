"""Draw the figures of the report from the saved result files.
The figures are the out-of-bag (OOB) curve of E1, the macro-F1 (the mean over the classes of the F1 score) and the strength and correlation against m of E2, the k-nearest-neighbour (k-NN) tuning curves and the confusion matrix of the final forest.
Every figure is one column wide with 10 pt Times-like text. The series differ by marker and line style as well as colour, so the figures stay readable in grayscale.
Run: python -m scripts.make_figures [results_dir]. The default results_dir is results. The script writes the PDF files to report/figures/.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLUMN = 3.5
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SHORT = ["Nor", "Rec", "Bac", "DoS", "Exp", "Ana", "Fuz", "Wor", "She", "Gen"]

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "lines.markersize": 4.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(results, name):
    """
    Read one result file.

    Args:
        results: Folder with the result files.
        name: Name of the file without the extension.

    Returns:
        The stored result.
    """
    return json.loads((results / f"{name}.json").read_text())


def seed_curves(runs, key="soft"):
    """
    Collect the OOB macro-F1 curves of all seeds of one forest.

    Args:
        runs: List of run dictionaries from run_forest.
        key: "soft" or "hard" voting.

    Returns:
        A pair: an array with one row per seed and one column per forest size, and the list of forest sizes.
    """
    return np.array([r["oob_curve"][key] for r in runs]), runs[0]["oob_curve"]["trees"]


def fig_e1(results, out):
    """
    Draw the out-of-bag macro-F1 against the number of trees. The band shows one standard deviation over the seeds.

    Args:
        results: Folder with the result files.
        out: Folder for the figure.
    """
    e1 = load(results, "e1_forest_size")
    curves, trees = seed_curves(e1["config"]["runs"])
    mean, sd = curves.mean(0), curves.std(0, ddof=1)
    fig, ax = plt.subplots(figsize=(COLUMN, 2.1))
    ax.fill_between(trees, mean - sd, mean + sd, color=BLUE, alpha=0.2, linewidth=0)
    ax.plot(trees, mean, color=BLUE, marker="o")
    ax.axvline(e1["trees_within_tolerance"], color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.set_xlabel(r"Number of trees, $T$")
    ax.set_ylabel(r"OOB macro-$F_1$")
    fig.savefig(out / "e1_oob_curve.pdf")
    plt.close(fig)


def fig_e2(results, out):
    """
    Draw two figures for E2: the macro-F1 against m, and the strength and correlation against m.

    Args:
        results: Folder with the result files.
        out: Folder for the figures.
    """
    e2 = load(results, "e2_features")
    ms = e2["m_grid"]
    res = e2["results"]
    mean = np.array([r["macro_f1_mean"] for r in res])
    sd = np.array([r["macro_f1_sd"] for r in res])
    sk = e2["scikit_learn_rule"]

    fig, ax = plt.subplots(figsize=(COLUMN, 2.0))
    ax.errorbar(ms, mean, yerr=sd, color=BLUE, marker="o", capsize=2, elinewidth=0.8,
                label="varying features only")
    ax.plot([6], [sk["macro_f1_mean"]], linestyle="none", marker="s", markerfacecolor="white",
            markeredgecolor=ORANGE, markeredgewidth=1.2, label="all features counted")
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.minorticks_off()
    ax.set_xlabel(r"Features examined per node, $m$")
    ax.set_ylabel(r"OOB macro-$F_1$")
    ax.legend(loc="lower right")
    fig.savefig(out / "e2_macro_f1.pdf")
    plt.close(fig)

    def diag(r, part, key):
        """
        Average one diagnostic over the seeds of a configuration.

        Args:
            r: Result dictionary of one configuration.
            part: "all_instances" or "class_balanced".
            key: Name of the diagnostic, such as "strength" or "correlation".

        Returns:
            The mean over the seeds.
        """
        return np.mean([run["diagnostics"][part][key] for run in r["runs"]])

    s = [diag(r, "all_instances", "strength") for r in res]
    rho = [diag(r, "all_instances", "correlation") for r in res]
    fig, ax = plt.subplots(figsize=(COLUMN, 2.0))
    ax.plot(ms, s, color=BLUE, marker="o", label=r"strength, $s$")
    ax.plot(ms, rho, color=ORANGE, marker="^", linestyle="--", label=r"correlation, $\bar\rho$")
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.minorticks_off()
    ax.set_xlabel(r"Features examined per node, $m$")
    ax.set_ylabel("Value")
    ax.legend(loc="center right")
    fig.savefig(out / "e2_strength.pdf")
    plt.close(fig)


def fig_knn(results, out):
    """
    Draw the cross-validated macro-F1 of the k-NN classifier against k for three voting rules.

    Args:
        results: Folder with the result files.
        out: Folder for the figure.
    """
    tk = load(results, "tune_knn")
    series = [("uniform", 0.0, BLUE, "o", "-", r"uniform, $\beta = 0$"),
              ("uniform", 0.5, ORANGE, "^", "--", r"uniform, $\beta = 0.5$"),
              ("distance", 0.5, AQUA, "s", "-.", r"distance, $\beta = 0.5$")]
    fig, ax = plt.subplots(figsize=(COLUMN, 2.2))
    for wt, b, colour, marker, ls, label in series:
        rows = sorted((r for r in tk["results"]
                       if r["params"]["weighting"] == wt and r["params"]["beta"] == b),
                      key=lambda r: r["params"]["k"])
        k = [r["params"]["k"] for r in rows]
        f = [r["macro_f1_mean"] for r in rows]
        ax.plot(k, f, color=colour, marker=marker, linestyle=ls, label=label,
                markeredgecolor=INK if colour == AQUA else colour, markeredgewidth=0.4)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 7, 11, 21, 51, 101])
    ax.set_xticklabels(["1", "3", "7", "11", "21", "51", "101"])
    ax.minorticks_off()
    ax.set_xlabel(r"Number of neighbours, $k$")
    ax.set_ylabel(r"CV macro-$F_1$")
    ax.set_ylim(0.405, 0.545)
    ax.legend(loc="upper left")
    fig.savefig(out / "knn_tuning.pdf")
    plt.close(fig)


def fig_confusion(results, out):
    """
    Draw the confusion matrix of the final forest. Each row shows the percentage of the true class that went to each predicted class.

    Args:
        results: Folder with the result files.
        out: Folder for the figure.
    """
    e6 = load(results, "e6_final")
    cm = np.array(e6["scores"]["final forest"]["summary"]["confusion_matrix"], float)
    share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(COLUMN, 3.1))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            pct = 100 * share[i, j]
            if pct >= 0.5:
                ax.text(j, i, f"{pct:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(10))
    ax.set_xticklabels(SHORT, rotation=90)
    ax.set_yticks(range(10))
    ax.set_yticklabels(SHORT)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.savefig(out / "e6_confusion.pdf")
    plt.close(fig)


def main():
    """Draw every figure. The folder of the results can be given as the first command-line argument."""
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    out = Path("report/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig_e1(results, out)
    fig_e2(results, out)
    fig_knn(results, out)
    fig_confusion(results, out)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
