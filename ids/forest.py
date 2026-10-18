"""Random forest of Breiman (2001), built from the trees of ids/tree.py.
Every tree is grown on its own random sample of the training data and examines only m randomly chosen features at each node.
An instance that a sample leaves out is out-of-bag (OOB) for that tree, and the OOB instances give a free test set.
The forest combines the trees by soft voting or hard voting and computes the OOB estimate, the strength and the correlation of the trees.
The report describes the forest in the section on forest construction, and Algorithm 1 lists the steps.
The experiments E1 to E6 use this file, and tests/test_stage3.py checks it.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config
from .metrics import macro_f1
from .tree import ClassificationTree, _apply, class_weights

DEFAULT_CHECKPOINTS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500,
                       700, 1000)


def draw_multiplicities(y, members, sampling, sample_size, rng):
    """
    Draw the training sample of one tree as a count for every instance.

    A bootstrap sample draws instances at random with replacement. Some instances appear more than once and others not at all.
    The three kinds of sampling are:
    bootstrap: round(sample_size * n) draws from all n instances, where sample_size is a share between 0 and 1. None means 1.
    balanced: sample_size draws from every class. None means the size of the smallest class.
    capped: min(class size, sample_size) draws from every class. Large classes are reduced and small classes keep their size.
    All draws are with replacement.

    Args:
        y: Class label of every training instance.
        members: List with one array per class, holding the indices of the instances of that class.
        sampling: "bootstrap", "balanced" or "capped".
        sample_size: Size of the sample as described above.
        rng: NumPy random generator.

    Returns:
        Array with the number of times each instance was drawn. An instance with 0 is out-of-bag.

    Raises:
        ValueError: If sample_size is not valid for the kind of sampling, or the kind of sampling is unknown.
    """
    n = len(y)
    if sampling == "bootstrap":
        f = 1.0 if sample_size is None else float(sample_size)
        if not 0.0 < f <= 1.0:
            raise ValueError(f"bootstrap sample_size must be a fraction in (0, 1], got {sample_size}")
        return np.bincount(rng.integers(0, n, int(round(f * n))), minlength=n)
    if sampling not in ("balanced", "capped"):
        raise ValueError(f"unknown sampling {sampling!r}")
    if sample_size is None and sampling == "capped":
        raise ValueError("capped sampling needs sample_size, the cap per class")
    if sample_size is not None and (sample_size < 2 or sample_size != int(sample_size)):
        raise ValueError(f"{sampling} sample_size must be a whole number of draws >= 2, "
                         f"got {sample_size}")
    sizes = [len(m) for m in members if len(m) > 0]
    s = min(sizes) if sample_size is None else int(sample_size)
    draws = []
    for m in members:
        if len(m) == 0:
            continue
        k = s if sampling == "balanced" else min(s, len(m))
        draws.append(m[rng.integers(0, len(m), k)])
    return np.bincount(np.concatenate(draws), minlength=n)


def resolve_max_features(max_features, n_features):
    """
    Turn the setting for the number of features per node into a number.

    Args:
        max_features: None or "all" for every feature, "sqrt" for the square root of the number of features rounded down (at least 1), or a whole number.
        n_features: Number of features.

    Returns:
        The number m of features examined at each node, between 1 and n_features.
    """
    if max_features is None or max_features == "all":
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    return int(min(max(1, max_features), n_features))


def vote(soft, hard, count):
    """
    Turn summed tree outputs into soft and hard predictions.

    Soft voting picks the class with the largest summed class share. Hard voting picks the class with the most tree votes.
    A tie in hard voting goes to the tied class with the larger soft score.

    Args:
        soft: Array with one row per instance and one column per class. Each entry is the sum of the class shares of the trees that voted.
        hard: Array of the same shape with the number of trees whose majority class is that class.
        count: Number of trees that voted on each instance.

    Returns:
        A pair of arrays with the soft prediction and the hard prediction of every instance.
    """
    soft_pred = soft.argmax(axis=1)
    # Every soft entry is at most count, so soft / (count + 1) is below 1. It can only break ties between equal vote counts.
    hard_pred = (hard + soft / (count[:, None] + 1.0)).argmax(axis=1)
    return soft_pred, hard_pred


def breiman_strength_correlation(tree_pred, y, n_classes, balanced=False):
    """
    Compute the strength and the mean correlation of the trees from their OOB predictions.

    The margin of an instance is the share of OOB trees that vote for the true class minus the largest share for a wrong class.
    The strength is the mean margin. The correlation is the variance of the margin divided by the squared mean of the standard deviations of the trees (Breiman 2001, Appendix II).
    The bound on the generalisation error is correlation * (1 - strength^2) / strength^2.
    Instances that are never out-of-bag are left out.

    Args:
        tree_pred: Array with one row per tree and one column per instance. Each entry is the class predicted by the tree, or -1 if the instance was in the sample of the tree.
        y: True class of every instance.
        n_classes: Number of classes.
        balanced: If True, every class carries the same total weight, so the result describes all classes equally. If False, large classes dominate.

    Returns:
        Dictionary with strength, correlation, bound, margin_variance and oob_error. A quantity that is undefined is None.
    """
    n_trees, n = tree_pred.shape
    oob = tree_pred >= 0
    count = oob.sum(axis=0)
    covered = count > 0
    votes = np.zeros((n, n_classes))
    for c in range(n_classes):
        votes[:, c] = (tree_pred == c).sum(axis=0)
    Q = votes / np.maximum(count, 1)[:, None]
    rows = np.arange(n)
    q_true = Q[rows, y]
    Q_wrong = Q.copy()
    Q_wrong[rows, y] = -np.inf
    j_star = Q_wrong.argmax(axis=1)
    margin = q_true - Q[rows, j_star]

    # Weights: 1 for each covered instance, or 1 / class size when balanced.
    weight = covered.astype(float)
    if balanced:
        n_c = np.bincount(y[covered], minlength=n_classes)
        weight = weight / np.maximum(n_c[y], 1)
    weight /= weight.sum()
    strength = float(np.sum(weight * margin))
    variance = float(np.sum(weight * margin ** 2) - strength ** 2)

    # sd[t] is the standard deviation of the raw margin of tree t. It uses p1, the share of its OOB instances that it labels correctly, and p2, the share that it assigns to the best wrong class.
    sd = np.full(n_trees, np.nan)
    for t in range(n_trees):
        w_t = weight * oob[t]
        total = w_t.sum()
        if total == 0:
            continue
        p1 = np.sum(w_t * (tree_pred[t] == y)) / total
        p2 = np.sum(w_t * (tree_pred[t] == j_star)) / total
        sd[t] = np.sqrt(max(p1 + p2 - (p1 - p2) ** 2, 0.0))
    mean_sd = np.nanmean(sd)
    correlation = float(variance / mean_sd ** 2) if mean_sd > 0 else None
    bound = (float(correlation * (1 - strength ** 2) / strength ** 2)
             if correlation is not None and strength > 0 else None)
    return {"strength": strength, "correlation": correlation, "bound": bound,
            "margin_variance": variance, "oob_error": float(np.sum(weight * (margin < 0)))}


class _CompactTree:
    """The parts of a fitted tree that prediction needs, stored in small data types to save memory."""

    def __init__(self, tree):
        """
        Copy the node arrays of a tree and turn its class totals into class shares.

        Args:
            tree: A fitted ClassificationTree, pruned or not.
        """
        self.feature = tree.feature.astype(np.int32)
        self.code = tree.code.astype(np.int32)
        self.left = tree.left.astype(np.int32)
        self.right = tree.right.astype(np.int32)
        self.proba = (tree.value / tree.value.sum(axis=1, keepdims=True)).astype(np.float32)


class RandomForest:
    """
    Random forest of classification trees.

    Attributes:
        n_trees: Number of trees T.
        max_features: Features examined per node, m. Use "sqrt", "all", None or a whole number.
        min_samples_leaf: Smallest number of instances in a leaf of each tree.
        max_depth: Largest depth of each tree, or None for fully grown trees.
        sampling: "bootstrap", "balanced" or "capped". See draw_multiplicities.
        sample_size: Size of the sample of each tree. See draw_multiplicities.
        class_beta: Class weights are proportional to n_c^(-class_beta), computed from the sample of each tree. 0 means no weighting and 1 gives every class the same total weight.
        prune_alpha: If above 0, every tree is pruned with this alpha. None or 0 means no pruning.
        count_constant: If True, features that are constant in a node use up the feature budget, as in scikit-learn.
        seed: Seed of the forest. Tree t draws from the stream rng("forest", seed, t).
        n_jobs: Number of threads that grow trees. None means all cores.
    """

    def __init__(self, n_trees=100, max_features="sqrt", min_samples_leaf=1, max_depth=None,
                 sampling="bootstrap", sample_size=None, class_beta=1.0, prune_alpha=None,
                 count_constant=False, seed=0, n_jobs=None):
        """
        Store the settings of the forest. The trees are grown in fit.

        Args:
            n_trees: Number of trees.
            max_features: Features examined per node.
            min_samples_leaf: Smallest number of instances in a leaf.
            max_depth: Largest depth of each tree, or None.
            sampling: Kind of sampling.
            sample_size: Size of the sample of each tree.
            class_beta: Exponent of the class weights.
            prune_alpha: Pruning parameter, or None.
            count_constant: Whether constant features use up the feature budget.
            seed: Seed of the forest.
            n_jobs: Number of threads, or None for all cores.
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.sampling = sampling
        self.sample_size = sample_size
        self.class_beta = class_beta
        self.prune_alpha = prune_alpha
        self.count_constant = count_constant
        self.seed = seed
        self.n_jobs = n_jobs

    def tree_sample(self, t, y, members):
        """
        Draw the sample of tree t and a seed for its growth.

        Both depend only on the seed of the forest and on t, so they can be repeated on their own.

        Args:
            t: Index of the tree.
            y: Class label of every training instance.
            members: List with the indices of the instances of each class.

        Returns:
            A pair: the multiplicity of every instance and the seed of the tree.
        """
        g = config.rng("forest", self.seed, t)
        mult = draw_multiplicities(y, members, self.sampling, self.sample_size, g)
        return mult, int(g.integers(2 ** 31 - 1))

    def _grow_one(self, t, X, y, members, eval_X):
        """
        Grow tree t and collect everything the forest needs from it.

        Args:
            t: Index of the tree.
            X: Matrix of codes of the training data.
            y: Class label of every training instance.
            members: List with the indices of the instances of each class.
            eval_X: List of code matrices of the evaluation sets.

        Returns:
            Dictionary with the compact tree (or None), the OOB indices, the class shares of the tree on the OOB instances, the class shares on every evaluation set, the number of leaves and the growing time in seconds.
        """
        start = time.perf_counter()
        mult, tree_seed = self.tree_sample(t, y, members)
        w = class_weights(y, self.n_classes, self.class_beta, multiplicity=mult)
        tree = ClassificationTree(self._m, self.min_samples_leaf, self.max_depth, tree_seed,
                                  self.count_constant).fit(
            X, y, self._n_codes, self.is_nominal, self.n_classes, class_weight=w,
            multiplicity=mult, code_values=self._code_values)
        if self.prune_alpha:
            tree = tree.prune(self.prune_alpha)
        compact = _CompactTree(tree)

        def proba(rows_X):
            """
            Return the class shares that the tree gives to every row.

            Args:
                rows_X: Matrix of codes with one row per instance.

            Returns:
                Array with one row per instance and one column per class.
            """
            leaves = _apply(rows_X, compact.feature, compact.code, compact.left, compact.right,
                            self.is_nominal)
            return compact.proba[leaves]

        oob = np.flatnonzero(mult == 0)
        return {
            "tree": compact if self._keep_trees else None,
            "oob": oob,
            "oob_proba": proba(X[oob]),
            "eval_proba": [proba(Xe) for Xe in eval_X],
            "leaves": tree.n_leaves,
            "seconds": time.perf_counter() - start,
        }

    def fit(self, X, y, n_codes, is_nominal, n_classes, code_values=None, eval_sets=None,
            keep_trees=True, diagnostics=False, checkpoints=DEFAULT_CHECKPOINTS):
        """
        Grow the forest on the training data.

        Trees grow in parallel threads. The results are added in tree order, so every score is the same for any number of threads.
        The OOB macro-F1 (the mean F1 score over the classes) is recorded at the forest sizes in checkpoints while the forest grows.

        Args:
            X: Matrix of codes with one row per instance and one column per feature.
            y: Class label of every instance.
            n_codes: Number of codes of every feature.
            is_nominal: True for each nominal feature.
            n_classes: Number of classes.
            code_values: Pair (values, offsets) from TreeEncoder.code_values, or None.
            eval_sets: Dictionary from a name to a pair (X_eval, y_eval). The sets are predicted while the forest grows, so a large forest can be scored without keeping its trees. With y_eval set to None, no score is recorded.
            keep_trees: If True, keep the trees so that predict can be used later.
            diagnostics: If True, record every tree's OOB predictions to compute strength and correlation.
            checkpoints: Forest sizes at which the macro-F1 is recorded.

        Returns:
            The fitted forest. The attributes oob_curve_, eval_curves_, oob_count_, oob_proba_, eval_proba_, diagnostics_, tree_leaves_ and fit_seconds_ hold the results.
        """
        X = np.ascontiguousarray(X)
        y = np.asarray(y, np.int64)
        n, n_features = X.shape
        self.n_classes = n_classes
        self.is_nominal = np.asarray(is_nominal, np.bool_)
        self._n_codes = np.asarray(n_codes, np.int64)
        self._code_values = code_values
        self._m = resolve_max_features(self.max_features, n_features)
        self._keep_trees = keep_trees
        members = [np.flatnonzero(y == c) for c in range(n_classes)]
        eval_sets = eval_sets or {}
        eval_names = list(eval_sets)
        eval_X = [np.ascontiguousarray(eval_sets[k][0]) for k in eval_names]
        eval_y = [eval_sets[k][1] for k in eval_names]

        oob_soft = np.zeros((n, n_classes))
        oob_hard = np.zeros((n, n_classes))
        oob_count = np.zeros(n)
        ev_soft = [np.zeros((len(Xe), n_classes)) for Xe in eval_X]
        ev_hard = [np.zeros((len(Xe), n_classes)) for Xe in eval_X]
        tree_pred = np.full((self.n_trees, n), -1, np.int8) if diagnostics else None
        # Forest sizes at which the scores are recorded.
        marks = sorted({T for T in checkpoints if T <= self.n_trees} | {self.n_trees})
        self.oob_curve_ = {"trees": [], "soft": [], "hard": [], "coverage": []}
        self.eval_curves_ = {k: {"trees": [], "soft": [], "hard": []}
                             for k, yk in zip(eval_names, eval_y) if yk is not None}
        self.trees_, self.tree_leaves_, self.tree_seconds_ = [], [], []

        def grow(t):
            """
            Grow tree t. This is the function that the threads run.

            Args:
                t: Index of the tree.

            Returns:
                The dictionary of RandomForest._grow_one.
            """
            return self._grow_one(t, X, y, members, eval_X)

        n_jobs = self.n_jobs or os.cpu_count() or 1
        start = time.perf_counter()
        pool = ThreadPoolExecutor(n_jobs) if n_jobs > 1 else None
        results = pool.map(grow, range(self.n_trees)) if pool else map(grow, range(self.n_trees))
        # pool.map returns the results in tree order, so the sums do not depend on the thread count.
        for t, res in enumerate(results):
            o, p = res["oob"], res["oob_proba"]
            pred = p.argmax(axis=1)
            oob_soft[o] += p
            oob_hard[o, pred] += 1
            oob_count[o] += 1
            if diagnostics:
                tree_pred[t, o] = pred
            for k, p in enumerate(res["eval_proba"]):
                ev_soft[k] += p
                ev_hard[k][np.arange(len(p)), p.argmax(axis=1)] += 1
            if keep_trees:
                self.trees_.append(res["tree"])
            self.tree_leaves_.append(res["leaves"])
            self.tree_seconds_.append(res["seconds"])

            if t + 1 in marks:
                covered = oob_count > 0
                soft_pred, hard_pred = vote(oob_soft[covered], oob_hard[covered], oob_count[covered])
                self.oob_curve_["trees"].append(t + 1)
                self.oob_curve_["soft"].append(macro_f1(y[covered], soft_pred, n_classes))
                self.oob_curve_["hard"].append(macro_f1(y[covered], hard_pred, n_classes))
                self.oob_curve_["coverage"].append(float(covered.mean()))
                for k, name in enumerate(eval_names):
                    if name in self.eval_curves_:
                        sp, hp = vote(ev_soft[k], ev_hard[k], np.full(len(ev_soft[k]), t + 1.0))
                        self.eval_curves_[name]["trees"].append(t + 1)
                        self.eval_curves_[name]["soft"].append(macro_f1(eval_y[k], sp, n_classes))
                        self.eval_curves_[name]["hard"].append(macro_f1(eval_y[k], hp, n_classes))
        if pool:
            pool.shutdown()
        self.fit_seconds_ = time.perf_counter() - start

        self.oob_count_ = oob_count
        self.oob_proba_ = oob_soft / np.maximum(oob_count, 1)[:, None]
        self._oob_sums = (oob_soft, oob_hard)
        self.eval_proba_ = {k: ev_soft[i] / self.n_trees for i, k in enumerate(eval_names)}
        self._eval_sums = {k: (ev_soft[i], ev_hard[i]) for i, k in enumerate(eval_names)}
        self.diagnostics_ = None
        if diagnostics:
            self.diagnostics_ = {
                "all_instances": breiman_strength_correlation(tree_pred, y, n_classes),
                "class_balanced": breiman_strength_correlation(tree_pred, y, n_classes, True),
            }
        return self

    def oob_predict(self, voting="soft"):
        """
        Predict every training instance with the trees that did not see it.

        Args:
            voting: "soft" or "hard".

        Returns:
            Array with the predicted class of every training instance, or -1 for an instance that was never out-of-bag.
        """
        soft_pred, hard_pred = vote(*self._oob_sums, self.oob_count_)
        pred = soft_pred if voting == "soft" else hard_pred
        return np.where(self.oob_count_ > 0, pred, -1)

    def eval_predict(self, name, voting="soft"):
        """
        Predict an evaluation set that was given to fit, using all trees.

        Args:
            name: Name of the evaluation set.
            voting: "soft" or "hard".

        Returns:
            Array with the predicted class of every instance of the set.
        """
        soft, hard = self._eval_sums[name]
        soft_pred, hard_pred = vote(soft, hard, np.full(len(soft), float(self.n_trees)))
        return soft_pred if voting == "soft" else hard_pred

    def _sums(self, X):
        """
        Add up the outputs of all kept trees for new data.

        Args:
            X: Matrix of codes with one row per instance.

        Returns:
            A pair of arrays with one row per instance and one column per class: the summed class shares and the number of trees that vote for each class.

        Raises:
            RuntimeError: If the forest was fitted with keep_trees=False.
        """
        if not self.trees_:
            raise RuntimeError("fit with keep_trees=True to predict new data")
        X = np.ascontiguousarray(X)
        soft = np.zeros((len(X), self.n_classes))
        hard = np.zeros((len(X), self.n_classes))
        rows = np.arange(len(X))
        for tr in self.trees_:
            p = tr.proba[_apply(X, tr.feature, tr.code, tr.left, tr.right, self.is_nominal)]
            soft += p
            hard[rows, p.argmax(axis=1)] += 1
        return soft, hard

    def predict_proba(self, X):
        """
        Estimate the class shares of new data as the mean of the class shares of the trees.

        Args:
            X: Matrix of codes with one row per instance.

        Returns:
            Array with one row per instance and one column per class.
        """
        return self._sums(X)[0] / len(self.trees_)

    def predict(self, X, voting="soft"):
        """
        Predict the class of new data by soft or hard voting.

        Args:
            X: Matrix of codes with one row per instance.
            voting: "soft" or "hard".

        Returns:
            Array with the predicted class of every instance.
        """
        soft, hard = self._sums(X)
        soft_pred, hard_pred = vote(soft, hard, np.full(len(X), float(len(self.trees_))))
        return soft_pred if voting == "soft" else hard_pred
