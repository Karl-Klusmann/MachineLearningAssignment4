"""The classification tree of the project, written from first principles.
It splits on the Gini impurity, accepts class weights and instance multiplicities, and supports cost-complexity pruning.
The tree works on the whole-number codes of ids/encoding.py. A numeric test is "code <= b" and a nominal test is "code == v".
The report describes the tree in the section on tree induction. The forest, experiments E3 and E5 and the final comparison use it.
The growing loop is compiled with Numba, which turns plain loops into fast machine code.
"""

import numpy as np
from numba import njit

LEAF = -1


@njit(cache=True, nogil=True)
def _split_impurity(left_dist, parent_dist, parent_w):
    """
    Compute the weighted Gini impurity of one candidate split.

    The impurity of a child with total weight W and class weights w_c is W * (1 - sum of (w_c / W)^2). This equals W - sum of w_c^2 / W.
    The result is the sum over the two children. A lower value means a purer split.
    The class weights of the right child are the parent weights minus the left weights.

    Args:
        left_dist: Weighted class totals of the left child, one entry per class.
        parent_dist: Weighted class totals of the node that is split.
        parent_w: Total weight of the node that is split.

    Returns:
        The weighted impurity of the two children.
    """
    w_left, sq_left, sq_right = 0.0, 0.0, 0.0
    for c in range(parent_dist.shape[0]):
        a = left_dist[c]
        b = parent_dist[c] - a
        w_left += a
        sq_left += a * a
        sq_right += b * b
    w_right = parent_w - w_left
    impurity = 0.0
    if w_left > 0.0:
        impurity += w_left - sq_left / w_left
    if w_right > 0.0:
        impurity += w_right - sq_right / w_right
    return impurity


@njit(cache=True, nogil=True)
def _next_occupied(hist_n, b):
    """
    Find the smallest code above b that occurs in the current node.

    Args:
        hist_n: Number of instances per code in the node. The function must find a nonzero entry above b.
        b: Code to start above.

    Returns:
        The next code with at least one instance.
    """
    b += 1
    while hist_n[b] == 0:
        b += 1
    return b


@njit(cache=True, nogil=True)
def _grow(X, y, mult, class_weight, n_codes, is_nominal, code_values, value_offset,
          n_classes, max_features, min_samples_leaf, max_depth, seed, count_constant):
    """
    Grow one tree with an explicit stack and return its nodes as arrays.

    Every node tests the features in a fresh random order and keeps the split with the lowest weighted Gini impurity.
    A node stays a leaf if it is pure, if it is too small for two leaves of the minimum size, if it is too deep, or if no allowed split exists.
    Children are numbered after their parent, so a child always has a larger index than its parent.

    Args:
        X: Matrix of codes with one row per instance and one column per feature.
        y: Class label of every instance.
        mult: Number of times each instance occurs in the sample. An instance with 0 is left out.
        class_weight: Weight of every class.
        n_codes: Number of codes of every feature.
        is_nominal: True for each nominal feature.
        code_values: Value of every code, all features joined end to end.
        value_offset: Start position of every feature in code_values.
        n_classes: Number of classes.
        max_features: Number of features examined at each node.
        min_samples_leaf: Smallest number of instances, counted with multiplicity, in a leaf.
        max_depth: Largest depth of the tree, or -1 for no limit.
        seed: Seed of the random number generator used inside the compiled code.
        count_constant: If True, a feature that is constant in the node also uses up the budget of max_features, as scikit-learn does.

    Returns:
        Six arrays with one entry per node: tested feature, tested code, left child, right child, number of instances and weighted class totals. A leaf has -1 as its children.
    """
    np.random.seed(seed)
    n_features = X.shape[1]
    rows = np.flatnonzero(mult > 0)
    n_rows = rows.shape[0]
    # A binary tree with at most n_rows leaves has fewer than 2 * n_rows nodes.
    capacity = 2 * n_rows + 1

    feature = np.full(capacity, LEAF, np.int64)
    code = np.zeros(capacity, np.int64)
    left = np.full(capacity, LEAF, np.int64)
    right = np.full(capacity, LEAF, np.int64)
    depth = np.zeros(capacity, np.int64)
    count = np.zeros(capacity, np.int64)
    value = np.zeros((capacity, n_classes))

    # Node k owns rows[start[k]:end[k]]. Its two children split that range in place.
    start = np.zeros(capacity, np.int64)
    end = np.zeros(capacity, np.int64)
    end[0] = n_rows
    stack = np.zeros(capacity, np.int64)
    top = 1
    n_nodes = 1

    max_codes = n_codes.max() if n_features > 0 else 1
    hist_w = np.zeros((max_codes, n_classes))
    hist_n = np.zeros(max_codes, np.int64)
    occupied = np.zeros(max_codes, np.int64)
    left_w = np.zeros(n_classes)
    order = np.arange(n_features)

    while top > 0:
        top -= 1
        k = stack[top]

        for i in range(start[k], end[k]):
            r = rows[i]
            value[k, y[r]] += mult[r] * class_weight[y[r]]
            count[k] += mult[r]
        total_w = value[k].sum()

        n_present = 0
        for c in range(n_classes):
            if value[k, c] > 0.0:
                n_present += 1
        if n_present <= 1 or count[k] < 2 * min_samples_leaf or \
                (max_depth >= 0 and depth[k] >= max_depth):
            continue

        # Fisher-Yates shuffle of the feature order. The first features in the order are the candidates, and ties go to the earlier one.
        for i in range(n_features - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            order[i], order[j] = order[j], order[i]

        best_imp = np.inf
        best_f = -1
        best_c = -1
        best_next = -1
        n_visited = 0
        n_varying = 0
        for oi in range(n_features):
            # Two rules for the feature budget. The scikit-learn rule counts every examined feature but always examines one varying feature. The default rule counts only features that vary in the node.
            if count_constant:
                if n_visited >= max_features and n_varying > 0:
                    break
            elif n_varying >= max_features:
                break
            f = order[oi]
            n_visited += 1

            n_occ = 0
            for i in range(start[k], end[k]):
                r = rows[i]
                b = X[r, f]
                if hist_n[b] == 0:
                    occupied[n_occ] = b
                    n_occ += 1
                hist_w[b, y[r]] += mult[r] * class_weight[y[r]]
                hist_n[b] += mult[r]

            if n_occ >= 2:
                n_varying += 1
                occ = occupied[:n_occ]
                if is_nominal[f]:
                    for v in occ:
                        n_left = hist_n[v]
                        if n_left < min_samples_leaf or count[k] - n_left < min_samples_leaf:
                            continue
                        imp = _split_impurity(hist_w[v], value[k], total_w)
                        if imp < best_imp:
                            best_imp, best_f, best_c = imp, f, v
                else:
                    K = n_codes[f]
                    # If the node covers at least one eighth of all codes, walking the full code range is cheaper than sorting the occupied codes.
                    dense = 8 * n_occ >= K
                    if not dense:
                        occ.sort()
                    left_w[:] = 0.0
                    n_left = 0
                    for i in range(K - 1 if dense else n_occ - 1):
                        b = i if dense else occ[i]
                        # A code that is absent gives the same split as the previous code.
                        if dense and hist_n[b] == 0:
                            continue
                        left_w += hist_w[b]
                        n_left += hist_n[b]
                        if n_left < min_samples_leaf:
                            continue
                        if count[k] - n_left < min_samples_leaf:
                            break
                        imp = _split_impurity(left_w, value[k], total_w)
                        if imp < best_imp:
                            nxt = _next_occupied(hist_n, b) if dense else occ[i + 1]
                            best_imp, best_f, best_c, best_next = imp, f, b, nxt

            for i in range(n_occ):
                b = occupied[i]
                hist_w[b, :] = 0.0
                hist_n[b] = 0

        if best_f < 0:
            continue

        if not is_nominal[best_f]:
            off = value_offset[best_f]
            low, high = code_values[off + best_c], code_values[off + best_next]
            # Move the threshold to the midpoint of the gap. Then use the largest code whose value does not exceed the midpoint, so unseen values inside the gap go to the correct side.
            mid = low + (high - low) / 2.0
            gap = code_values[off + best_c: off + best_next]
            best_c = best_c + np.searchsorted(gap, mid, side="right") - 1

        lo, hi = start[k], end[k] - 1
        while lo <= hi:
            b = X[rows[lo], best_f]
            goes_left = (b == best_c) if is_nominal[best_f] else (b <= best_c)
            if goes_left:
                lo += 1
            else:
                rows[lo], rows[hi] = rows[hi], rows[lo]
                hi -= 1

        feature[k], code[k] = best_f, best_c
        for child, s, e in ((n_nodes, start[k], lo), (n_nodes + 1, lo, end[k])):
            start[child], end[child] = s, e
            depth[child] = depth[k] + 1
            stack[top] = child
            top += 1
        left[k], right[k] = n_nodes, n_nodes + 1
        n_nodes += 2

    return (feature[:n_nodes].copy(), code[:n_nodes].copy(), left[:n_nodes].copy(),
            right[:n_nodes].copy(), count[:n_nodes].copy(), value[:n_nodes].copy())


@njit(cache=True, nogil=True)
def _apply(X, feature, code, left, right, is_nominal):
    """
    Send every instance down the tree and return the leaf that it reaches.

    Args:
        X: Matrix of codes with one row per instance.
        feature: Tested feature of every node.
        code: Tested code of every node.
        left: Left child of every node, or -1 for a leaf.
        right: Right child of every node.
        is_nominal: True for each nominal feature.

    Returns:
        Array with the index of the leaf of every instance.
    """
    leaves = np.empty(X.shape[0], np.int64)
    for r in range(X.shape[0]):
        k = 0
        while left[k] != LEAF:
            f = feature[k]
            b = X[r, f]
            goes_left = (b == code[k]) if is_nominal[f] else (b <= code[k])
            k = left[k] if goes_left else right[k]
        leaves[r] = k
    return leaves


def class_weights(y, n_classes, beta=1.0, multiplicity=None):
    """
    Compute class weights proportional to n_c^(-beta), where n_c is the size of class c.

    With beta = 0 every class gets the same weight. With beta = 1 every class gets the same total weight.
    The weights are scaled so that the total weight of the sample equals the size of the sample.
    Each instance counts as many times as its multiplicity. A class that does not occur gets weight 0.

    Args:
        y: Class label of every instance.
        n_classes: Number of classes.
        beta: Exponent of the weighting.
        multiplicity: Number of times each instance occurs in the sample, or None for once each.

    Returns:
        Array with one weight per class.
    """
    mult = np.ones(len(y)) if multiplicity is None else multiplicity
    n_c = np.bincount(y, weights=mult, minlength=n_classes)
    w = np.zeros(n_classes)
    present = n_c > 0
    w[present] = n_c[present] ** (-beta)
    return w * n_c.sum() / np.sum(w * n_c)


class ClassificationTree:
    """
    Binary classification tree on whole-number codes.

    Attributes:
        max_features: Number of features examined at each node, or None for all.
        min_samples_leaf: Smallest number of instances, counted with multiplicity, in a leaf.
        max_depth: Largest depth, or None to grow until the leaves are pure or minimal.
        seed: Seed for the choice of features and for ties between equal splits.
        count_constant: If True, constant features use up the feature budget at a node, as in scikit-learn. A deep node then examines fewer useful features.
    """

    def __init__(self, max_features=None, min_samples_leaf=1, max_depth=None, seed=0,
                 count_constant=False):
        """
        Store the settings of the tree. The tree is grown in fit.

        Args:
            max_features: Number of features examined at each node, or None for all.
            min_samples_leaf: Smallest number of instances, counted with multiplicity, in a leaf.
            max_depth: Largest depth, or None for no limit.
            seed: Seed for the random choices.
            count_constant: Whether constant features use up the feature budget.
        """
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.seed = seed
        self.count_constant = count_constant

    def fit(self, X, y, n_codes, is_nominal, n_classes, class_weight=None, multiplicity=None,
            code_values=None):
        """
        Grow the tree on the training data.

        Args:
            X: Matrix of codes with one row per instance and one column per feature.
            y: Class label of every instance.
            n_codes: Number of codes of every feature.
            is_nominal: True for each nominal feature.
            n_classes: Number of classes.
            class_weight: Weight of every class, or None for weight 1 each.
            multiplicity: Number of times each instance occurs in the sample, or None for once each. A forest passes a bootstrap sample this way.
            code_values: Pair (values, offsets) from TreeEncoder.code_values, or None. Without it, the codes themselves are the values and a threshold lies midway between codes.

        Returns:
            The fitted tree.
        """
        n_features = X.shape[1]
        n_codes = np.asarray(n_codes, np.int64)
        if code_values is None:
            offsets = np.concatenate([[0], np.cumsum(n_codes)]).astype(np.int64)
            values = np.concatenate([np.arange(k, dtype=float) for k in n_codes])
        else:
            values, offsets = code_values
        mult = np.ones(len(y), np.int64) if multiplicity is None else np.asarray(multiplicity, np.int64)
        cw = np.ones(n_classes) if class_weight is None else np.asarray(class_weight, float)
        m = n_features if self.max_features is None else min(int(self.max_features), n_features)

        self.is_nominal = np.asarray(is_nominal, np.bool_)
        self.n_classes = n_classes
        (self.feature, self.code, self.left, self.right,
         self.count, self.value) = _grow(
            X, np.asarray(y, np.int64), mult, cw, n_codes, self.is_nominal,
            np.asarray(values, float), np.asarray(offsets, np.int64), n_classes, m,
            int(self.min_samples_leaf),
            -1 if self.max_depth is None else int(self.max_depth), int(self.seed),
            bool(self.count_constant))
        return self

    def apply(self, X):
        """
        Return the leaf that every instance reaches.

        Args:
            X: Matrix of codes with one row per instance.

        Returns:
            Array with the node index of the leaf of every instance.
        """
        return _apply(X, self.feature, self.code, self.left, self.right, self.is_nominal)

    def predict_proba(self, X):
        """
        Estimate the class shares of every instance from its leaf.

        Args:
            X: Matrix of codes with one row per instance.

        Returns:
            Array with one row per instance and one column per class. The weighted class totals of the leaf are divided by their sum.
        """
        dist = self.value[self.apply(X)]
        return dist / dist.sum(axis=1, keepdims=True)

    def predict(self, X):
        """
        Predict the class with the largest weighted total in the leaf of every instance.

        Args:
            X: Matrix of codes with one row per instance.

        Returns:
            Array with the predicted class of every instance.
        """
        return self.value[self.apply(X)].argmax(axis=1)

    def _reachable(self):
        """
        List the nodes that can be reached from the root.

        After pruning, the arrays still hold the removed nodes. They are not reachable.

        Returns:
            Array of node indices.
        """
        nodes, stack = [], [0]
        while stack:
            k = stack.pop()
            nodes.append(k)
            if self.left[k] != LEAF:
                stack += [self.left[k], self.right[k]]
        return np.array(nodes)

    @property
    def n_leaves(self):
        """Number of leaves that can be reached from the root."""
        return int(np.sum(self.left[self._reachable()] == LEAF))

    @property
    def depth(self):
        """Depth of the deepest reachable node. The root has depth 0."""
        depth = {0: 0}
        for k in sorted(self._reachable()):
            if self.left[k] != LEAF:
                depth[self.left[k]] = depth[self.right[k]] = depth[k] + 1
        return max(depth.values())

    def prune(self, alpha):
        """
        Prune the tree with minimal cost-complexity pruning for one value of alpha.

        The cost of a tree is its weighted training error plus alpha for every leaf.
        The error of a node is the weight of its instances outside the largest class, divided by the total weight at the root.
        One pass goes from the deepest nodes to the root. A subtree becomes a leaf if the leaf costs no more than the subtree.
        The result is the smallest subtree with the lowest cost.

        Args:
            alpha: Cost of one leaf as a share of the total weight at the root. With 0, only subtrees that add no error are removed.

        Returns:
            A new, pruned tree. The original tree is unchanged.
        """
        root_w = self.value[0].sum()
        error = (self.value.sum(axis=1) - self.value.max(axis=1)) / root_w
        cost = np.empty(len(self.left))
        left, right = self.left.copy(), self.right.copy()
        for k in range(len(left) - 1, -1, -1):
            as_leaf = error[k] + alpha
            if left[k] == LEAF:
                cost[k] = as_leaf
            elif as_leaf <= cost[left[k]] + cost[right[k]]:
                cost[k] = as_leaf
                left[k] = right[k] = LEAF
            else:
                cost[k] = cost[left[k]] + cost[right[k]]

        pruned = object.__new__(ClassificationTree)
        pruned.__dict__.update(self.__dict__)
        pruned.left, pruned.right = left, right
        return pruned

    def describe(self, encoder, class_names, max_depth=2):
        """
        Write the top of the tree as indented text.

        Args:
            encoder: The TreeEncoder that made the codes. It turns a test into words.
            class_names: Name of every class.
            max_depth: Number of levels to show.

        Returns:
            Text with one line per test or leaf. Each line gives the number of instances in the node.
        """
        lines = []

        def visit(k, d, prefix):
            """
            Add the lines of node k and of its children to the output.

            Args:
                k: Index of the node.
                d: Depth of the node.
                prefix: Indentation of the node.
            """
            dist = self.value[k]
            label = class_names[int(dist.argmax())]
            if self.left[k] == LEAF or d == max_depth:
                lines.append(f"{prefix}-> {label}  (n={self.count[k]:,})")
                return
            test = encoder.describe_test(self.feature[k], self.code[k])
            lines.append(f"{prefix}if {test}  (n={self.count[k]:,})")
            visit(self.left[k], d + 1, prefix + "    ")
            lines.append(f"{prefix}else")
            visit(self.right[k], d + 1, prefix + "    ")

        visit(0, 0, "")
        return "\n".join(lines)
