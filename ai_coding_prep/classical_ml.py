"""Classical ML: k-means, perceptron, linear regression (batch gradient
descent), binary-classifier evaluation metrics (average precision,
ROC/AUC, confusion-matrix metrics, Jaccard similarity), and
information-theoretic / loss functions (entropy, cross-entropy, KL and
JS divergence, binary cross-entropy)."""

import numpy as np
from numpy.typing import NDArray


def kmeans(X, k, max_iters=100, tol=1e-4, seed=None):
    """K-means clustering (Lloyd's algorithm).

    Goal: partition n points into k clusters so that each point is assigned
    to the cluster whose centroid (mean) is closest to it, minimizing total
    within-cluster squared distance (inertia).

    Algorithm:
      1. Initialize k centroids by picking k random points from the data.
      2. Assignment step: assign every point to its nearest centroid
         (by Euclidean distance) -> this partitions the data into k clusters.
      3. Update step: recompute each centroid as the mean of the points
         currently assigned to it.
      4. Repeat steps 2-3 until centroids stop moving (converge) or a max
         iteration count is hit. This is a coordinate-descent-style
         procedure: it's guaranteed to decrease (or keep) inertia every
         iteration, so it converges, but only to a local optimum (result
         depends on initialization).

    Returns the final centroids (k x d) and the cluster label (0..k-1) for
    every input point.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # Step 1: initialize centroids to k distinct randomly chosen data points.
    centroids = X[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        # Step 2 (assignment): pairwise distance from every point to every
        # centroid via broadcasting -> shape (n, k), then take the argmin
        # along the centroid axis to get each point's nearest cluster.
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Step 3 (update): new centroid = mean of its assigned points.
        # If a cluster ends up with zero members (can happen with bad
        # initialization), keep its previous centroid instead of producing
        # a NaN from an empty mean.
        new_centroids = centroids.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members):
                new_centroids[j] = members.mean(axis=0)

        # Convergence check: stop early once centroids barely move.
        if np.linalg.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids, labels


class Perceptron:
    """Single-layer perceptron (binary linear classifier, labels in {-1, +1}).

    Model: predicts sign(w . x + b) -- a hyperplane splitting the input
    space into two half-spaces.

    Training rule (the classic Perceptron Learning Algorithm): for each
    misclassified example (xi, yi), nudge the weight vector toward the
    correct side of the boundary:
        w <- w + lr * yi * xi
        b <- b + lr * yi
    Intuition: if yi = +1 but the prediction was wrong (score <= 0), adding
    yi*xi = xi increases w.xi, pushing the score up. If yi = -1, we subtract
    xi, pushing the score down. Repeat over the dataset until an epoch with
    zero mistakes (this is guaranteed to terminate if the data is linearly
    separable; otherwise it runs until `epochs` is exhausted).
    """

    def __init__(self, n_features, lr=0.1):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr

    def predict(self, x):
        # Linear score, thresholded at 0 -> the decision boundary is the
        # hyperplane {x : w.x + b = 0}.
        return 1 if np.dot(self.w, x) + self.b > 0 else -1

    def fit(self, X, y, epochs=100):
        for _ in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                # Online update: only touch weights on a mistake.
                if self.predict(xi) != yi:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1
            # Early stop once a full pass makes no mistakes (converged).
            if errors == 0:
                break
        return self


class Solution:
    """Linear regression trained by batch gradient descent on MSE loss.
    (Class/method names kept as `Solution`/`get_derivative`/
    `get_model_prediction`/`train_model` to match the interview-style
    interface this was adapted from.)

    Model: y_hat = X @ w  (no implicit intercept -- include a constant
    column of 1s in X yourself if you want a bias term).

    Loss (Mean Squared Error): L(w) = (1/N) * sum_i (y_i - y_hat_i)^2

    Gradient: differentiate L w.r.t. a single weight w_j. Since
    y_hat_i = sum_k X_ik * w_k, we get d(y_hat_i)/d(w_j) = X_ij, so by the
    chain rule:
        dL/dw_j = (1/N) * sum_i 2*(y_i - y_hat_i) * (-X_ij)
                = -(2/N) * X[:, j] . (y - y_hat)
    Stacking that over every weight at once gives the full gradient
    vector in one matrix expression:
        grad = -(2/N) * X.T @ (y - y_hat)

    Training is batch gradient descent: repeatedly step opposite the
    gradient (the direction of steepest loss *increase*), scaled by
    `learning_rate`:
        w <- w - learning_rate * grad
    "Batch" means every step uses the *entire* dataset (as opposed to a
    single example or a random mini-batch, as in true SGD) -- so each
    step is an exact gradient step on the true MSE loss, at the cost of
    O(N) work per step.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """y_hat = X @ w, flattened to a 1D array of per-example predictions.

        `reshape(-1)` is used instead of the original `np.squeeze`
        deliberately: squeeze drops *every* size-1 axis, which silently
        collapses the batch axis too when N == 1 (a (1, 1) prediction
        would become a 0-d scalar instead of a length-1 vector).
        reshape(-1) always flattens to 1D regardless of N, matching what
        every caller here expects.
        """
        predictions = X @ weights
        return predictions.reshape(-1)

    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
        desired_weight: int,
    ) -> float:
        """dL/d(weights[desired_weight]) for the MSE loss above.
        (N is just len(X); kept as an explicit parameter so this method
        matches the required interface rather than recomputing it.)
        """
        return -2.0 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Run `num_iterations` steps of batch gradient descent starting
        from `initial_weights`, and return the final weights rounded to 5
        decimal places.

        Bug fixes vs. the original: the original aliased
        `weights = initial_weights` and updated it with in-place
        `weights[i] -= ...`, which (a) silently mutates the caller's
        array as a side effect, and (b) raises a numpy casting error if
        `initial_weights` has an integer dtype (in-place subtraction
        can't write a float result into an int array under numpy's
        "same_kind" casting rule). Copying into a fresh float64 array up
        front avoids both.
        """
        weights = np.array(initial_weights, dtype=np.float64, copy=True)
        N = X.shape[0]

        for _ in range(num_iterations):
            y_pred = self.get_model_prediction(X, weights)

            # Mathematically identical to calling get_derivative(...) for
            # every weight index j and collecting the results into an
            # array (get_derivative is kept above so each per-weight
            # gradient can still be inspected/tested individually), but
            # computed here as a single matrix-vector product instead of
            # `d` separate Python-level dot products -- one BLAS call
            # instead of d of them.
            gradient = -2.0 * (X.T @ (Y - y_pred)) / N

            weights -= self.learning_rate * gradient

        return np.round(weights, 5)


def average_precision(y_true, y_scores):
    """Average Precision (AP): the area under the precision-recall curve
    for a binary classifier, computed exactly (not via interpolation).

    Algorithm:
      1. Sort all examples by predicted score, descending. This simulates
         sweeping the decision threshold from "accept nothing" down to
         "accept everything".
      2. Walk down the sorted list keeping a running count of true
         positives seen so far (tp_cum). At each position i (1-indexed):
           precision_i = tp_cum_i / i                (of the top-i predictions, how many are correct)
           recall_i    = tp_cum_i / total_positives   (of all positives, how many have we found)
      3. AP is the weighted sum of precision values, where the weight at
         each step is the increase in recall at that step:
           AP = sum_i (recall_i - recall_{i-1}) * precision_i
         Recall only increases at positions where the example is an actual
         positive, so this sum is exactly the step-function area under the
         precision-recall curve (equivalent to the sklearn definition).
    """
    # Sort scores descending (argsort of the negation gives descending order).
    order = np.argsort(-np.asarray(y_scores))
    y_true = np.asarray(y_true)[order]

    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0

    tp_cum = np.cumsum(y_true)                        # running count of correct positives
    precision = tp_cum / (np.arange(len(y_true)) + 1)  # precision@i for every prefix length i
    recall = tp_cum / total_positives                  # recall@i for every prefix length i
    recall_prev = np.concatenate(([0], recall[:-1]))   # recall@(i-1), shifted by one

    # Riemann-sum the precision-recall curve using recall deltas as weights.
    return float(np.sum((recall - recall_prev) * precision))


def roc_curve(y_true, y_scores):
    """Receiver Operating Characteristic (ROC) curve.

    Sweeps a decision threshold over all scores (from "accept everything"
    down to "accept nothing") and, at each distinct threshold, plots:
        TPR (True Positive Rate / recall) = TP / (TP + FN) = TP / P
        FPR (False Positive Rate)         = FP / (FP + TN) = FP / N
    against each other. TPR is "of all actual positives, how many did we
    catch"; FPR is "of all actual negatives, how many did we wrongly flag".
    A perfect classifier hugs the top-left corner (TPR=1, FPR=0); a random
    classifier traces the diagonal TPR=FPR.

    Algorithm: sort by score descending (same setup as average_precision).
    Walking down the sorted list, every positive example increases TP by
    1 and every negative increases FP by 1, so cumulative sums directly
    give TP/FP counts for "threshold = this example's score". We keep only
    one point per *distinct* score -- ties must move together, since no
    threshold value can separate two examples with equal scores -- and
    prepend (0, 0) for the "threshold = +infinity, reject everything"
    endpoint.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_scores))
    y_true_sorted = y_true[order]
    scores_sorted = np.asarray(y_scores, dtype=float)[order]

    total_positives = y_true_sorted.sum()
    total_negatives = len(y_true_sorted) - total_positives
    if total_positives == 0 or total_negatives == 0:
        raise ValueError("roc_curve needs at least one positive and one negative example")

    tps = np.cumsum(y_true_sorted)         # running true-positive count
    fps = np.cumsum(1 - y_true_sorted)     # running false-positive count

    # Keep only the last index of each run of tied scores (a threshold can
    # only be drawn *between* distinct score values, not inside a tie).
    distinct = np.where(np.diff(scores_sorted, append=np.nan) != 0)[0]

    tpr = tps[distinct] / total_positives
    fpr = fps[distinct] / total_negatives
    thresholds = scores_sorted[distinct]

    # Prepend the (0, 0) point for the "reject everything" threshold.
    fpr = np.concatenate(([0.0], fpr))
    tpr = np.concatenate(([0.0], tpr))
    return fpr, tpr, thresholds


def auc_roc(y_true, y_scores):
    """Area Under the ROC Curve (AUC), via the trapezoidal rule over the
    curve produced by roc_curve(). Ranges from 0.5 (no better than random
    guessing) to 1.0 (perfect ranking of positives above negatives); below
    0.5 means the scores are anti-correlated with the labels (flipping the
    classifier's sign would beat it).
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # fpr is non-decreasing by construction (sorted by threshold), so this
    # is a valid left-to-right trapezoidal integration of tpr d(fpr): each
    # segment's area is its width (change in fpr) times the average of
    # its two endpoint heights (tpr values).
    widths = np.diff(fpr)
    avg_heights = (tpr[1:] + tpr[:-1]) / 2
    return float(np.sum(widths * avg_heights))


def auc_roc_rank_based(y_true, y_scores):
    """Alternate O(n log n) computation of AUC using its probabilistic
    definition: AUC = P(score of a random positive > score of a random
    negative), with ties counted as half a "win". This is the (normalized)
    Mann-Whitney U statistic, and gives the same value as integrating the
    ROC curve -- useful as a cross-check, and a common alternate way this
    metric is asked for in interviews ("compute AUC without building a
    curve").

    Algorithm: rank all n = P + N scores together, giving tied scores the
    *average* rank of the positions they span (required for an unbiased
    result under ties). Sum the ranks landing on positive examples. If
    positives always ranked below negatives, that sum would be minimized
    at 1+2+...+P = P*(P+1)/2; subtracting that baseline and normalizing by
    the total number of positive-negative pairs (P*N) gives exactly the
    fraction of pairs where the positive outranks the negative.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(y_scores, dtype=float)
    total_positives = y_true.sum()
    total_negatives = len(y_true) - total_positives
    if total_positives == 0 or total_negatives == 0:
        raise ValueError("auc_roc_rank_based needs at least one positive and one negative example")

    # Average ranks (1-indexed): walk the ascending-sorted scores in runs
    # of equal value, assigning every member of a run the mean rank of
    # the positions that run occupies.
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # +1 because ranks are 1-indexed
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    rank_sum_positive = ranks[y_true == 1].sum()
    return float((rank_sum_positive - total_positives * (total_positives + 1) / 2) /
                 (total_positives * total_negatives))


def confusion_matrix_counts(y_true, y_pred):
    """The four counts (TP/FP/TN/FN) that every point-metric below
    (accuracy, precision, recall, F1) is built from, for binary
    true/predicted labels.
        TP: predicted positive, actually positive
        FP: predicted positive, actually negative  (a "false alarm")
        TN: predicted negative, actually negative
        FN: predicted negative, actually positive  (a "miss")
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tp": int(np.sum(y_true & y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
    }


def classification_metrics(y_true, y_pred):
    """Standard point-metrics for a classifier at a single, already-fixed
    threshold (i.e. after scores have been turned into hard 0/1
    predictions), all derived from the confusion matrix:
        accuracy  = (TP+TN) / total        -- overall fraction correct
        precision = TP / (TP+FP)           -- of predicted positives, how many are right
        recall    = TP / (TP+FN)           -- of actual positives, how many did we find
        f1        = 2 * P * R / (P + R)    -- harmonic mean of precision & recall
    Precision and recall trade off against each other as the threshold
    moves (that's exactly what average_precision/roc_curve sweep over);
    F1 collapses that trade-off into one number at one specific threshold,
    and the harmonic mean (vs. a plain average) punishes having one of the
    two be very low even if the other is high.
    """
    counts = confusion_matrix_counts(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, **counts}


def entropy(p, base=np.e):
    """Shannon entropy: H(p) = -sum_i p_i * log(p_i).

    Measures the average "surprise"/uncertainty of a distribution `p`:
    it's the theoretical lower bound on the average number of nats (or
    bits, if base=2) needed to encode samples drawn from `p` using an
    optimal code built specifically for `p`. A distribution concentrated
    on one outcome has entropy 0 (no surprise, you already know the
    answer); the uniform distribution over n outcomes has the *maximum*
    possible entropy (log(n)) -- maximal uncertainty.

    Convention: 0 * log(0) is treated as 0 (its limit as p_i -> 0+),
    rather than the log(0) = -inf that would otherwise appear -- an
    outcome that never happens contributes no uncertainty.
    """
    p = np.asarray(p, dtype=float)
    mask = p > 0   # skip zero-probability terms per the 0*log(0)=0 convention
    h_nats = -np.sum(p[mask] * np.log(p[mask]))
    return float(h_nats / np.log(base))   # convert from nats to the requested base


def cross_entropy(p, q, base=np.e):
    """Cross-entropy: H(p, q) = -sum_i p_i * log(q_i).

    The average number of nats needed to encode samples from the *true*
    distribution `p`, using a code built for a (possibly wrong) model
    distribution `q`. This is exactly why it's used as a training loss:
    with `p` = the one-hot true label and `q` = the model's predicted
    distribution, H(p, q) is minimized (down to the irreducible
    entropy(p), which is 0 for a one-hot `p`) precisely when `q` matches
    `p` -- so minimizing cross-entropy pushes predictions toward the
    truth.

    Diverges to +inf if `q` assigns zero probability to any outcome `p`
    considers possible (log(0) = -inf, times a positive p_i) -- that's
    the correct math: a code built assuming that outcome is impossible
    needs infinitely many bits to encode it if it happens anyway. This
    is why practical training losses usually clip/smooth `q` away from
    exactly 0 (see binary_cross_entropy below).
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    h_nats = -np.sum(p[mask] * np.log(q[mask]))
    return float(h_nats / np.log(base))


def kl_divergence(p, q, base=np.e):
    """Kullback-Leibler divergence: D_KL(p || q) = sum_i p_i*log(p_i/q_i)
                                                  = cross_entropy(p, q) - entropy(p)

    The *extra* nats wasted by using a code built for `q` instead of the
    optimal code for `p`, beyond the unavoidable entropy(p) nats you'd
    need either way -- i.e. purely "the cost of q being wrong," with the
    inherent uncertainty of p subtracted out.

    Properties: D_KL(p||q) >= 0 always (Gibbs' inequality), with equality
    iff p == q exactly. It is NOT symmetric in general
    (D_KL(p||q) != D_KL(q||p)) -- it's a directional measure of "how bad
    is q as a stand-in for p," not a true distance/metric.
    """
    return cross_entropy(p, q, base) - entropy(p, base)


def js_divergence(p, q, base=np.e):
    """Jensen-Shannon divergence: a symmetrized, always-finite smoothing
    of KL divergence, built from the average of the two distributions
    m = (p + q) / 2:
        JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)

    Unlike raw KL, JSD is symmetric (JSD(p, q) == JSD(q, p) by
    construction) and bounded (0 <= JSD <= log(2) in nats, or <= 1 in
    bits), and it never diverges to infinity even when p and q have
    completely disjoint support -- because m = (p+q)/2 always has
    positive probability everywhere either p or q does, so neither
    KL(p||m) nor KL(q||m) can hit a log(0).
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)


def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy (a.k.a. log loss): the per-example
    cross-entropy between the true label -- viewed as the degenerate
    two-outcome distribution [1-y, y] -- and the model's predicted
    probability of the positive class, averaged over all examples:
        BCE = -mean_i( y_i*log(p_i) + (1-y_i)*log(1-p_i) )

    This is the standard training loss for logistic regression / binary
    classifiers: it's the *training-time* counterpart to the
    *evaluation-time* metrics above (average_precision, auc_roc,
    classification_metrics) -- those measure how good a trained
    classifier's rankings/decisions are, while this is what you'd
    actually differentiate and minimize to get there, since (unlike
    accuracy) it's smooth and has a useful gradient everywhere.

    Predictions are clipped away from exactly 0 or 1 before taking the
    log, since a genuinely wrong confident prediction (y=1, p=0 or
    y=0, p=1) would otherwise make this diverge to +inf -- a real risk
    in practice, unlike the pure-math entropy/cross_entropy/kl_divergence
    above, which are left free to diverge since they're meant to
    represent the exact theoretical quantities.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def jaccard_similarity(a, b):
    """Jaccard similarity for two sets: |intersection| / |union|.
    Ranges from 0 (disjoint) to 1 (identical sets). Two empty sets are
    treated as identical (similarity 1) by convention.
    """
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def weighted_jaccard_similarity(u, v):
    """Generalized (weighted) Jaccard similarity for two non-negative
    vectors, used e.g. for comparing bag-of-words / count vectors.
    It reduces to the set version when entries are all 0/1:
        sum(min(u_i, v_i)) / sum(max(u_i, v_i))
    min() plays the role of "intersection" (how much overlap exists per
    dimension) and max() plays the role of "union" (total extent per
    dimension).
    """
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    denominator = np.maximum(u, v).sum()
    return float(np.minimum(u, v).sum() / denominator) if denominator else 1.0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=(0, 0), scale=0.3, size=(30, 2))
    cluster_b = rng.normal(loc=(5, 5), scale=0.3, size=(30, 2))
    X = np.vstack([cluster_a, cluster_b])
    centroids, labels = kmeans(X, k=2, seed=0)
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1
    print("kmeans OK", centroids)

    X_lin = np.array([[2, 3], [1, 1], [-2, -3], [-1, -2]], dtype=float)
    y_lin = np.array([1, 1, -1, -1])
    clf = Perceptron(n_features=2).fit(X_lin, y_lin)
    assert all(clf.predict(x) == y for x, y in zip(X_lin, y_lin))
    print("perceptron OK")

    # Noiseless linear regression: with enough well-conditioned data and
    # iterations, batch gradient descent should recover the true weights
    # to (near-)exact precision.
    X_reg = rng.normal(size=(200, 2))
    true_weights = np.array([3.0, -2.0])
    Y_reg = X_reg @ true_weights

    sol = Solution(learning_rate=0.1)
    initial_weights = np.zeros(2)
    learned_weights = sol.train_model(X_reg, Y_reg, num_iterations=1000, initial_weights=initial_weights)
    assert np.allclose(learned_weights, true_weights, atol=1e-4)
    assert np.array_equal(initial_weights, [0.0, 0.0])  # caller's array must not be mutated
    print("Solution.train_model (linear regression) OK", learned_weights)

    # get_derivative (single-weight gradient) must agree with the
    # vectorized gradient train_model actually uses internally.
    y_pred = sol.get_model_prediction(X_reg, learned_weights)
    vectorized_grad = -2.0 * (X_reg.T @ (Y_reg - y_pred)) / X_reg.shape[0]
    for j in range(2):
        assert np.isclose(sol.get_derivative(y_pred, Y_reg, X_reg.shape[0], X_reg, j), vectorized_grad[j])
    print("Solution.get_derivative OK")

    # N=1 edge case: prediction must stay a 1D length-1 array, not
    # collapse into a 0-d scalar (the np.squeeze bug this replaced).
    pred_single = sol.get_model_prediction(np.array([[2.0, 3.0]]), np.array([1.0, 1.0]))
    assert pred_single.shape == (1,)
    print("Solution.get_model_prediction N=1 OK", pred_single)

    # Integer initial_weights must not crash (the original's in-place
    # `weights[i] -= float_grad` raised a numpy casting error here).
    int_weights = np.array([0, 0])
    sol.train_model(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([5.0, 11.0]), 5, int_weights)
    print("Solution.train_model integer-dtype input OK")

    ap = average_precision([1, 0, 1, 1, 0], [0.9, 0.8, 0.7, 0.4, 0.3])
    assert 0 <= ap <= 1
    print("average_precision OK", ap)

    # Perfectly separable case (both positive scores rank above both
    # negative scores) -> AUC should be exactly 1.0 under every method.
    y_true_perfect = [1, 1, 0, 0]
    y_scores_perfect = [0.9, 0.4, 0.35, 0.1]
    fpr, tpr, _ = roc_curve(y_true_perfect, y_scores_perfect)
    assert np.allclose(fpr, [0.0, 0.0, 0.0, 0.5, 1.0])
    assert np.allclose(tpr, [0.0, 0.5, 1.0, 1.0, 1.0])
    print("roc_curve OK")

    auc_trap = auc_roc(y_true_perfect, y_scores_perfect)
    auc_rank = auc_roc_rank_based(y_true_perfect, y_scores_perfect)
    assert abs(auc_trap - 1.0) < 1e-9
    # Both implementations of AUC must agree with each other on any input,
    # not just this perfect-separation case -- that's the whole point of
    # having a cross-check implementation.
    assert abs(auc_trap - auc_rank) < 1e-9
    print("auc_roc / auc_roc_rank_based OK", auc_trap)

    # A worse-than-random case: negatives score higher than positives, so
    # a random positive beats a random negative 0 times out of 4 -> AUC 0.
    y_true_inverted = [1, 1, 0, 0]
    y_scores_inverted = [0.1, 0.2, 0.8, 0.9]
    assert abs(auc_roc(y_true_inverted, y_scores_inverted) - 0.0) < 1e-9
    assert abs(auc_roc_rank_based(y_true_inverted, y_scores_inverted) - 0.0) < 1e-9
    print("auc_roc inverted-ranking OK")

    counts = confusion_matrix_counts([1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0])
    assert counts == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}
    metrics = classification_metrics([1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0])
    assert abs(metrics["accuracy"] - 4 / 6) < 1e-9
    assert abs(metrics["precision"] - 2 / 3) < 1e-9
    assert abs(metrics["recall"] - 2 / 3) < 1e-9
    assert abs(metrics["f1"] - 2 / 3) < 1e-9
    print("confusion_matrix_counts / classification_metrics OK", metrics)

    # A fair coin has exactly 1 bit of entropy; a certain outcome has 0.
    assert np.isclose(entropy([0.5, 0.5], base=2), 1.0)
    assert np.isclose(entropy([1.0, 0.0]), 0.0)
    print("entropy OK")

    # H(p, p) == H(p) exactly (cross-entropy of a distribution with
    # itself is just its own entropy), so KL(p||p) == 0.
    p_dist = np.array([0.7, 0.2, 0.1])
    assert np.isclose(cross_entropy(p_dist, p_dist), entropy(p_dist))
    assert np.isclose(kl_divergence(p_dist, p_dist), 0.0, atol=1e-9)
    print("cross_entropy / kl_divergence (self-comparison) OK")

    # Gibbs' inequality (KL >= 0) and asymmetry (KL(p||q) != KL(q||p))
    # for a genuinely different pair.
    q_dist = np.array([0.4, 0.5, 0.1])
    kl_pq = kl_divergence(p_dist, q_dist)
    kl_qp = kl_divergence(q_dist, p_dist)
    assert kl_pq > 0 and kl_qp > 0
    assert not np.isclose(kl_pq, kl_qp)
    print("kl_divergence (Gibbs' inequality + asymmetry) OK", kl_pq, kl_qp)

    # Cross-entropy must diverge to +inf when q rules out an outcome p
    # considers possible (log(0), correctly propagated, not a crash).
    with np.errstate(divide="ignore"):
        assert cross_entropy([0.5, 0.5], [1.0, 0.0]) == np.inf
    print("cross_entropy divergence OK")

    # JSD is symmetric by construction, and hits its known maximum
    # (log(2) nats) for two distributions with disjoint support.
    assert np.isclose(js_divergence(p_dist, q_dist), js_divergence(q_dist, p_dist))
    assert np.isclose(js_divergence([1.0, 0.0], [0.0, 1.0]), np.log(2))
    print("js_divergence (symmetry + bound) OK")

    # Confident-and-correct predictions should cost far less loss than
    # confident-and-wrong ones.
    bce_good = binary_cross_entropy([1, 0], [0.999, 0.001])
    bce_bad = binary_cross_entropy([1, 0], [0.001, 0.999])
    assert bce_good < 0.01 < 5.0 < bce_bad
    print("binary_cross_entropy OK", bce_good, bce_bad)

    assert jaccard_similarity({1, 2, 3}, {2, 3, 4}) == 0.5
    assert abs(weighted_jaccard_similarity([1, 2, 0], [1, 0, 3]) - (1 / 6)) < 1e-9
    print("jaccard_similarity OK")
