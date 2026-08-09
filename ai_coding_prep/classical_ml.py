"""Classical ML: k-means, perceptron, average precision, Jaccard similarity."""

import numpy as np


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

    ap = average_precision([1, 0, 1, 1, 0], [0.9, 0.8, 0.7, 0.4, 0.3])
    assert 0 <= ap <= 1
    print("average_precision OK", ap)

    assert jaccard_similarity({1, 2, 3}, {2, 3, 4}) == 0.5
    assert abs(weighted_jaccard_similarity([1, 2, 0], [1, 0, 3]) - (1 / 6)) < 1e-9
    print("jaccard_similarity OK")
