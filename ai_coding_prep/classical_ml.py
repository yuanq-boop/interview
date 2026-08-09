"""Classical ML: k-means, perceptron, average precision, Jaccard similarity."""

import numpy as np


def kmeans(X, k, max_iters=100, tol=1e-4, seed=None):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centroids = X[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        new_centroids = centroids.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members):
                new_centroids[j] = members.mean(axis=0)

        if np.linalg.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids, labels


class Perceptron:
    def __init__(self, n_features, lr=0.1):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr

    def predict(self, x):
        return 1 if np.dot(self.w, x) + self.b > 0 else -1

    def fit(self, X, y, epochs=100):
        for _ in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                if self.predict(xi) != yi:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1
            if errors == 0:
                break
        return self


def average_precision(y_true, y_scores):
    order = np.argsort(-np.asarray(y_scores))
    y_true = np.asarray(y_true)[order]

    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0

    tp_cum = np.cumsum(y_true)
    precision = tp_cum / (np.arange(len(y_true)) + 1)
    recall = tp_cum / total_positives
    recall_prev = np.concatenate(([0], recall[:-1]))

    return float(np.sum((recall - recall_prev) * precision))


def jaccard_similarity(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def weighted_jaccard_similarity(u, v):
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
