"""Deep learning primitives: batch norm, 1D convolution, attention, beam search."""

import numpy as np


def batch_norm_forward(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    cache = (x_hat, gamma, var, eps)
    return out, cache


def batch_norm_backward(dout, cache):
    x_hat, gamma, var, eps = cache
    n = dout.shape[0]
    std_inv = 1.0 / np.sqrt(var + eps)

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_hat = dout * gamma
    dx = (1.0 / n) * std_inv * (
        n * dx_hat
        - np.sum(dx_hat, axis=0)
        - x_hat * np.sum(dx_hat * x_hat, axis=0)
    )
    return dx, dgamma, dbeta


def conv1d(x, kernel, stride=1, padding=0):
    if padding > 0:
        x = np.pad(x, (padding, padding))

    k = len(kernel)
    out_len = (len(x) - k) // stride + 1
    out = np.empty(out_len)
    for i in range(out_len):
        start = i * stride
        out[i] = np.dot(x[start:start + k], kernel)
    return out


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V, weights


def beam_search(start_seq, score_fn, beam_width, max_len, eos_token):
    """score_fn(seq) -> iterable of (next_token, log_prob) candidates."""
    beams = [(start_seq, 0.0)]

    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq[-1] == eos_token:
                candidates.append((seq, score))
                continue
            for token, token_score in score_fn(seq):
                candidates.append((seq + [token], score + token_score))

        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = candidates[:beam_width]

        if all(seq[-1] == eos_token for seq, _ in beams):
            break

    return max(beams, key=lambda c: c[1])


if __name__ == "__main__":
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0], [1.0, -2.0]])
    gamma, beta = np.ones(2), np.zeros(2)
    out, cache = batch_norm_forward(x, gamma, beta)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(out.std(axis=0), 1, atol=1e-3)
    print("batch_norm_forward OK")

    dout = np.ones_like(x)
    dx, dgamma, dbeta = batch_norm_backward(dout, cache)
    assert dx.shape == x.shape and dgamma.shape == gamma.shape and dbeta.shape == beta.shape
    numerical_dgamma = np.array([
        (batch_norm_forward(x, gamma + np.eye(2)[i] * 1e-5, beta)[0] * dout).sum()
        - (batch_norm_forward(x, gamma - np.eye(2)[i] * 1e-5, beta)[0] * dout).sum()
        for i in range(2)
    ]) / (2 * 1e-5)
    assert np.allclose(dgamma, numerical_dgamma, atol=1e-3)
    print("batch_norm_backward OK")

    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    kernel = np.array([1.0, 0.0, -1.0])
    result = conv1d(signal, kernel)
    assert np.allclose(result, [-2.0, -2.0, -2.0])
    print("conv1d OK", result)

    Q = K = V = np.eye(3)
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(weights.sum(axis=-1), 1.0)
    print("scaled_dot_product_attention OK")

    # toy scorer: token "b" always scores higher than "a"; sequence ends at length 3
    vocab_scores = {"a": np.log(0.4), "b": np.log(0.6)}

    def score_fn(seq):
        if len(seq) >= 3:
            return [("<eos>", 0.0)]
        return list(vocab_scores.items())

    best_seq, best_score = beam_search(["<bos>"], score_fn, beam_width=2, max_len=5, eos_token="<eos>")
    assert best_seq == ["<bos>", "b", "b", "<eos>"]
    print("beam_search OK", best_seq, best_score)
