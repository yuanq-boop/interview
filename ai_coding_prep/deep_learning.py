"""Deep learning primitives: batch norm, 1D convolution, attention, beam search."""

import numpy as np


def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """Batch Normalization (forward pass), as in Ioffe & Szegedy 2015.

    Model: for a mini-batch x of shape (N, D), normalize each feature
    (column) independently to zero mean / unit variance across the batch
    dimension, then apply a learnable per-feature affine transform
    (gamma * x_hat + beta) so the network can still represent any scale
    and shift if that's optimal -- normalization alone would otherwise
    constrain every layer's output distribution.

    Why: keeping layer inputs at a stable distribution during training
    reduces internal covariate shift, letting you use higher learning
    rates and making training less sensitive to initialization.

    Steps:
      1. mean, var: per-feature statistics computed across the batch (axis 0).
      2. x_hat: standardize -> (x - mean) / sqrt(var + eps). eps avoids
         division by zero when a feature has ~0 variance in the batch.
      3. out: scale and shift by learnable gamma/beta.
    """
    mean = x.mean(axis=0)          # per-feature batch mean, shape (D,)
    var = x.var(axis=0)            # per-feature batch variance, shape (D,)
    x_hat = (x - mean) / np.sqrt(var + eps)   # standardized activations
    out = gamma * x_hat + beta     # learnable affine re-scale/shift
    cache = (x_hat, gamma, var, eps)  # stash what backward needs
    return out, cache


def batch_norm_backward(dout, cache):
    """Batch Normalization backward pass: given dL/d(out), compute
    dL/dx, dL/dgamma, dL/dbeta via the chain rule.

    dgamma / dbeta are simple sums because out = gamma * x_hat + beta is
    an elementwise affine map, applied independently per batch row.

    dx is the tricky part because mean and var (used to build x_hat) are
    themselves functions of every x_i in the batch -- so each x_i affects
    every x_hat_j through the shared mean/var, not just its own x_hat_i.
    The closed-form below is the standard simplified derivation (see the
    original paper's appendix) that avoids computing dmean/dvar separately:

        dx_i = (1/N) * std_inv * ( N*dx_hat_i
                                    - sum_j(dx_hat_j)
                                    - x_hat_i * sum_j(dx_hat_j * x_hat_j) )

    where dx_hat = dout * gamma (gradient w.r.t. the standardized values).
    """
    x_hat, gamma, var, eps = cache
    n = dout.shape[0]
    std_inv = 1.0 / np.sqrt(var + eps)

    dgamma = np.sum(dout * x_hat, axis=0)   # d(out)/d(gamma) = x_hat, summed over batch
    dbeta = np.sum(dout, axis=0)            # d(out)/d(beta) = 1, summed over batch

    dx_hat = dout * gamma                   # backprop through the affine step
    dx = (1.0 / n) * std_inv * (
        n * dx_hat
        - np.sum(dx_hat, axis=0)                      # correction from d(mean)/dx
        - x_hat * np.sum(dx_hat * x_hat, axis=0)       # correction from d(var)/dx
    )
    return dx, dgamma, dbeta


def conv1d(x, kernel, stride=1, padding=0):
    """1D convolution (cross-correlation, as used in practice by ML
    frameworks -- the kernel is not flipped).

    Model: slide a length-k kernel over the (optionally zero-padded) input
    signal; at each position, take the dot product of the kernel with the
    overlapping window of the signal. This is the core operation behind
    Conv1d layers (e.g. for audio/sequence data).

    Output length formula: floor((len(x) - k) / stride) + 1, after padding.
    """
    if padding > 0:
        x = np.pad(x, (padding, padding))   # zero-pad both ends before sliding

    k = len(kernel)
    out_len = (len(x) - k) // stride + 1
    out = np.empty(out_len)
    for i in range(out_len):
        start = i * stride
        # Dot product of kernel with the current window -> one output value.
        out[i] = np.dot(x[start:start + k], kernel)
    return out


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention, the core of the Transformer
    (Vaswani et al., 2017): "Attention is All You Need".

    Model: for each query, compute a similarity score against every key
    (dot product), turn those scores into a probability distribution via
    softmax, then output a weighted sum of the values using that
    distribution. Intuitively: "how much should this query attend to each
    key/value pair?"

    Q: (n_q, d_k), K: (n_k, d_k), V: (n_k, d_v).
    Output: (n_q, d_v) -- one weighted-value vector per query.

    Why scale by sqrt(d_k): raw dot products grow in magnitude with d_k,
    which pushes softmax into regions with tiny gradients (near
    one-hot). Dividing by sqrt(d_k) keeps the scores' variance ~constant
    regardless of dimension, keeping softmax well-behaved.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)     # (n_q, n_k) raw similarity scores, scaled
    if mask is not None:
        # Positions where mask is False are forced to -inf so softmax
        # assigns them ~0 probability (e.g. causal masking, padding masking).
        scores = np.where(mask, scores, -np.inf)

    # Numerically-stable softmax: subtract the row max before exponentiating.
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)   # normalize each query's row to sum to 1
    return weights @ V, weights   # weighted combination of the value vectors


def beam_search(start_seq, score_fn, beam_width, max_len, eos_token):
    """Beam search decoding: an approximate search for the highest-scoring
    output sequence from an autoregressive model (used e.g. for decoding
    translations or generated text), trading optimality for tractability.

    Greedy decoding (beam_width=1) can get stuck in a locally-good but
    globally-poor sequence because it commits to one token at a time.
    Beam search instead tracks the `beam_width` best partial sequences
    (by cumulative log-probability) at every step, expanding all of them
    and keeping only the overall best `beam_width` continuations -- a
    middle ground between greedy decoding and exhaustive search.

    score_fn(seq) -> iterable of (next_token, log_prob) candidates for
    what could follow `seq`.

    Steps per iteration:
      1. For every current beam, if it already ended (last token == eos),
         carry it forward unchanged (don't keep extending finished beams).
         Otherwise, expand it with every possible next token and add each
         token's log-prob to the sequence's running score.
      2. Sort all candidates (old finished beams + new expansions) by
         cumulative score, keep only the top `beam_width`.
      3. Stop early once every surviving beam has terminated in eos.
    Return the single highest-scoring beam found.
    """
    beams = [(start_seq, 0.0)]   # list of (sequence, cumulative_log_prob)

    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq[-1] == eos_token:
                # Already finished: keep it as-is so it can still compete
                # with beams that are still growing.
                candidates.append((seq, score))
                continue
            for token, token_score in score_fn(seq):
                # Extend: new sequence, score accumulates additively
                # because log P(seq) = sum of log P(token | prefix).
                candidates.append((seq + [token], score + token_score))

        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = candidates[:beam_width]   # prune back down to the beam width

        if all(seq[-1] == eos_token for seq, _ in beams):
            break   # nothing left to grow

    return max(beams, key=lambda c: c[1])   # best-scoring completed/partial beam


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
    # Numerical gradient check for dgamma via central differences, to
    # verify the closed-form backward formula against a brute-force estimate.
    numerical_dgamma = np.array([
        (batch_norm_forward(x, gamma + np.eye(2)[i] * 1e-5, beta)[0] * dout).sum()
        - (batch_norm_forward(x, gamma - np.eye(2)[i] * 1e-5, beta)[0] * dout).sum()
        for i in range(2)
    ]) / (2 * 1e-5)
    assert np.allclose(dgamma, numerical_dgamma, atol=1e-3)
    print("batch_norm_backward OK")

    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    kernel = np.array([1.0, 0.0, -1.0])   # a simple edge-detector kernel
    result = conv1d(signal, kernel)
    assert np.allclose(result, [-2.0, -2.0, -2.0])
    print("conv1d OK", result)

    Q = K = V = np.eye(3)   # identity Q/K/V -> each query attends to itself
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(weights.sum(axis=-1), 1.0)   # each row is a valid probability distribution
    print("scaled_dot_product_attention OK")

    # Toy scorer: token "b" always scores higher than "a"; force <eos> once
    # the sequence reaches length 3, so beam search should learn to prefer
    # picking "b" repeatedly before terminating.
    vocab_scores = {"a": np.log(0.4), "b": np.log(0.6)}

    def score_fn(seq):
        if len(seq) >= 3:
            return [("<eos>", 0.0)]
        return list(vocab_scores.items())

    best_seq, best_score = beam_search(["<bos>"], score_fn, beam_width=2, max_len=5, eos_token="<eos>")
    assert best_seq == ["<bos>", "b", "b", "<eos>"]
    print("beam_search OK", best_seq, best_score)
