"""Deep learning primitives: batch norm, 1D convolution, activations
(sigmoid, ReLU), a two-layer MLP forward/backward pass, attention, beam
search."""

from typing import List

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


def sigmoid(x):
    """Sigmoid (logistic) activation: squashes any real number into (0, 1).

        sigmoid(x) = 1 / (1 + e^-x)

    Model: historically used as the activation in early neural nets, and
    still used today wherever you want an output interpretable as a
    probability or a "gate" (e.g. binary classification output, LSTM/GRU
    gates). It's monotonic, smooth, and symmetric around x=0 -> 0.5.

    Numerical stability: computing e^-x directly overflows for very
    negative x (e.g. x = -1000 makes e^-x astronomically large) and, less
    obviously, computing e^x directly overflows for very positive x. The
    standard fix is to pick whichever of the two equivalent forms keeps
    the exponent's argument negative:
        for x >= 0:  1 / (1 + e^-x)      (e^-x is small and safe)
        for x <  0:  e^x / (1 + e^x)     (e^x is small and safe; this is
                                           algebraically the same function,
                                           just multiplying top and bottom
                                           of the first form by e^x)

    Known weakness (why ReLU below is often preferred in deep nets): the
    derivative sigmoid(x)*(1-sigmoid(x)) maxes out at 0.25 (at x=0) and
    shrinks toward 0 as |x| grows -- so in deep stacks of sigmoid layers,
    gradients shrink multiplicatively at every layer during backprop
    ("vanishing gradients"), slowing or stalling learning in early layers.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))     # safe: exponent is <= 0

    neg = ~pos
    exp_x = np.exp(x[neg])                       # safe: exponent is < 0
    out[neg] = exp_x / (1.0 + exp_x)

    return out


def sigmoid_backward(dout, sigmoid_out):
    """Backward pass for sigmoid, given dL/d(sigmoid_out) and the cached
    forward output (not the raw input -- the derivative is naturally
    expressed in terms of the output itself, which is why forward passes
    typically cache their output for later reuse here):

        d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))

    Quick derivation: let s = 1/(1+e^-x). Using the quotient rule,
    ds/dx = e^-x / (1+e^-x)^2 = [1/(1+e^-x)] * [e^-x/(1+e^-x)]
           = s * (1 - s)          (since 1 - s = e^-x / (1+e^-x)).
    """
    return dout * sigmoid_out * (1.0 - sigmoid_out)


def relu(x):
    """ReLU (Rectified Linear Unit): the default activation in most modern
    deep nets.

        relu(x) = max(0, x)

    Model: pass positive values through unchanged, zero out negative
    values. It's the identity function on the positive side, which is
    exactly why it largely avoids the vanishing-gradient problem sigmoid
    has -- the gradient there is a constant 1, not something that shrinks
    with |x|. It's also far cheaper to compute (one comparison) than an
    exponential.

    Known weakness ("dying ReLU"): if a unit's input is negative, its
    gradient is exactly 0, so that unit gets zero learning signal and its
    weights never update on that example -- if this happens persistently
    across training, the unit can become permanently inactive. (Variants
    like Leaky ReLU address this by letting a small gradient leak through
    for x < 0; not implemented here.)
    """
    return np.maximum(0.0, x)


def relu_backward(dout, x):
    """Backward pass for ReLU, given dL/d(relu(x)) and the cached forward
    *input* x (unlike sigmoid, ReLU's derivative depends on which side of
    0 the original input was, not on the output value):

        d(relu(x))/dx = 1  if x > 0
                       = 0  if x < 0
                       = undefined at x = 0 (a measure-zero case in
                         practice; frameworks conventionally treat it as 0,
                         which is what the strict `x > 0` mask below does)

    So the gradient either passes through unchanged (x > 0) or gets
    zeroed out entirely (x <= 0) -- there's no "shrinking," just an
    on/off gate, which is what makes ReLU cheap to backprop through too.
    """
    return dout * (x > 0)


class TwoLayerMLP:
    """Two-layer MLP forward and backward pass, backprop derived and coded
    by hand (no autodiff): x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2)
    -> y_pred, trained against Mean Squared Error loss.

    This is the smallest network that still needs the full machinery of
    backprop: two layers of chain rule, a nonlinearity to differentiate
    through, and a shared computation graph where the hidden layer's
    output feeds every output unit (so its gradient has to sum
    contributions from all of them).

    Forward pass:
        z1_pre = W1 @ x + b1            # hidden pre-activation, shape (hidden,)
        z1     = relu(z1_pre)           # hidden activation, shape (hidden,)
        y_pred = W2 @ z1 + b2           # output, shape (output,)
        loss   = mean((y_pred - y_true)^2)

    Backward pass -- walk the forward pass in reverse, at each step
    computing "how much does the loss change if I nudge *this*
    quantity," using the chain rule to turn the previous step's answer
    into the next one:
        d_y_pred  = 2*(y_pred - y_true) / n     # dL/d(y_pred), n = len(y_true)
        dW2       = outer(d_y_pred, z1)         # dL/dW2
        db2       = d_y_pred                    # dL/db2
        d_z1      = W2.T @ d_y_pred              # dL/dz1 (error routed back through W2)
        d_z1_pre  = d_z1 * (z1_pre > 0)          # dL/d(z1_pre), ReLU backward
        dW1       = outer(d_z1_pre, x)           # dL/dW1
        db1       = d_z1_pre                     # dL/db1

    Why outer products for the weight gradients: for any linear layer
    `out = W @ in + b`, out[i] = sum_j W[i][j]*in[j] + b[i], so
    d(out[i])/d(W[i][j]) = in[j] -- meaning dL/dW[i][j] = d_out[i] * in[j]
    for *every* (i, j) pair simultaneously, which is exactly the
    definition of np.outer(d_out, in): a matrix whose [i][j] entry is
    d_out[i] * in[j].

    Why W2.T for the hidden-layer gradient: forward, W2 routes hidden
    units to outputs (hidden -> output). Backward, the same connections
    carry the error signal the opposite way (output -> hidden), which is
    exactly what transposing the weight matrix does: W2.T[j][i] =
    W2[i][j], so (W2.T @ d_y_pred)[j] sums, over every output i, how much
    of output i's error flowed in through the connection from hidden
    unit j -- i.e. every path by which nudging z1[j] could affect the
    loss, added together (multivariable chain rule: when one quantity
    influences several downstream outputs, its total gradient is the sum
    of its gradient through each of them).
    """

    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float],
    ) -> dict:
        x = np.array(x, dtype=float)
        W1, b1 = np.array(W1, dtype=float), np.array(b1, dtype=float)
        W2, b2 = np.array(W2, dtype=float), np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # --- Forward pass ---
        # Keep z1_pre (before ReLU) around separately from z1 (after
        # ReLU): the backward pass needs to know *where* the ReLU zeroed
        # things out, which only z1_pre's sign can tell you -- z1 alone
        # can't distinguish "input was exactly 0" from "input was very
        # negative," both of which look like 0 after the max().
        z1_pre = W1 @ x + b1          # hidden pre-activation, shape (hidden,)
        z1 = relu(z1_pre)             # ReLU activation, shape (hidden,)
        y_pred = W2 @ z1 + b2         # network output, shape (output,)

        n = y_true.shape[0]
        loss = np.mean((y_pred - y_true) ** 2)

        # --- Backward pass (mirror the forward pass, last op first) ---
        # d(mean((y_pred - y_true)^2))/d(y_pred[i]) = 2*(y_pred[i] - y_true[i])/n
        # -- each output only appears in its own term of the mean, so
        # every entry's gradient depends only on itself, no cross terms.
        d_y_pred = 2.0 * (y_pred - y_true) / n

        # Output layer (y_pred = W2 @ z1 + b2): weight gradient is the
        # outer product of this layer's upstream gradient and its input
        # (z1); bias gradient is just the upstream gradient itself, since
        # d(out)/d(b) = 1.
        dW2 = np.outer(d_y_pred, z1)
        db2 = d_y_pred.copy()

        # Route the error backward through W2 to get the gradient at the
        # hidden layer's *output* (post-ReLU) -- reusing this file's own
        # relu_backward to then push it through ReLU into the gradient at
        # the hidden layer's *pre-activation*.
        d_z1 = W2.T @ d_y_pred
        d_z1_pre = relu_backward(d_z1, z1_pre)

        # Hidden layer (z1_pre = W1 @ x + b1): identical outer-product
        # pattern to the output layer, just one layer deeper -- this is
        # the general shape of backprop through any stack of linear
        # layers, repeated once per layer.
        dW1 = np.outer(d_z1_pre, x)
        db1 = d_z1_pre.copy()

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }


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

    assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)         # symmetric midpoint
    assert np.allclose(sigmoid(np.array([1000.0, -1000.0])), [1.0, 0.0])  # no overflow
    s = sigmoid(np.array([-2.0, 0.0, 2.0]))
    assert np.all(np.diff(s) > 0)   # sigmoid is strictly increasing
    ds_numerical = (sigmoid(np.array([0.0]) + 1e-5) - sigmoid(np.array([0.0]) - 1e-5)) / (2e-5)
    assert np.isclose(sigmoid_backward(1.0, sigmoid(np.array([0.0])))[0], ds_numerical[0], atol=1e-4)
    print("sigmoid / sigmoid_backward OK", s)

    r_input = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    r = relu(r_input)
    assert np.array_equal(r, [0.0, 0.0, 0.0, 0.5, 2.0])
    dr = relu_backward(np.ones_like(r_input), r_input)
    assert np.array_equal(dr, [0.0, 0.0, 0.0, 1.0, 1.0])
    print("relu / relu_backward OK", r, dr)

    mlp = TwoLayerMLP()
    x_mlp = [1.0, -2.0, 0.5]
    W1_mlp = [[0.5, -1.0, 0.2], [0.3, 0.1, -0.4]]   # hidden=2, input=3
    b1_mlp = [0.1, -0.2]
    W2_mlp = [[1.0, -1.0], [0.5, 0.5]]              # output=2, hidden=2
    b2_mlp = [0.0, 0.1]
    y_true_mlp = [1.0, 0.5]

    result = mlp.forward_and_backward(x_mlp, W1_mlp, b1_mlp, W2_mlp, b2_mlp, y_true_mlp)
    assert set(result.keys()) == {"loss", "dW1", "db1", "dW2", "db2"}

    # Numerical gradient check: hold every parameter fixed except the one
    # array being tested, perturb each of its entries by +-eps, recompute
    # the (unrounded) loss, and estimate that entry's gradient via a
    # central difference. This must match the hand-derived analytic
    # gradients above to confirm the backward pass is actually correct,
    # not just plausible-looking -- the same technique used for
    # batch_norm_backward's dgamma check earlier in this file.
    W1_arr = np.array(W1_mlp, dtype=float)
    b1_arr = np.array(b1_mlp, dtype=float)
    W2_arr = np.array(W2_mlp, dtype=float)
    b2_arr = np.array(b2_mlp, dtype=float)
    x_arr = np.array(x_mlp, dtype=float)
    y_true_arr = np.array(y_true_mlp, dtype=float)

    def mlp_loss(W1, b1, W2, b2):
        z1 = relu(W1 @ x_arr + b1)
        y_pred = W2 @ z1 + b2
        return np.mean((y_pred - y_true_arr) ** 2)

    def numerical_gradient(param, loss_fn, eps=1e-5):
        grad = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + eps
            loss_plus = loss_fn()
            param[idx] = original - eps
            loss_minus = loss_fn()
            param[idx] = original   # restore before moving to the next entry
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        return grad

    numerical_dW1 = numerical_gradient(W1_arr, lambda: mlp_loss(W1_arr, b1_arr, W2_arr, b2_arr))
    numerical_db1 = numerical_gradient(b1_arr, lambda: mlp_loss(W1_arr, b1_arr, W2_arr, b2_arr))
    numerical_dW2 = numerical_gradient(W2_arr, lambda: mlp_loss(W1_arr, b1_arr, W2_arr, b2_arr))
    numerical_db2 = numerical_gradient(b2_arr, lambda: mlp_loss(W1_arr, b1_arr, W2_arr, b2_arr))

    assert np.allclose(result["dW1"], numerical_dW1, atol=1e-4)
    assert np.allclose(result["db1"], numerical_db1, atol=1e-4)
    assert np.allclose(result["dW2"], numerical_dW2, atol=1e-4)
    assert np.allclose(result["db2"], numerical_db2, atol=1e-4)
    print("TwoLayerMLP.forward_and_backward OK", result)

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
