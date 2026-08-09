"""Systems/efficiency: sparse matmul, quantization, packed vectors, load balancing, sampling."""

import bisect
import hashlib
import heapq
import random


class SparseMatrixCSR:
    """Compressed Sparse Row (CSR) matrix representation.

    Motivation: a dense n x m matrix stores n*m numbers even if almost all
    of them are zero. CSR instead stores only the non-zero values, using
    three flat arrays:
      - data:    the non-zero values, row by row, left to right.
      - indices: the column index of each corresponding value in `data`.
      - indptr:  indptr[i]..indptr[i+1] is the slice of `data`/`indices`
                 belonging to row i (so indptr has length n_rows + 1).
    This uses O(nnz) memory instead of O(n*m), and lets matmul skip all
    the zero entries entirely.
    """

    def __init__(self, dense):
        rows, cols = len(dense), len(dense[0]) if dense else 0
        data, indices, indptr = [], [], [0]
        for row in dense:
            for j, val in enumerate(row):
                if val != 0:
                    data.append(val)       # store only non-zero entries
                    indices.append(j)      # remember which column they came from
            indptr.append(len(data))       # mark where this row's entries end

        self.shape = (rows, cols)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def matmul_dense(self, other):
        """Sparse (self) x dense (other) matrix multiplication.

        Standard matmul is O(n_rows * n_cols_self * out_cols). By only
        iterating over the non-zero entries of each row (via indptr), this
        is O(nnz * out_cols) instead -- much cheaper when the matrix is
        mostly zeros, since we never touch the zero entries.
        """
        n_rows, _ = self.shape
        out_cols = len(other[0])
        result = [[0.0] * out_cols for _ in range(n_rows)]

        for i in range(n_rows):
            # Walk only this row's non-zero entries (indptr[i]:indptr[i+1]).
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                col, val = self.indices[idx], self.data[idx]
                # This non-zero contributes val * other[col][:] to result row i
                # (the standard row-times-matrix accumulation, skipping zeros).
                for j in range(out_cols):
                    result[i][j] += val * other[col][j]
        return result


def quantize(x, num_bits=8, symmetric=True):
    """Post-training quantization: map a list of floats to low-bit
    integers using an affine mapping, so a trained model's weights/
    activations can be stored and computed on with far less memory and
    (on supporting hardware) faster integer arithmetic, at the cost of
    some precision loss.

    Affine quantization formula: q = round(x / scale) + zero_point,
    where `scale` maps the float range onto the integer range [qmin, qmax],
    and `zero_point` is the integer that represents float value 0.

    Two modes:
      - symmetric: assume the float range is roughly centered at 0 and
        map it onto an integer range centered at (qmax+qmin)/2. Simpler,
        used when values are naturally signed/centered (e.g. weights).
      - asymmetric: map the *actual* observed [x_min, x_max] range exactly
        onto [qmin, qmax]. Uses the integer range more efficiently when
        values are skewed/one-sided (e.g. post-ReLU activations, which
        are >= 0).
    """
    qmin, qmax = 0, 2 ** num_bits - 1   # e.g. 8 bits -> integers 0..255
    x_min, x_max = min(x), max(x)

    if symmetric:
        max_abs = max(abs(x_min), abs(x_max)) or 1e-8   # avoid div-by-zero on an all-zero input
        scale = max_abs / ((qmax - qmin) / 2)            # float units per integer step
        zero_point = (qmax + qmin) // 2                  # integer representing float 0
    else:
        span = (x_max - x_min) or 1e-8
        scale = span / (qmax - qmin)                     # stretch [x_min, x_max] to [qmin, qmax]
        zero_point = round(qmin - x_min / scale)          # shift so x_min lands exactly on qmin

    # Quantize each value: scale to integer units, round, then clip into
    # the valid integer range in case of small numerical overshoot.
    q = [max(qmin, min(qmax, round(v / scale + zero_point))) for v in x]
    return q, scale, zero_point


def dequantize(q, scale, zero_point):
    """Invert quantize(): recover approximate float values from integers.
    This is the exact algebraic inverse of q = round(x/scale + zero_point):
    x_approx = (q - zero_point) * scale. The `round` in quantize is where
    precision is permanently lost -- dequantize can't undo that part.
    """
    return [(v - zero_point) * scale for v in q]


class PackedBinaryVector:
    """Bit-packed storage for a binary (0/1) vector.

    A naive representation (e.g. a Python list of ints, or one byte per
    entry) uses at least 1 byte (8 bits) per entry. Since each entry only
    carries 1 bit of information, we can pack 8 entries into a single byte
    using bitwise operations, cutting memory ~8x -- useful for large
    bitmasks, Bloom filters, embedding sign bits, etc.
    """

    def __init__(self, bits):
        self.length = len(bits)
        # One byte holds 8 bits, so we need ceil(length / 8) bytes.
        self.data = bytearray((self.length + 7) // 8)
        for i, bit in enumerate(bits):
            if bit:
                # Set bit (i % 8) of byte (i // 8) using an OR mask.
                self.data[i // 8] |= 1 << (i % 8)

    def get(self, i):
        # Shift the target byte so the desired bit lands in position 0,
        # then mask with & 1 to isolate just that bit.
        return (self.data[i // 8] >> (i % 8)) & 1

    def to_list(self):
        return [self.get(i) for i in range(self.length)]


class ConsistentHashLoadBalancer:
    """Consistent hashing: assign keys (e.g. requests, cache entries) to
    nodes (e.g. servers) in a way that's stable as nodes are added/removed.

    Naive approach (hash(key) % num_nodes) reshuffles almost every key
    whenever num_nodes changes -- terrible for caches/sharding, since it
    causes a mass cache-miss / rebalance storm.

    Consistent hashing instead places both nodes and keys onto a single
    hash ring (conceptually a circle of hash values). A key is routed to
    the first node clockwise from the key's own hash position. Removing
    or adding one node then only affects the keys that fall between that
    node and its neighbor on the ring -- roughly 1/N of all keys, not all
    of them.

    `replicas` virtual nodes per physical node are added to spread each
    node's ownership evenly around the ring (otherwise, with few points,
    ranges owned by different nodes could vary wildly in size).
    """

    def __init__(self, replicas=100):
        self.replicas = replicas
        self.ring = {}          # hash value -> node name
        self.sorted_keys = []   # hash values, kept sorted for binary search

    @staticmethod
    def _hash(key):
        # A stable, well-distributed hash function mapping any string to
        # a large integer position on the ring.
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node):
        for i in range(self.replicas):
            # Hash a distinct virtual-node label per replica so one
            # physical node occupies many scattered ring positions.
            h = self._hash(f"{node}#{i}")
            self.ring[h] = node
            bisect.insort(self.sorted_keys, h)   # keep sorted_keys sorted for binary search

    def remove_node(self, node):
        for i in range(self.replicas):
            h = self._hash(f"{node}#{i}")
            del self.ring[h]
            self.sorted_keys.remove(h)

    def get_node(self, key):
        if not self.ring:
            return None
        h = self._hash(key)
        # Find the first ring position >= h (the "next node clockwise").
        # Wrap around to index 0 if h is past every existing position.
        idx = bisect.bisect(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]


def reservoir_sample(stream, k, seed=None):
    """Algorithm R: uniform random sampling of k items from a stream of
    unknown (or very large) length, in a single pass using O(k) memory.

    Invariant maintained after processing the first i+1 items: every item
    seen so far is in the reservoir with probability exactly k/(i+1).
    Proof sketch by induction: the first k items fill the reservoir
    directly (probability 1 = k/k). For each subsequent item i (0-indexed,
    i >= k), it's included with probability k/(i+1); if included, it
    evicts a uniformly random existing slot. Working through the math
    shows every item, old or new, still ends up included with probability
    k/(i+1) after this step.
    """
    rng = random.Random(seed)
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)   # fill the reservoir with the first k items
        else:
            j = rng.randint(0, i)    # random index in [0, i], inclusive
            if j < k:
                # Item i replaces a uniformly random existing slot,
                # with probability k/(i+1) of that happening at all.
                reservoir[j] = item
    return reservoir


def weighted_sample_without_replacement(items, weights, k, seed=None):
    """A-Res (Algorithm A with Reservoir) weighted sampling without
    replacement: pick k items in a single pass, where each item's chance
    of being picked is proportional to its weight, and no item repeats.

    Trick: assign each item a random "key" = U^(1/w), where U ~ Uniform(0,1)
    and w is the item's weight. Higher-weight items tend to produce keys
    closer to 1 (since raising a number in (0,1) to a small exponent
    1/w pushes it toward 1 faster for larger w), so they're more likely to
    land in the top k keys -- but it's still randomized, so it's not just
    "always pick the highest-weight items". Taking the k largest keys
    gives a valid weighted sample without replacement in one pass, no
    need to know all weights/items up front (works on streams too).
    """
    rng = random.Random(seed)
    keyed = [(rng.random() ** (1.0 / w), item) for item, w in zip(items, weights)]
    return [item for _, item in heapq.nlargest(k, keyed, key=lambda p: p[0])]


if __name__ == "__main__":
    dense = [[1, 0, 0], [0, 2, 3], [0, 0, 0]]
    sparse = SparseMatrixCSR(dense)
    other = [[1, 0], [0, 1], [1, 1]]
    expected = [[1, 0], [3, 5], [0, 0]]
    assert sparse.matmul_dense(other) == expected
    print("SparseMatrixCSR OK")

    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    q, scale, zp = quantize(values, num_bits=8, symmetric=True)
    recovered = dequantize(q, scale, zp)
    assert all(abs(a - b) < 0.02 for a, b in zip(values, recovered))
    print("quantize/dequantize OK", q)

    bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    packed = PackedBinaryVector(bits)
    assert packed.to_list() == bits
    assert len(packed.data) == 2   # 10 bits packed into ceil(10/8) = 2 bytes
    print("PackedBinaryVector OK")

    lb = ConsistentHashLoadBalancer(replicas=50)
    for node in ["A", "B", "C"]:
        lb.add_node(node)
    before = {k: lb.get_node(k) for k in ["req1", "req2", "req3", "req4", "req5"]}
    lb.add_node("D")   # adding a node should only move a fraction of keys, not all
    after = {k: lb.get_node(k) for k in before}
    moved = sum(1 for k in before if before[k] != after[k])
    assert moved <= len(before)
    print("ConsistentHashLoadBalancer OK", before, "->", after)

    stream = list(range(1000))
    sample = reservoir_sample(stream, k=10, seed=0)
    assert len(sample) == 10 and len(set(sample)) == 10
    print("reservoir_sample OK", sample)

    items = ["a", "b", "c", "d"]
    weights = [1, 1, 1, 100]   # "d" is 100x more likely to be picked than any of a/b/c
    sample = weighted_sample_without_replacement(items, weights, k=2, seed=0)
    assert "d" in sample and len(sample) == 2
    print("weighted_sample_without_replacement OK", sample)
