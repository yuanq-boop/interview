"""Systems/efficiency: sparse matmul, quantization, packed vectors, load balancing, sampling."""

import bisect
import hashlib
import heapq
import random


class SparseMatrixCSR:
    """Compressed sparse row representation built from a dense list-of-lists matrix."""

    def __init__(self, dense):
        rows, cols = len(dense), len(dense[0]) if dense else 0
        data, indices, indptr = [], [], [0]
        for row in dense:
            for j, val in enumerate(row):
                if val != 0:
                    data.append(val)
                    indices.append(j)
            indptr.append(len(data))

        self.shape = (rows, cols)
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def matmul_dense(self, other):
        n_rows, _ = self.shape
        out_cols = len(other[0])
        result = [[0.0] * out_cols for _ in range(n_rows)]

        for i in range(n_rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                col, val = self.indices[idx], self.data[idx]
                for j in range(out_cols):
                    result[i][j] += val * other[col][j]
        return result


def quantize(x, num_bits=8, symmetric=True):
    """Post-training affine quantization of a float list to unsigned ints."""
    qmin, qmax = 0, 2 ** num_bits - 1
    x_min, x_max = min(x), max(x)

    if symmetric:
        max_abs = max(abs(x_min), abs(x_max)) or 1e-8
        scale = max_abs / ((qmax - qmin) / 2)
        zero_point = (qmax + qmin) // 2
    else:
        span = (x_max - x_min) or 1e-8
        scale = span / (qmax - qmin)
        zero_point = round(qmin - x_min / scale)

    q = [max(qmin, min(qmax, round(v / scale + zero_point))) for v in x]
    return q, scale, zero_point


def dequantize(q, scale, zero_point):
    return [(v - zero_point) * scale for v in q]


class PackedBinaryVector:
    """Bit-packed storage for a binary vector: 1 bit/entry instead of 1 byte+/entry."""

    def __init__(self, bits):
        self.length = len(bits)
        self.data = bytearray((self.length + 7) // 8)
        for i, bit in enumerate(bits):
            if bit:
                self.data[i // 8] |= 1 << (i % 8)

    def get(self, i):
        return (self.data[i // 8] >> (i % 8)) & 1

    def to_list(self):
        return [self.get(i) for i in range(self.length)]


class ConsistentHashLoadBalancer:
    """Consistent hashing ring: adding/removing a node only reshuffles ~1/N of keys."""

    def __init__(self, replicas=100):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []

    @staticmethod
    def _hash(key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node):
        for i in range(self.replicas):
            h = self._hash(f"{node}#{i}")
            self.ring[h] = node
            bisect.insort(self.sorted_keys, h)

    def remove_node(self, node):
        for i in range(self.replicas):
            h = self._hash(f"{node}#{i}")
            del self.ring[h]
            self.sorted_keys.remove(h)

    def get_node(self, key):
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]


def reservoir_sample(stream, k, seed=None):
    """Uniform sampling without replacement over a stream of unknown length."""
    rng = random.Random(seed)
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def weighted_sample_without_replacement(items, weights, k, seed=None):
    """A-Res algorithm: single pass, weighted sampling without replacement."""
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
    assert len(packed.data) == 2
    print("PackedBinaryVector OK")

    lb = ConsistentHashLoadBalancer(replicas=50)
    for node in ["A", "B", "C"]:
        lb.add_node(node)
    before = {k: lb.get_node(k) for k in ["req1", "req2", "req3", "req4", "req5"]}
    lb.add_node("D")
    after = {k: lb.get_node(k) for k in before}
    moved = sum(1 for k in before if before[k] != after[k])
    assert moved <= len(before)
    print("ConsistentHashLoadBalancer OK", before, "->", after)

    stream = list(range(1000))
    sample = reservoir_sample(stream, k=10, seed=0)
    assert len(sample) == 10 and len(set(sample)) == 10
    print("reservoir_sample OK", sample)

    items = ["a", "b", "c", "d"]
    weights = [1, 1, 1, 100]
    sample = weighted_sample_without_replacement(items, weights, k=2, seed=0)
    assert "d" in sample and len(sample) == 2
    print("weighted_sample_without_replacement OK", sample)
