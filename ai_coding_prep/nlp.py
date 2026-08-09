"""NLP: byte-pair encoding (BPE) tokenizer training."""

import re
from collections import Counter, defaultdict


def get_word_freqs(corpus):
    freqs = Counter()
    for line in corpus:
        for word in line.strip().split():
            freqs[" ".join(list(word)) + " </w>"] += 1
    return freqs


def get_pair_counts(word_freqs):
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair, word_freqs):
    pattern = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
    merged = "".join(pair)
    return {pattern.sub(merged, word): freq for word, freq in word_freqs.items()}


def train_bpe(corpus, num_merges):
    word_freqs = get_word_freqs(corpus)
    merges = []

    for _ in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)

    vocab = set()
    for word in word_freqs:
        vocab.update(word.split())
    return vocab, merges


def apply_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for pair in merges:
        i = 0
        while i < len(symbols) - 1:
            if (symbols[i], symbols[i + 1]) == pair:
                symbols[i:i + 2] = ["".join(pair)]
            else:
                i += 1
    return symbols


if __name__ == "__main__":
    corpus = ["low low low low low", "lower lower", "newest newest newest", "widest widest"]
    vocab, merges = train_bpe(corpus, num_merges=10)

    assert ("l", "o") in merges or ("lo", "w") in merges
    print("train_bpe OK", "merges:", merges)

    tokens = apply_bpe("lowest", merges)
    assert isinstance(tokens, list) and "".join(t.replace("</w>", "") for t in tokens) == "lowest"
    print("apply_bpe OK", tokens)
