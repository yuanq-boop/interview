"""NLP: byte-pair encoding (BPE) tokenizer training."""

import re
from collections import Counter, defaultdict


def get_word_freqs(corpus):
    """Turn raw text into a symbol-level representation BPE can operate on.

    Each word is split into individual characters, separated by spaces,
    with an end-of-word marker `</w>` appended (so BPE can learn that a
    merge like "e r </w>" only applies at word boundaries, distinguishing
    e.g. "er" at the end of "teacher" from "er" inside "erase"). Words are
    counted by frequency across the corpus since identical words get
    identical merge treatment -- no need to process duplicates separately.
    """
    freqs = Counter()
    for line in corpus:
        for word in line.strip().split():
            # "low" -> "l o w </w>"
            freqs[" ".join(list(word)) + " </w>"] += 1
    return freqs


def get_pair_counts(word_freqs):
    """Count how often every adjacent symbol pair occurs across the corpus,
    weighted by word frequency. E.g. if "l o w </w>" has frequency 5, the
    pairs ("l","o"), ("o","w"), ("w","</w>") each get +5.
    This identifies the merge candidate that would most reduce the total
    token count if applied everywhere.
    """
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair, word_freqs):
    """Apply one merge everywhere it occurs: replace every occurrence of
    the space-separated pair (e.g. "l o") with the fused symbol ("lo") in
    every word. The negative lookaround/lookahead (?<!\\S)...(?!\\S)
    ensures we only match the pair as whole adjacent tokens (bounded by
    spaces or string edges), not as a substring inside a longer symbol.
    """
    pattern = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
    merged = "".join(pair)
    return {pattern.sub(merged, word): freq for word, freq in word_freqs.items()}


def train_bpe(corpus, num_merges):
    """Byte-Pair Encoding tokenizer training (Sennrich et al., 2016).

    Motivation: word-level vocabularies can't handle unseen/rare words,
    and character-level vocabularies produce very long sequences. BPE is
    a middle ground: start from characters and greedily merge the most
    frequent adjacent pair into a new subword unit, repeated `num_merges`
    times. Frequent whole words end up as single merged tokens (e.g.
    "the"), while rare/unseen words fall back to smaller, still-meaningful
    subword pieces instead of an <unk> token.

    Algorithm:
      1. Represent the corpus as space-separated character sequences
         (get_word_freqs).
      2. Repeat num_merges times:
           a. Count all adjacent symbol-pair frequencies (get_pair_counts).
           b. Pick the single most frequent pair.
           c. Merge that pair everywhere it appears (merge_pair).
           d. Record the merge (order matters -- it's replayed later by
              apply_bpe on new text).
      3. Stop early if there are no pairs left to merge (fully merged).

    Returns the final vocabulary (set of all symbols in use) and the
    ordered list of merges learned.
    """
    word_freqs = get_word_freqs(corpus)
    merges = []

    for _ in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)   # greedy: most frequent pair wins
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)   # remember the merge, and the order it happened in

    vocab = set()
    for word in word_freqs:
        vocab.update(word.split())
    return vocab, merges


def apply_bpe(word, merges):
    """Tokenize a new word using previously learned merges.

    Start from individual characters (+ end-of-word marker), then replay
    the learned merges *in the order they were learned* (earlier merges
    were more frequent/foundational, so applying them first mirrors how
    the training process built up larger units from smaller ones).
    At each merge, scan left to right and fuse any adjacent pair that
    matches, shrinking the symbol list in place.
    """
    symbols = list(word) + ["</w>"]
    for pair in merges:
        i = 0
        while i < len(symbols) - 1:
            if (symbols[i], symbols[i + 1]) == pair:
                # Fuse the two symbols into one, shortening the list by 1.
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
