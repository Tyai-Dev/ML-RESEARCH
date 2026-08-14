r"""Byte-Pair Encoding, by hand — characters are the floor, not the ceiling.

Everything on the ladder modeled text one CHARACTER at a time. That
choice was never examined; this file examines it.

The idea. A tokenizer is a bijection between text and sequences over a
vocabulary we get to DESIGN. Characters give a tiny vocabulary (V=65)
but long sequences — the model spends its context window and its
capacity re-learning that 't-h-e' is a unit. BPE builds a better
vocabulary greedily, from data:

    start: vocabulary = the characters
    repeat: find the most frequent ADJACENT PAIR of tokens,
            merge it into one new token, add it to the vocabulary

Each merge is chosen to maximize the immediate drop in corpus length —
greedy compression (each occurrence of the pair shrinks the corpus by
one token). After ~450 merges on Tiny Shakespeare the vocabulary holds
units like 'th', 'the ', 'and ', 'ing ', 'you' — English structure,
discovered by counting, nobody labeled a thing (the embedding-space
lesson of rung 2, one level down the stack).

Pre-tokenization: like GPT-2, we first split the text into chunks (a
word with its preceding whitespace) and merge only INSIDE chunks —
merges never cross word boundaries, which keeps tokens interpretable
and the algorithm fast (merge over unique chunks weighted by count,
not over the raw 1M-char stream).

THE MEASURING-STICK TRAP (the reason this file matters for the
ladder): perplexity PER TOKEN is meaningless across tokenizers — a
model over bigger tokens takes fewer, harder guesses. The exchange
rate is the compression ratio:

    NLL per char = NLL per token * (tokens / chars).

All cross-tokenizer comparisons in this repo are made in NLL/char
(equivalently bits/char); bpe_practical_pytorch.py uses exactly this
to compare a BPE-GPT against the ladder's char-GPT.

Verified here: encode/decode roundtrips the corpus EXACTLY (a
tokenizer bug is a silent data corruption); compression is monotone in
vocabulary size; the merge list is deterministic given the corpus.
The animation: token boundaries dissolving on a fixed passage as the
vocabulary grows.

Run me with F5. Discussion: tokenization.tex.
"""

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_corpus

N_MERGES = 450                       # final vocab ≈ 65 + 450


def pretokenize(text: str):
    """Chunks = a word plus its preceding whitespace (GPT-2 style).
    Merges happen only inside chunks. The second alternative catches
    trailing whitespace at end of text — a tokenizer must account for
    EVERY character or the roundtrip check below fails."""
    return re.findall(r"\s*\S+|\s+\Z", text)


def train_bpe(text: str, n_merges: int):
    """The algorithm from the docstring, run over unique chunks
    weighted by frequency. Returns the ordered merge list and the
    compression history."""
    chunk_counts = Counter(pretokenize(text))
    # each unique chunk as a tuple of single-char tokens
    chunks = {c: tuple(c) for c in chunk_counts}

    n_chars = sum(len(c) * n for c, n in chunk_counts.items())
    merges, history = [], []
    for step in range(n_merges):
        pair_counts = Counter()
        for c, toks in chunks.items():
            n = chunk_counts[c]
            for pair in zip(toks, toks[1:]):
                pair_counts[pair] += n
        if not pair_counts:
            break
        pair = max(pair_counts, key=pair_counts.get)   # most frequent
        merges.append(pair)
        merged = pair[0] + pair[1]
        for c, toks in chunks.items():                 # apply the merge
            out, i = [], 0
            while i < len(toks):
                if (i < len(toks) - 1 and toks[i] == pair[0]
                        and toks[i + 1] == pair[1]):
                    out.append(merged)
                    i += 2
                else:
                    out.append(toks[i])
                    i += 1
            chunks[c] = tuple(out)
        n_tokens = sum(len(t) * chunk_counts[c]
                       for c, t in chunks.items())
        history.append(dict(vocab=65 + step + 1,
                            chars_per_token=n_chars / n_tokens))
    return merges, history


def encode_chunk(chunk: str, ranks: dict):
    """Apply merges to one chunk, best (earliest-learned) first —
    the standard BPE encoding order."""
    toks = list(chunk)
    while len(toks) > 1:
        pairs = list(zip(toks, toks[1:]))
        best = min(pairs, key=lambda p: ranks.get(p, float("inf")))
        if best not in ranks:
            break
        out, i = [], 0
        while i < len(toks):
            if (i < len(toks) - 1
                    and (toks[i], toks[i + 1]) == best):
                out.append(toks[i] + toks[i + 1])
                i += 2
            else:
                out.append(toks[i])
                i += 1
        toks = out
    return toks


def encode(text: str, merges, _cache={}):
    """Text -> token strings, memoized per chunk."""
    ranks = {pair: i for i, pair in enumerate(merges)}
    out = []
    for chunk in pretokenize(text):
        if chunk not in _cache:
            _cache[chunk] = encode_chunk(chunk, ranks)
        out.extend(_cache[chunk])
    return out


if __name__ == "__main__":
    text = load_corpus()
    print(f"training BPE, {N_MERGES} merges over "
          f"{len(text):,} chars ...")
    merges, history = train_bpe(text, N_MERGES)

    print("\nfirst 20 merges learned (in order):")
    for i, (a, b) in enumerate(merges[:20]):
        print(f"  {i + 1:>3}. {a!r} + {b!r} -> {(a + b)!r}")

    # roundtrip: the tokenizer must be lossless
    tokens = encode(text, merges)
    assert "".join(tokens) == text, "roundtrip failed — data corruption!"
    ratio = len(text) / len(tokens)
    print(f"\nroundtrip exact over the full corpus: OK")
    print(f"tokens: {len(tokens):,}   chars/token: {ratio:.3f} "
          f"(a 128-token context now spans ≈ {128 * ratio:.0f} chars)")

    # compression must be monotone in vocabulary size
    cpt = [h["chars_per_token"] for h in history]
    assert all(b >= a for a, b in zip(cpt, cpt[1:])), \
        "compression should be monotone!"

    # longest learned tokens — English, discovered by counting
    vocab = sorted({t for t in tokens}, key=len, reverse=True)
    print("longest tokens:", ", ".join(repr(v) for v in vocab[:8]))

    # ------------------------------------------------------------------
    # Figure + animation: boundaries dissolving as the vocab grows
    # ------------------------------------------------------------------
    passage = text[957:1085]                     # a famous bit of text
    stages = [0, 5, 15, 40, 80, 150, 250, 350, N_MERGES]
    staged = [encode(passage, merges[:s], {}) for s in stages]

    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fig.suptitle("BPE: the vocabulary learning English units",
                 fontsize=11)

    ax1.plot([h["vocab"] for h in history], cpt, color="#2a78d6", lw=2)
    dot, = ax1.plot([], [], "o", color="#eb6834", ms=8)
    ax1.set(xlabel="vocabulary size", ylabel="chars per token",
            title="compression vs vocabulary")
    ax1.grid(alpha=.3)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    ax2.set_axis_off()
    ax2.set_title("token boundaries on a fixed passage", fontsize=10)
    head = ax2.text(0, .98, "", transform=ax2.transAxes, fontsize=9,
                    va="top", color="#eb6834", family="monospace")
    body = ax2.text(0, .88, "", transform=ax2.transAxes, fontsize=9,
                    va="top", family="monospace", wrap=True)

    def render(toks):
        return "|".join(t.replace("\n", "¶") for t in toks)

    def update(k):
        s = stages[k]
        v = 65 + s
        i = min(max(s - 1, 0), len(history) - 1)
        dot.set_data([65 + s if s else 65],
                     [cpt[i] if s else 1.0])
        head.set_text(f"vocab {v:>4}   merges {s:>3}   "
                      f"tokens {len(staged[k]):>3}")
        body.set_text(render(staged[k]))
        return dot, head, body

    ani = FuncAnimation(fig, update, frames=len(stages), interval=900,
                        blit=False, repeat=True)
    plt.show()
