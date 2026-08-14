r"""Shared infrastructure for the LLM track.

One place for the corpus, the character vocabulary, the train/val split,
and the device — so every rung of the ladder (bigram -> ngram-mlp ->
attention -> gpt) trains on the SAME data and reports validation NLL /
perplexity on the SAME held-out slice. That shared measuring stick is
what makes "each rung must beat the last" a checkable claim rather than
a slogan.

Corpus: Tiny Shakespeare (~1.1MB of plays, public domain), modeled at
the character level. Auto-downloads to llm/data/ on first use.
"""

import os
import urllib.request

import numpy as np
import torch

SEED = 7
VAL_FRACTION = 0.1          # last 10% of the corpus is the val split

_DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
             "master/data/tinyshakespeare/input.txt")
_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "tinyshakespeare.txt")


def load_corpus() -> str:
    """The raw text, downloading it once if missing."""
    if not os.path.exists(_DATA_PATH):
        os.makedirs(os.path.dirname(_DATA_PATH), exist_ok=True)
        print(f"downloading Tiny Shakespeare -> {_DATA_PATH}")
        urllib.request.urlretrieve(_DATA_URL, _DATA_PATH)
    with open(_DATA_PATH, encoding="utf-8") as f:
        return f.read()


def make_vocab(text: str):
    """Character vocabulary: every distinct char, sorted for
    reproducibility. Returns (chars, stoi, itos) — string-to-index and
    index-to-string maps. Tiny Shakespeare has 65 characters."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return chars, stoi, itos


def encode(text: str, stoi: dict) -> np.ndarray:
    """Text -> int64 id array."""
    return np.array([stoi[c] for c in text], dtype=np.int64)


def decode(ids, itos: dict) -> str:
    """Id sequence -> text."""
    return "".join(itos[int(i)] for i in ids)


def train_val_split(ids: np.ndarray):
    """Deterministic split: first 90% train, last 10% val. Contiguous
    (not shuffled) so the val slice is genuinely unseen future text."""
    n_train = int(len(ids) * (1 - VAL_FRACTION))
    return ids[:n_train], ids[n_train:]


def get_device() -> torch.device:
    """The RTX 4070 when visible, CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_everything():
    """The standard setup every rung starts from:
    (train_ids, val_ids, chars, stoi, itos)."""
    text = load_corpus()
    chars, stoi, itos = make_vocab(text)
    ids = encode(text, stoi)
    train_ids, val_ids = train_val_split(ids)
    return train_ids, val_ids, chars, stoi, itos
