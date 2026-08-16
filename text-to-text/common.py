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
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DATA_PATH = os.path.join(_DATA_DIR, "tinyshakespeare.txt")

# The pantry (see data/prepare_datasets.py). Pick a corpus per run with
# load_everything(corpus=...) or the LLM_CORPUS environment variable;
# the default stays Tiny Shakespeare so existing scores stay comparable.
CORPORA = {
    "tinyshakespeare": "tinyshakespeare.txt",
    "tinystories": "tinystories.txt",              # pretrain: general English
    "chat": "soda_chat.txt",                       # finetune: User:/Bot: turns
    "instruct": "tinystories_instruct.txt",        # finetune: spec -> story
    "summarize": "tinystories_summarize.txt",      # finetune: story -> summary
}
DEFAULT_CORPUS = os.environ.get("LLM_CORPUS", "tinyshakespeare")


def load_corpus(corpus: str = None) -> str:
    """The raw text of one corpus, downloading Tiny Shakespeare once if
    missing; the pantry corpora must be prepared first."""
    corpus = corpus or DEFAULT_CORPUS
    path = os.path.join(_DATA_DIR, CORPORA[corpus])
    if corpus == "tinyshakespeare" and not os.path.exists(path):
        os.makedirs(_DATA_DIR, exist_ok=True)
        print(f"downloading Tiny Shakespeare -> {path}")
        urllib.request.urlretrieve(_DATA_URL, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing - run llm/data/prepare_datasets.py first")
    with open(path, encoding="utf-8") as f:
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


def load_everything(corpus: str = None):
    """The standard setup every rung starts from:
    (train_ids, val_ids, chars, stoi, itos).

    For pantry corpora the vocabulary comes from data/alphabet.txt (the
    union over all prepared datasets), NOT from the corpus itself: a
    model pretrained on tinystories must already own an embedding for
    every character the chat/instruct finetunes will feed it later.
    Tiny Shakespeare keeps its self-derived 65-char vocab so existing
    checkpoints and scores are untouched."""
    corpus = corpus or DEFAULT_CORPUS
    text = load_corpus(corpus)
    alphabet_path = os.path.join(_DATA_DIR, "alphabet.txt")
    if corpus != "tinyshakespeare" and os.path.exists(alphabet_path):
        with open(alphabet_path, encoding="ascii") as f:
            chars, stoi, itos = make_vocab(f.read())
    else:
        chars, stoi, itos = make_vocab(text)
    ids = encode(text, stoi)
    train_ids, val_ids = train_val_split(ids)
    return train_ids, val_ids, chars, stoi, itos
