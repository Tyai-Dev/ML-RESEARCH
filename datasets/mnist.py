r"""MNIST — the repo's first real dataset, loaded from the raw bytes.

70,000 handwritten digits (60k train / 10k test), 28x28 grayscale,
collected by NIST from US Census Bureau employees and high-school
students in the 1990s. We download the four original IDX files once
(from the S3 mirror the torch ecosystem uses), parse them directly
with numpy — the format is a 4-byte magic number, the dimensions, then
raw bytes — and cache as .npz. No torchvision, no framework: reading
the actual bytes is part of the point.

Files land in datasets/mnist/ (gitignored, ~12MB).

    from mnist import load_mnist, load_binary
    X_train, y_train, X_test, y_test = load_mnist()        # all digits
    X_train, y_train, X_test, y_test = load_binary(3, 5)   # one pair,
                                                           # labels 0/1

Pixels are scaled to [0, 1]; images are flattened to 784-vectors
(no intercept column — experiments add their own).
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent / "mnist"
_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def _parse_idx(raw: bytes) -> np.ndarray:
    """The IDX format: magic (2 zero bytes, a dtype byte, a rank
    byte), then rank big-endian uint32 dimensions, then the data."""
    zero, dtype, rank = raw[0:2], raw[2], raw[3]
    assert zero == b"\x00\x00" and dtype == 0x08, "not a ubyte IDX file"
    dims = struct.unpack(f">{rank}I", raw[4:4 + 4 * rank])
    return np.frombuffer(raw, dtype=np.uint8,
                         offset=4 + 4 * rank).reshape(dims)


def load_mnist():
    """(X_train (60000, 784) in [0,1], y_train, X_test, y_test),
    downloading and caching on first call."""
    cache = _DIR / "mnist.npz"
    if not cache.exists():
        _DIR.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for key, fname in _FILES.items():
            path = _DIR / fname
            if not path.exists():
                print(f"downloading {fname} ...")
                urllib.request.urlretrieve(_URL + fname, path)
            arrays[key] = _parse_idx(gzip.decompress(path.read_bytes()))
        np.savez_compressed(cache, **arrays)
    z = np.load(cache)
    X_train = z["train_x"].reshape(-1, 784).astype(np.float64) / 255.0
    X_test = z["test_x"].reshape(-1, 784).astype(np.float64) / 255.0
    return (X_train, z["train_y"].astype(np.int64),
            X_test, z["test_y"].astype(np.int64))


def load_binary(d0: int, d1: int):
    """The two-digit subset with labels 0 (= digit d0) / 1 (= d1)."""
    X_train, y_train, X_test, y_test = load_mnist()
    tr = (y_train == d0) | (y_train == d1)
    te = (y_test == d0) | (y_test == d1)
    return (X_train[tr], (y_train[tr] == d1).astype(np.float64),
            X_test[te], (y_test[te] == d1).astype(np.float64))


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"train {X_train.shape}, test {X_test.shape}, "
          f"labels {np.bincount(y_train)}")
