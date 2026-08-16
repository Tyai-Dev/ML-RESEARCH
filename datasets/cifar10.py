r"""CIFAR-10 — the first color dataset, via the Hugging Face mirror.

60,000 32x32 RGB images (50k train / 10k test), 10 object classes
(Krizhevsky/Nair/Hinton). The canonical toronto.edu tarball crawls
from here, so we pull the uoft-cs/cifar10 parquet files from the HF
CDN instead (~144MB total), decode the PNGs once, and cache as npz in
datasets/cifar10/ (gitignored).

    from cifar10 import load_cifar10, CIFAR10_CLASSES
    X_train, y_train, X_test, y_test = load_cifar10()

X is (n, 3072) float in [0,1], channel-planar RGB: reshape to
(3, 32, 32) and transpose(1, 2, 0) for display.
"""

import io
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np

_DIR = Path(__file__).resolve().parent / "cifar10"
_BASE = ("https://huggingface.co/datasets/uoft-cs/cifar10/resolve/"
         "main/plain_text/")
_FILES = {"train": "train-00000-of-00001.parquet",
          "test": "test-00000-of-00001.parquet"}
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def _download(url: str, path: Path):
    ctx = ssl.create_default_context(cafile=certifi.where())
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, context=ctx) as r, \
            open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {path.name}: {done / 1e6:6.1f} MB", end="",
                  flush=True)
    print()
    tmp.rename(path)


def _parse(path: Path):
    """Parquet -> (X flat channel-planar [0,255] uint8, y). The 'img'
    column holds PNG bytes (HF image feature); decode with PIL."""
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(path)
    imgs = t.column("img").to_pylist()
    X = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])))
                  for d in imgs])                  # (n, 32, 32, 3)
    X = X.transpose(0, 3, 1, 2).reshape(len(X), -1)  # channel-planar
    y = np.asarray(t.column("label").to_pylist(), dtype=np.int64)
    return X.astype(np.uint8), y


def load_cifar10():
    cache = _DIR / "cifar10.npz"
    if not cache.exists():
        _DIR.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for split, fname in _FILES.items():
            path = _DIR / fname
            if not path.exists():
                print(f"downloading cifar10 {split} (HF mirror) ...")
                _download(_BASE + fname, path)
            print(f"decoding {split} PNGs ...")
            arrays[f"{split}_x"], arrays[f"{split}_y"] = _parse(path)
        np.savez_compressed(cache, **arrays)
    z = np.load(cache)
    return (z["train_x"].astype(np.float64) / 255.0,
            z["train_y"].astype(np.int64),
            z["test_x"].astype(np.float64) / 255.0,
            z["test_y"].astype(np.int64))


if __name__ == "__main__":
    X, y, Xt, yt = load_cifar10()
    print(f"train {X.shape}, test {Xt.shape}, classes {np.bincount(y)}")
