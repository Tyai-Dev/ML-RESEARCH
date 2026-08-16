r"""CIFAR-10 — the first color dataset, from the original pickles.

60,000 32x32 RGB images (50k train / 10k test), 10 object classes,
collected by Krizhevsky/Nair/Hinton. Downloaded once as the original
python tar (~170MB), parsed directly from the pickled batches, cached
as npz in datasets/cifar10/ (gitignored).

    from cifar10 import load_cifar10, CIFAR10_CLASSES
    X_train, y_train, X_test, y_test = load_cifar10()

X is (n, 3072) float in [0,1], channel-planar RGB: reshape to
(3, 32, 32) and transpose(1, 2, 0) for display.
"""

import pickle
import ssl
import tarfile
import urllib.request
from pathlib import Path

import certifi
import numpy as np

_DIR = Path(__file__).resolve().parent / "cifar10"
_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def load_cifar10():
    cache = _DIR / "cifar10.npz"
    if not cache.exists():
        _DIR.mkdir(parents=True, exist_ok=True)
        tar_path = _DIR / "cifar-10-python.tar.gz"
        if not tar_path.exists():
            print("downloading CIFAR-10 (~170MB) ...")
            ctx = ssl.create_default_context(cafile=certifi.where())
            tmp = tar_path.with_suffix(".part")
            with urllib.request.urlopen(_URL, context=ctx) as r, \
                    open(tmp, "wb") as f:
                done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    print(f"\r  {done / 1e6:6.1f} MB", end="",
                          flush=True)
            print()
            tmp.rename(tar_path)
        Xs, ys, Xt, yt = [], [], None, None
        with tarfile.open(tar_path, "r:gz") as tar:
            for m in tar.getmembers():
                name = Path(m.name).name
                if name.startswith("data_batch") or name == "test_batch":
                    d = pickle.loads(tar.extractfile(m).read(),
                                     encoding="latin1")
                    if name == "test_batch":
                        Xt, yt = d["data"], np.array(d["labels"])
                    else:
                        Xs.append(d["data"])
                        ys.append(np.array(d["labels"]))
        np.savez_compressed(cache,
                            train_x=np.vstack(Xs), train_y=np.concatenate(ys),
                            test_x=Xt, test_y=yt)
    z = np.load(cache)
    return (z["train_x"].astype(np.float64) / 255.0,
            z["train_y"].astype(np.int64),
            z["test_x"].astype(np.float64) / 255.0,
            z["test_y"].astype(np.int64))


if __name__ == "__main__":
    X, y, Xt, yt = load_cifar10()
    print(f"train {X.shape}, test {Xt.shape}, "
          f"classes {np.bincount(y)}")
