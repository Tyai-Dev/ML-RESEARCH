r"""The training engine — one loop, any model, any loss.

fit(model, loss_fn, ...) is deliberately generic: the model is any
nn.Module producing logits, the loss is any callable
loss_fn(logits, targets) -> scalar. Swapping either is a one-line
change in the experiment — that is the whole design, so that models
and losses can be raced against each other on identical data, batches,
and seeds.

What the loop does, and logs, per epoch:
  - minibatch SGD over a fresh shuffle (the schedule is seeded:
    identical runs are identical);
  - running train loss/accuracy printed at quarter-epoch marks, so
    you see progress INSIDE the epoch, not just after it;
  - at epoch end: full train loss, val loss, val accuracy, the epoch's
    wall time and throughput;
  - best-validation tracking with a '*' on record epochs; the best
    weights are RESTORED at the end (early stopping's gentle form —
    the last epoch is not necessarily the best one, a lesson measured
    the hard way in Generative/text-to-text/tokenization).

Returns a history dict (per-epoch curves) for plotting.
"""

import time

import numpy as np
import torch

SEED = 7


@torch.no_grad()
def evaluate(model, loss_fn, X, y, batch: int = 1024):
    """Average loss and accuracy over a full split, batched."""
    model.eval()
    total_loss, correct = 0.0, 0
    for i in range(0, len(X), batch):
        logits = model(X[i : i + batch])
        total_loss += loss_fn(logits, y[i : i + batch]).item() * len(logits)
        correct += int((logits.argmax(dim=1) == y[i : i + batch]).sum())
    model.train()
    return total_loss / len(X), correct / len(X)


def fit(
    model,
    loss_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    epochs: int = 10,
    batch: int = 128,
    lr: float = 1e-3,
    optimizer: str = "adam",
):
    """Train `model` under `loss_fn`; log everything; keep the best."""
    device = X_train.device
    opt = (
        torch.optim.Adam(model.parameters(), lr=lr)
        if optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=lr)
    )
    g = torch.Generator().manual_seed(SEED)
    n = len(X_train)
    n_batches = (n + batch - 1) // batch
    quarters = {round(q * n_batches) for q in (0.25, 0.5, 0.75)}

    from models import describe

    bar = "=" * 66
    print(bar)
    print(f" model     {describe(model)}")
    print(f" loss      {type(loss_fn).__name__}    optimizer " f"{optimizer} lr={lr:g}")
    print(
        f" data      {n:,} train / {len(X_val):,} val    "
        f"batch {batch}    device {device.type}"
    )
    print(bar)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=g).to(device)
        running_loss, running_correct, seen = 0.0, 0, 0
        for b in range(n_batches):
            idx = perm[b * batch : (b + 1) * batch]
            logits = model(X_train[idx])
            loss = loss_fn(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * len(idx)
            running_correct += int((logits.argmax(dim=1) == y_train[idx]).sum())
            seen += len(idx)
            if b + 1 in quarters:
                print(
                    f"   epoch {epoch:>2}  batch {b + 1:>4}/"
                    f"{n_batches}  running loss "
                    f"{running_loss / seen:.4f}  acc "
                    f"{running_correct / seen:.4f}"
                )
        dt = time.perf_counter() - t0
        train_loss, _ = evaluate(model, loss_fn, X_train, y_train)
        val_loss, val_acc = evaluate(model, loss_fn, X_val, y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        star = " "
        if val_acc > best_acc:
            best_acc, best_epoch, star = val_acc, epoch, "*"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:>2}/{epochs} | train loss "
            f"{train_loss:.4f} | val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f}{star}| {dt:4.1f}s "
            f"({seen / dt:,.0f} img/s)"
        )

    model.load_state_dict(best_state)
    print(bar)
    print(f" best epoch {best_epoch} (val acc {best_acc:.4f}) — " f"weights restored")
    print(bar)
    return history
