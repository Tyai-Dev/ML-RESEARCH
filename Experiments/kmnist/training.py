r"""The training engine — one loop, any model, any loss, your log style.

fit(model, loss_fn, ...) is generic: the model is any nn.Module
producing logits, the loss any callable loss_fn(logits, targets).
The `log_style` argument selects how the loop reports:

  "delta"  one line per epoch with an ASCII progress bar, metric
           deltas vs the previous epoch, a *best marker, and ETA
  "card"   a small block per epoch: losses, accuracy, lr, grad norm,
           throughput, best-so-far
  "tqdm"   a live per-batch tqdm bar with loss/acc postfix, then a
           one-line epoch summary
  "quiet"  prints only on validation improvement (silent epochs
           collapse to dots) plus a final summary

All styles share the same mechanics: seeded shuffles, per-epoch
validation, best-checkpoint restore at the end.
"""

import sys
import time

import numpy as np
import torch

SEED = 7


@torch.no_grad()
def evaluate(model, loss_fn, X, y, batch: int = 1024):
    model.eval()
    total, correct = 0.0, 0
    for i in range(0, len(X), batch):
        logits = model(X[i:i + batch])
        total += loss_fn(logits, y[i:i + batch]).item() * len(logits)
        correct += int((logits.argmax(dim=1) == y[i:i + batch]).sum())
    model.train()
    return total / len(X), correct / len(X)


def fit(model, loss_fn, X_train, y_train, X_val, y_val, *,
        epochs=10, batch=128, lr=1e-3, log_style="delta"):
    from models import describe
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEED)
    n = len(X_train)
    n_batches = (n + batch - 1) // batch

    if log_style != "quiet":
        print(f"{describe(model)} | {type(loss_fn).__name__} | "
              f"adam lr={lr:g} | {n:,} train / {len(X_val):,} val | "
              f"batch size:{batch}, num:{n_batches:,} | "
              f"{X_train.device.type}")
    tty = sys.stdout.isatty()          # live bar only in a real terminal

    bar = None
    if log_style == "tqdm":
        from tqdm import tqdm as _tqdm

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_state, best_epoch = -1.0, None, 0
    prev = {}
    t_start = time.perf_counter()
    quiet_dots = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=g).to(X_train.device)
        run_loss, run_corr, seen, gnorm = 0.0, 0, 0, 0.0
        it = range(n_batches)
        if log_style == "tqdm":
            bar = _tqdm(it, desc=f"epoch {epoch}/{epochs}", ncols=78,
                        unit="b", leave=False)
            it = bar
        for b in it:
            idx = perm[b * batch:(b + 1) * batch]
            logits = model(X_train[idx])
            loss = loss_fn(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            gnorm = float(sum((p.grad ** 2).sum()
                              for p in model.parameters()) ** 0.5)
            opt.step()
            run_loss += loss.item() * len(idx)
            run_corr += int((logits.argmax(1) == y_train[idx]).sum())
            seen += len(idx)
            if log_style == "tqdm" and b % 20 == 0:
                bar.set_postfix(loss=f"{run_loss / seen:.3f}",
                                acc=f"{run_corr / seen:.3f}")
            elif log_style == "delta" and tty \
                    and (b % 20 == 0 or b == n_batches - 1):
                # the bar tracks THIS epoch's batches, live
                fill = round(22 * (b + 1) / n_batches)
                print(f"\repoch {epoch:>2}/{epochs} "
                      f"[{'=' * fill}>{'-' * (22 - fill)}] "
                      f"{b + 1:>5,}/{n_batches:,} | "
                      f"loss {run_loss / seen:.3f} | "
                      f"acc {run_corr / seen:6.2%}",
                      end="", flush=True)
        if bar is not None:
            bar.close()

        dt = time.perf_counter() - t0
        tr_loss = run_loss / seen
        val_loss, val_acc = evaluate(model, loss_fn, X_val, y_val)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        is_best = val_acc > best_acc
        if is_best:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}

        eta = (time.perf_counter() - t_start) / epoch * (epochs - epoch)
        d = lambda k, v: v - prev[k] if k in prev else 0.0
        if log_style == "delta":
            if tty:
                print("\r" + " " * 78 + "\r", end="")   # clear live bar
            print(f"epoch {epoch:>2}/{epochs} [{'=' * 22}>] "
                  f"train {tr_loss:.3f} ({d('t', tr_loss):+.3f}) | "
                  f"val {val_loss:.3f} ({d('v', val_loss):+.3f}) | "
                  f"acc {val_acc:6.2%} ({d('a', val_acc):+.2%})"
                  f"{' *best' if is_best else '      '} | "
                  f"{dt:4.1f}s | eta {eta:3.0f}s")
        elif log_style == "card":
            print(f"-- epoch {epoch}/{epochs} " + "-" * 28)
            print(f"   train loss {tr_loss:.4f}   val loss "
                  f"{val_loss:.4f}")
            print(f"   val acc    {val_acc:.2%}   best "
                  f"{best_acc:.2%} @{best_epoch}")
            print(f"   lr {lr:.1e}   |grad| {gnorm:.3f}   "
                  f"{dt:.1f}s   {seen / dt:,.0f} img/s")
        elif log_style == "tqdm":
            print(f"epoch {epoch}/{epochs} done: train {tr_loss:.3f}, "
                  f"val {val_loss:.3f}, acc {val_acc:.2%}"
                  f"{' *' if is_best else ''}")
        elif log_style == "quiet":
            if is_best:
                if quiet_dots:
                    print("." * quiet_dots)
                    quiet_dots = 0
                print(f"epoch {epoch:>2}: val acc {val_acc:.2%}  "
                      f"(new best)")
            else:
                quiet_dots += 1
        prev = {"t": tr_loss, "v": val_loss, "a": val_acc}

    if quiet_dots:
        print("." * quiet_dots)
    model.load_state_dict(best_state)
    total = time.perf_counter() - t_start
    print(f"best: val acc {best_acc:.2%} @ epoch {best_epoch} "
          f"({epochs} run, {total:.1f}s) - weights restored")
    return history
