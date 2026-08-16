r"""Logistic regression by SGD — one sample per step, loop in plain sight.

The update. Draw one sample i, step down ITS gradient:

    w  <-  w - lr * (sigmoid(w . x_i) - y_i) x_i

Unbiased (averaging over i gives the full gradient), cheap (O(d) per
step, never touches the other n-1 samples), noisy (each step obeys one
sample's opinion). With a constant learning rate the iterate never
converges — it descends, then hovers in a NOISE BALL around the
optimum — so the returned estimator is the Polyak average of the last
epoch's iterates, which cancels the hovering.

Watch the training log for exactly that story: the train NLL falls
fast, then stops improving and just rattles; test accuracy locks in
long before the NLL settles (the threshold classifier needs only the
DIRECTION of w — logistic_animation.py shows it as a film).

Verification: the loop below and torch autograd driven through the
IDENTICAL sample schedule produce the same trajectory to machine
precision — the standard proof that the hand gradient is right.

Run me with F5. Derivations: logistic-regression.tex.
"""

import numpy as np
import torch

from common import SEED, evaluate, gradient, make_data, nll, predict, sigmoid

SGD_LR = 0.05
EPOCHS = 3
PRINT_EVERY = 100  # steps between log lines


def train(X, y, X_test, y_test, verbose=True, record=False):
    """The training loop. Returns (w_polyak, trajectory, order) —
    trajectory and visiting order only if record=True (used by the
    identity check and by logistic_animation.py's film)."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    w = np.zeros(X.shape[1])
    w_sum = np.zeros_like(w)  # Polyak: average the
    n_avg = 0  # last epoch's iterates
    traj, order = [], []
    step = 0
    if verbose:
        print(
            f"SGD: lr={SGD_LR}, {EPOCHS} epochs x {n:,} samples = "
            f"{EPOCHS * n:,} steps"
        )
        print(f"{'epoch':>6} {'step':>12} {'train NLL':>10} " f"{'test acc':>9}")
    for epoch in range(1, EPOCHS + 1):
        for i in rng.permutation(n):
            w -= SGD_LR * (sigmoid(X[i] @ w) - y[i]) * X[i]
            step += 1
            if record:
                traj.append(w.copy())
                order.append(i)
            if epoch == EPOCHS:  # final pass: accumulate
                w_sum += w
                n_avg += 1
            if verbose and step % PRINT_EVERY == 0:
                acc = float(np.mean(predict(w, X_test) == y_test))
                print(
                    f"{epoch:>4}/{EPOCHS} {step:>7,}/{EPOCHS * n:,} "
                    f"{nll(w, X, y):>10.4f} {acc:>9.4f}"
                )
    w_polyak = w_sum / n_avg
    if verbose:
        print(f"\nlast iterate  NLL {nll(w, X, y):.4f}   (noise ball)")
        print(f"Polyak (last epoch avg) NLL {nll(w_polyak, X, y):.4f}")
    return w_polyak, traj, order


def train_torch_mirror(X, y, n_steps):
    """The identical first n_steps via autograd (same schedule, same
    lr, float64) — must reproduce the hand loop exactly."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))[:n_steps]
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([w], lr=SGD_LR)
    traj = []
    for i in order:
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X_t[i] @ w, y_t[i])
        loss.backward()  # (sigmoid(w.x_i) - y_i) x_i
        opt.step()
        traj.append(w.detach().numpy().copy())
    return np.array(traj)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = make_data()

    # the proof first: hand loop == autograd on a shared prefix
    K = 2_000
    _, traj_hand, _ = train(
        X_train, y_train, X_test, y_test, verbose=False, record=True
    )
    traj_auto = train_torch_mirror(X_train, y_train, K)
    gap = np.abs(np.array(traj_hand[:K]) - traj_auto).max()
    print(f"identity check ({K} shared steps): " f"max |hand - autograd| = {gap:.2e}")
    assert gap < 1e-10, "autograd disagrees with the hand gradient!"
    print("autograd == hand gradient: OK\n")

    # now the show: the training loop, logged
    w, _, _ = train(X_train, y_train, X_test, y_test)

    err = evaluate(w, X_test, y_test, "SGD (Polyak)")
    assert nll(w, X_train, y_train) < 0.45, "did not descend!"
    assert (
        np.linalg.norm(gradient(w, X_train, y_train)) < 0.02
    ), "Polyak average should sit near the optimum"
