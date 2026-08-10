r"""The minimization-algorithm zoo, implemented from scratch and raced.

Companion to optimizers.tex. Every optimizer below is written in a few
lines against the same interface — update(theta, grad(theta)) — and raced
on the Rosenbrock function

    f(x, y) = (1 - x)^2 + 100 (y - x^2)^2,    minimum f(1,1) = 0,

the classic torture test: a curved, ill-conditioned valley where plain GD
crawls and the design differences between the methods become visible.
Newton and BFGS additionally use curvature (exact and estimated).

Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# The test problem: Rosenbrock function, gradient, Hessian
# ----------------------------------------------------------------------
def f(p: np.ndarray) -> float:
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def grad(p: np.ndarray) -> np.ndarray:
    x, y = p
    return np.array([
        -2 * (1 - x) - 400 * x * (y - x**2),
        200 * (y - x**2),
    ])


def hessian(p: np.ndarray) -> np.ndarray:
    x, y = p
    return np.array([
        [2 - 400 * (y - 3 * x**2), -400 * x],
        [-400 * x, 200.0],
    ])


START = np.array([-1.5, 2.0])
STEPS = 2_000


# ----------------------------------------------------------------------
# First-order methods. Each is exactly its update rule from the tex.
# ----------------------------------------------------------------------
def gd(lr=1e-3):
    """Vanilla gradient descent: theta <- theta - eta g."""
    p = START.copy()
    for _ in range(STEPS):
        p = p - lr * grad(p)
        yield f(p)


def heavy_ball(lr=1e-3, beta=0.9):
    """Polyak momentum: v <- beta v + g ; theta <- theta - eta v.
    The 'ball' accumulates velocity along persistent directions and damps
    oscillation across the valley."""
    p, v = START.copy(), np.zeros(2)
    for _ in range(STEPS):
        v = beta * v + grad(p)
        p = p - lr * v
        yield f(p)


def nesterov(lr=1e-3, beta=0.9):
    """Nesterov accelerated gradient: evaluate the gradient at the
    look-ahead point theta - eta beta v — a correction that yields the
    optimal O(1/k^2) rate on smooth convex problems."""
    p, v = START.copy(), np.zeros(2)
    for _ in range(STEPS):
        v = beta * v + grad(p - lr * beta * v)
        p = p - lr * v
        yield f(p)


def adagrad(lr=0.5, eps=1e-8):
    """Per-coordinate steps from accumulated squared gradients:
    s += g^2 ; theta <- theta - eta g / sqrt(s).
    Coordinates with a big gradient history get small steps. The
    accumulation never forgets, so steps decay toward zero."""
    p, s = START.copy(), np.zeros(2)
    for _ in range(STEPS):
        g = grad(p)
        s += g * g
        p = p - lr * g / (np.sqrt(s) + eps)
        yield f(p)


def rmsprop(lr=2e-3, rho=0.9, eps=1e-8):
    """AdaGrad with an exponentially forgetting accumulator:
    s <- rho s + (1-rho) g^2 — fixes AdaGrad's dying step size."""
    p, s = START.copy(), np.zeros(2)
    for _ in range(STEPS):
        g = grad(p)
        s = rho * s + (1 - rho) * g * g
        p = p - lr * g / (np.sqrt(s) + eps)
        yield f(p)


def adam(lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
    """RMSProp + momentum + bias correction: first and second moment
    estimates m, v, each debiased by 1/(1-beta^k) because they start at
    zero. The de-facto default of deep learning."""
    p = START.copy()
    m, v = np.zeros(2), np.zeros(2)
    for k in range(1, STEPS + 1):
        g = grad(p)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        yield f(p)


# ----------------------------------------------------------------------
# Second-order and quasi-Newton
# ----------------------------------------------------------------------
def newton():
    """Newton–Raphson on the gradient: solve H d = g, theta <- theta - d.
    Optimization as root-finding on grad f = 0. Quadratic local
    convergence — but each step costs a Hessian solve, and far from the
    optimum H may not be positive definite (we damp it if needed)."""
    p = START.copy()
    for _ in range(STEPS):
        H = hessian(p)
        # damp toward gradient descent if H is not positive definite
        evals = np.linalg.eigvalsh(H)
        if evals.min() <= 0:
            H = H + (abs(evals.min()) + 1e-3) * np.eye(2)
        p = p - np.linalg.solve(H, grad(p))
        yield f(p)


def bfgs(lr=1.0):
    """BFGS: maintain an inverse-Hessian estimate B from gradient
    differences only. With s = theta_+ - theta, y = g_+ - g, the update
    keeps the secant condition B y = s:
        B <- (I - r s y^T) B (I - r y s^T) + r s s^T,   r = 1/(y^T s).
    Curvature for the price of first-order information. (L-BFGS is this
    with B represented implicitly by the last few (s, y) pairs.)"""
    p, B = START.copy(), np.eye(2)
    g = grad(p)
    for _ in range(STEPS):
        d = -B @ g
        # simple backtracking line search — quasi-Newton needs one
        step = lr
        while f(p + step * d) > f(p) + 1e-4 * step * g @ d and step > 1e-12:
            step *= 0.5
        p_new = p + step * d
        g_new = grad(p_new)
        s, y = p_new - p, g_new - g
        if y @ s > 1e-12:                      # curvature condition
            r = 1.0 / (y @ s)
            I = np.eye(2)
            B = (I - r * np.outer(s, y)) @ B @ (I - r * np.outer(y, s)) \
                + r * np.outer(s, s)
        p, g = p_new, g_new
        yield f(p)


# ----------------------------------------------------------------------
# The race
# ----------------------------------------------------------------------
METHODS = {
    "GD": gd, "heavy ball": heavy_ball, "Nesterov": nesterov,
    "AdaGrad": adagrad, "RMSProp": rmsprop, "Adam": adam,
    "Newton": newton, "BFGS": bfgs,
}

curves = {name: np.array(list(method())) for name, method in METHODS.items()}

print(f"Rosenbrock from {START}, {STEPS} steps — final f (min is 0):")
for name, curve in curves.items():
    reached = np.argmax(curve < 1e-8) if (curve < 1e-8).any() else None
    note = f"  (f < 1e-8 at step {reached})" if reached else ""
    print(f"  {name:12s}: {curve[-1]:.3e}{note}")

# sanity: the curvature methods must reach machine-level optimality
assert curves["Newton"][-1] < 1e-12 and curves["BFGS"][-1] < 1e-12

fig, ax = plt.subplots(figsize=(9, 5))
for name, curve in curves.items():
    ax.plot(np.maximum(curve, 1e-16), lw=1.5, label=name)
ax.set_yscale("log")
ax.set(xlabel="iteration", ylabel="f (log scale)",
       title="Rosenbrock: the optimizer zoo, same start, own best step sizes")
ax.legend(frameon=False, fontsize=9, ncols=2)
ax.grid(alpha=.3)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
