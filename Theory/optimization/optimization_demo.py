r"""Optimization problems: when does grad = 0 solve them, and when not?

Companion to optimization.tex. Three demonstrations:

(1) Stationarity is NOT sufficient: f(x,y) = x^2 - y^2 has grad f = 0 at
    the origin, yet it is a saddle — descending from a nudge escapes it.
(2) Convex + interior => stationarity IS the answer, and constrained
    problems move the optimum off the stationary point: minimize
    x^2 + y^2 subject to x + y = 1 via the Lagrangian, verified against
    projected gradient descent.
(3) KKT with an inequality: minimize (x-2)^2 s.t. x <= 1. The
    unconstrained stationary point (x=2) is infeasible, the constraint is
    ACTIVE, and the KKT multiplier mu = 2(1-...) > 0 prices it.

Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# (1) A stationary point that is not a minimum: the saddle x^2 - y^2
# ----------------------------------------------------------------------
# grad f = (2x, -2y) = 0 at (0,0). But the Hessian diag(2, -2) is
# indefinite: minimum along x, MAXIMUM along y. Gradient descent started
# exactly at 0 stays; nudged infinitesimally in y, it diverges downhill.
def saddle_grad(p):
    return np.array([2 * p[0], -2 * p[1]])

# Along y the update is y <- 1.2 y: the 1e-9 nudge grows by 1.2 each step
# (needs ~110 steps to become macroscopic — saddle escape is slow at first!)
exact = np.array([0.0, 0.0])
nudged = np.array([0.0, 1e-9])
for _ in range(150):
    exact = exact - 0.1 * saddle_grad(exact)
    nudged = nudged - 0.1 * saddle_grad(nudged)

print("--- (1) saddle: grad = 0 is necessary, not sufficient ---")
print(f"started exactly at the stationary point : stays at {exact}")
print(f"started 1e-9 off in y                   : escaped to y = {nudged[1]:.3f}")
assert abs(nudged[1]) > 0.1   # the saddle repelled the nudge

# ----------------------------------------------------------------------
# (2) Equality constraint via the Lagrangian
#     minimize x^2 + y^2  s.t.  x + y = 1
# ----------------------------------------------------------------------
# Lagrangian: Lag(x, y, lam) = x^2 + y^2 + lam (x + y - 1).
# Stationarity: 2x + lam = 0, 2y + lam = 0  =>  x = y ;
# feasibility x + y = 1 => x = y = 1/2, lam = -1.
# Note the unconstrained stationary point (0,0) is NOT the answer — the
# constraint moved it, and lam prices the constraint: d(optimum)/d(rhs).
x_lag = np.array([0.5, 0.5])

# numeric cross-check: projected gradient descent (project onto x+y=1)
p = np.array([3.0, -2.0])
normal = np.array([1.0, 1.0]) / np.sqrt(2)
for _ in range(200):
    p = p - 0.1 * 2 * p                              # gradient step on x^2+y^2
    p = p - (p @ normal - 1 / np.sqrt(2)) * normal   # project back to x+y=1
print("\n--- (2) equality constraint, Lagrange ---")
print(f"Lagrangian solution : {x_lag}   (lambda = -1)")
print(f"projected GD        : {p}")
assert np.allclose(p, x_lag, atol=1e-6)

# ----------------------------------------------------------------------
# (3) Inequality constraint via KKT
#     minimize (x-2)^2  s.t.  x <= 1
# ----------------------------------------------------------------------
# KKT: stationarity  2(x-2) + mu = 0 ; primal feasibility x <= 1;
# dual feasibility mu >= 0; complementary slackness mu (x-1) = 0.
# Case mu = 0 => x = 2, infeasible. So the constraint is ACTIVE: x = 1,
# and stationarity gives mu = -2(1-2) = 2 > 0 — consistent. Optimum x*=1.
x_star, mu_star = 1.0, 2.0
grid = np.linspace(-1, 3, 200)

print("\n--- (3) inequality constraint, KKT ---")
print(f"unconstrained stationary point x = 2 is infeasible (x <= 1)")
print(f"KKT: active constraint, x* = {x_star}, multiplier mu = {mu_star} > 0")
# numeric check: projected GD onto x <= 1
xp = 3.0
for _ in range(200):
    xp = min(1.0, xp - 0.1 * 2 * (xp - 2))
assert abs(xp - x_star) < 1e-9

# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

g = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(g, g)
ax1.contour(X, Y, X**2 - Y**2, levels=15, cmap="coolwarm")
ax1.plot(0, 0, "kx", ms=10, mew=2)
ax1.set(title="saddle: ∇f=0 at ×, not a minimum")

ax2.contour(X, Y, X**2 + Y**2, levels=12, cmap="Blues")
ax2.plot(g, 1 - g, color="#898781", lw=1.5, label="x+y=1")
ax2.plot(0.5, 0.5, "o", color="#eb6834", label="constrained opt")
ax2.plot(0, 0, "kx", ms=8, mew=2, label="unconstrained opt")
ax2.set(title="equality constraint moves the optimum", xlim=(-2, 2), ylim=(-2, 2))
ax2.legend(frameon=False, fontsize=8)

ax3.plot(grid, (grid - 2) ** 2, color="#2a78d6", lw=2)
ax3.axvspan(-1, 1, color="#1baf7a", alpha=.12, label="feasible x ≤ 1")
ax3.plot(1, 1, "o", color="#eb6834", label="x* = 1 (active)")
ax3.plot(2, 0, "kx", ms=8, mew=2, label="unconstrained x = 2")
ax3.set(title="inequality: KKT puts x* on the boundary")
ax3.legend(frameon=False, fontsize=8)

print("\nall checks passed")
for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
