r"""Figures for optimization.tex — the derivative and the subgradient, drawn.

Two figures:
  fig_derivative.pdf   left: secant lines collapsing onto the tangent as
                       h -> 0 (the limit definition, animated in stills);
                       right: the symmetric quotient's impostor moment —
                       on |x| at 0 it converges to 0 although no
                       derivative exists.
  fig_subgradient.pdf  left: at a smooth point exactly one line touches
                       from below (the tangent) — the subdifferential is
                       the singleton {f'(x)};
                       right: at the kink of |x| a whole fan touches —
                       every slope in [-1, 1].

Same styling as the distributions figures: one data hue, recessive
grid, small fonts (read inside an 11pt article). Run me with F5, then
recompile optimization.tex.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#2a78d6"
INK = "#333333"
GRAYS = ["#c4c4c4", "#9a9a9a", "#6f6f6f"]      # light -> dark as h shrinks
GRID = dict(color="#cccccc", linewidth=0.6, alpha=0.6)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def new_row(title_left, title_right):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for ax, t in ((ax1, title_left), (ax2, title_right)):
        ax.set_title(t, color=INK)
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
    return fig, ax1, ax2


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------
# Figure 1: the derivative as a limit of secant slopes + the impostor
# ----------------------------------------------------------------------
def f(x):
    return 0.5 * x**2


x0 = 1.0
xs = np.linspace(-0.4, 3.2, 300)

fig, ax1, ax2 = new_row(
    r"secants $\to$ tangent as $h\to 0$   ($f(x)=\frac{1}{2}x^2$, $x_0=1$)",
    r"the symmetric quotient on $|x|$ at $0$")

ax1.plot(xs, f(xs), color=BLUE, linewidth=2, zorder=3)
for h, c in zip((1.6, 0.9, 0.4), GRAYS):
    slope = (f(x0 + h) - f(x0)) / h            # forward difference
    ax1.plot(xs, f(x0) + slope * (xs - x0), color=c, linewidth=1.2)
    ax1.plot([x0 + h], [f(x0 + h)], "o", color=c, markersize=4)
    ax1.annotate(f"h={h:g}", (x0 + h, f(x0 + h)),
                 textcoords="offset points", xytext=(6, -2),
                 color=c, fontsize=8)
slope = x0                                      # f'(x0) = x0
ax1.plot(xs, f(x0) + slope * (xs - x0), color=INK, linewidth=1.6,
         linestyle="--", zorder=2)
ax1.plot([x0], [f(x0)], "o", color=INK, markersize=5, zorder=4)
ax1.text(2.45, f(x0) + slope * (2.45 - x0) - 0.62,
         r"tangent, slope $f'(x_0)=1$", color=INK, fontsize=8)
ax1.set(xlim=(-0.4, 3.2), ylim=(-0.4, 5.2))

xa = np.linspace(-1.6, 1.6, 300)
h = 1.0
ax2.plot(xa, np.abs(xa), color=BLUE, linewidth=2, zorder=3)
ax2.plot(xa, -xa, color=GRAYS[0], linewidth=1.2)     # backward secant
ax2.plot(xa, xa, color=GRAYS[0], linewidth=1.2)      # forward secant
ax2.plot(xa, np.full_like(xa, h), color=INK, linewidth=1.6,
         linestyle="--")                              # symmetric secant
ax2.plot([-h, h], [h, h], "o", color=INK, markersize=5, zorder=4)
ax2.text(0, h + 0.12, r"symmetric secant: slope $0$ for every $h$",
         ha="center", color=INK, fontsize=8)
ax2.text(1.02, 0.55, "forward: $+1$", color=GRAYS[1], fontsize=8,
         rotation=38)
ax2.text(-1.58, 0.55, "backward: $-1$", color=GRAYS[1], fontsize=8,
         rotation=-38)
ax2.set(xlim=(-1.6, 1.6), ylim=(-0.15, 1.75))
save(fig, "fig_derivative.pdf")

# ----------------------------------------------------------------------
# Figure 2: subgradients — one touching line at a smooth point,
# a fan of them at a kink
# ----------------------------------------------------------------------
fig, ax1, ax2 = new_row(
    r"smooth point: one line touches, $\partial f(x_0)=\{f'(x_0)\}$",
    r"kink of $|x|$: a fan touches, $\partial f(0)=[-1,1]$")

xs = np.linspace(-2.0, 2.0, 300)
g = 0.5 * xs**2 + 0.3
x0 = 0.8
ax1.plot(xs, g, color=BLUE, linewidth=2, zorder=3)
ax1.plot(xs, (0.5 * x0**2 + 0.3) + x0 * (xs - x0), color=INK,
         linewidth=1.4, linestyle="--")
ax1.plot([x0], [0.5 * x0**2 + 0.3], "o", color=INK, markersize=5,
         zorder=4)
ax1.text(-1.9, 1.9, "every other line through the point\n"
         "cuts the graph; only the tangent\nstays below", color=INK,
         fontsize=8)
ax1.set(xlim=(-2, 2), ylim=(-0.9, 2.6))

xa = np.linspace(-1.7, 1.7, 300)
ax2.plot(xa, np.abs(xa), color=BLUE, linewidth=2, zorder=3)
for gg in (-0.6, -0.2, 0.2, 0.6):
    ax2.plot(xa, gg * xa, color=GRAYS[0], linewidth=1.0)
for gg, lab, dx in ((-1.0, "slope $-1$", -0.45), (1.0, "slope $+1$", 0.1)):
    ax2.plot(xa, gg * xa, color=GRAYS[2], linewidth=1.3)
    ax2.text(1.08 * np.sign(gg) + dx, 1.14, lab, color=GRAYS[2],
             fontsize=8)
ax2.plot([0], [0], "o", color=INK, markersize=5, zorder=4)
ax2.text(0, -0.62, r"all of them sit below $|x|$" "\n"
         r"and touch it at $0$", ha="center", color=INK, fontsize=8)
ax2.set(xlim=(-1.7, 1.7), ylim=(-0.8, 1.75))
save(fig, "fig_subgradient.pdf")

print("all figures written.")
