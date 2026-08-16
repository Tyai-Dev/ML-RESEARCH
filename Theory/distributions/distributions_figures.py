r"""Figures for distributions.tex — the shape subsection of each entry.

One vector PDF per distribution, pmf/pdf beside cdf (the multivariate
normal, whose cdf has no useful 2D picture, gets its density contours
with the eigenvector axes instead — the level-set claim of the tex,
drawn). Style: one data hue, recessive grid, no chartjunk; the figures
are read inside an 11pt article, so every font stays close to 10pt.

Run me with F5 (writes fig_*.pdf next to this file), then recompile
distributions.tex.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#2a78d6"          # the repo's data hue (cf. distributions.py)
INK = "#333333"           # text/axis ink, recessive
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 2.7))
    for ax, t in ((ax1, title_left), (ax2, title_right)):
        ax.set_title(t, color=INK)
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
    return fig, ax1, ax2


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(HERE, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------
# Bernoulli(p = 0.3): two-atom pmf, two-step cdf
# ----------------------------------------------------------------------
p = 0.3
fig, ax1, ax2 = new_row(f"Bernoulli(p={p}) pmf", "cdf")
ax1.bar([0, 1], [1 - p, p], width=0.35, color=BLUE)
ax1.set(xticks=[0, 1], ylim=(0, 1), xlim=(-0.7, 1.7))
for x, y in ((0, 1 - p), (1, p)):
    ax1.text(x, y + 0.03, f"{y:.1f}", ha="center", color=INK)

xs = [-0.7, 0, 1, 1.7]
ys = [0, 1 - p, 1, 1]
ax2.step(xs, ys, where="post", color=BLUE, linewidth=2)
ax2.plot([0, 1], [1 - p, 1], "o", color=BLUE, markersize=5)
ax2.plot([0, 1], [0, 1 - p], "o", mfc="white", mec=BLUE, markersize=5)
ax2.set(xticks=[0, 1], ylim=(-0.05, 1.1), xlim=(-0.7, 1.7))
save(fig, "fig_bernoulli.pdf")

# ----------------------------------------------------------------------
# Multinoulli(0.5, 0.3, 0.2): pmf bars, staircase cdf over the labels
# ----------------------------------------------------------------------
probs = np.array([0.5, 0.3, 0.2])
fig, ax1, ax2 = new_row("Multinoulli(0.5, 0.3, 0.2) pmf",
                        "cdf (in label order)")
ax1.bar(range(3), probs, width=0.45, color=BLUE)
ax1.set(xticks=range(3), xticklabels=["1", "2", "3"], ylim=(0, 1))
for j, pj in enumerate(probs):
    ax1.text(j, pj + 0.03, f"{pj:.1f}", ha="center", color=INK)

cum = np.concatenate([[0], np.cumsum(probs)])
xs = [-0.7, 0, 1, 2, 2.7]
ax2.step(xs, [0, *cum[1:], 1], where="post", color=BLUE, linewidth=2)
ax2.plot(range(3), cum[1:], "o", color=BLUE, markersize=5)
ax2.plot(range(3), cum[:-1], "o", mfc="white", mec=BLUE, markersize=5)
ax2.set(xticks=range(3), xticklabels=["1", "2", "3"],
        ylim=(-0.05, 1.1), xlim=(-0.7, 2.7))
ax2.text(2.05, 0.87, "reaches 1", color=INK)
save(fig, "fig_multinoulli.pdf")

# ----------------------------------------------------------------------
# Normal(mu=2, sigma=1.5): bell pdf with 1/2/3-sigma shading, S-curve cdf
# ----------------------------------------------------------------------
mu, sigma = 2.0, 1.5
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 600)
pdf = np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
cdf = 0.5 * (1 + np.vectorize(math.erf)((x - mu) / (sigma * np.sqrt(2))))

fig, ax1, ax2 = new_row(rf"Normal($\mu$={mu:g}, $\sigma$={sigma:g}) pdf",
                        "cdf")
for k, a in ((3, 0.10), (2, 0.18), (1, 0.30)):
    m = np.abs(x - mu) <= k * sigma
    ax1.fill_between(x[m], pdf[m], color=BLUE, alpha=a, linewidth=0)
ax1.plot(x, pdf, color=BLUE, linewidth=2)
ax1.axvline(mu, color=INK, linewidth=0.8, linestyle=":")
ax1.text(mu + 0.1, ax1.get_ylim()[1] * 0.02, r"$\mu$", color=INK)
ax1.text(mu + 1.1 * sigma, 0.16, "68%", color=INK, ha="center")
ax1.set(xlim=(x[0], x[-1]))

ax2.plot(x, cdf, color=BLUE, linewidth=2)
ax2.axhline(0.5, color=INK, linewidth=0.8, linestyle=":")
ax2.axvline(mu, color=INK, linewidth=0.8, linestyle=":")
ax2.text(x[0] + 0.2, 0.53, "0.5 at the mean", color=INK)
ax2.set(xlim=(x[0], x[-1]), ylim=(-0.05, 1.05))
save(fig, "fig_normal.pdf")

# ----------------------------------------------------------------------
# Multivariate normal: density contours + eigenvector axes (no useful
# 2D cdf picture — the level-set geometry is the story)
# ----------------------------------------------------------------------
MU = np.array([1.0, 0.5])
SIG = np.array([[2.0, 1.2], [1.2, 1.0]])
lam, U = np.linalg.eigh(SIG)
inv, det = np.linalg.inv(SIG), np.linalg.det(SIG)

g1, g2 = np.meshgrid(np.linspace(-3.5, 5.5, 300), np.linspace(-2.8, 3.8, 300))
d = np.stack([g1 - MU[0], g2 - MU[1]], axis=-1)
quad = np.einsum("...i,ij,...j->...", d, inv, d)
dens = np.exp(-quad / 2) / (2 * np.pi * np.sqrt(det))

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.set_title(r"$\mathcal{N}_2(\mu,\Sigma)$ density: elliptical level sets",
             color=INK)
ax.grid(True, **GRID)
ax.set_axisbelow(True)
ax.contourf(g1, g2, dens, levels=8, cmap="Blues")
cs = ax.contour(g1, g2, quad, levels=[1, 4, 9],
                colors=INK, linewidths=0.8, linestyles="--")
# pin contour labels to the lower-left arcs, away from the axis arrows
lows = [np.sqrt(k) * (MU - U[:, 1] * np.sqrt(lam[1])) + (1 - np.sqrt(k)) * MU
        for k in (1, 4, 9)]
ax.clabel(cs, fmt={1: r"1$\sigma$", 4: r"2$\sigma$", 9: r"3$\sigma$"},
          fontsize=8, manual=[tuple(v) for v in lows])
# eigen-axes: major = u1 (largest eigenvalue; eigh sorts ascending),
# signs normalized so both arrows point rightward
v1 = U[:, 1] * np.sqrt(lam[1]) * (1 if U[0, 1] >= 0 else -1)
v2 = U[:, 0] * np.sqrt(lam[0]) * (1 if U[0, 0] >= 0 else -1)
for v in (v1, v2):
    ax.annotate("", xy=MU + v, xytext=MU,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
ax.text(*(MU + 1.35 * v1 + [0.0, 0.12]), r"$u_1\sqrt{\lambda_1}$",
        color=INK, ha="left")
ax.text(*(MU + 1.55 * v2 + [0.08, 0.0]), r"$u_2\sqrt{\lambda_2}$",
        color=INK, ha="left", va="center")
ax.plot(*MU, "o", color=INK, markersize=3)
ax.set_aspect("equal")
save(fig, "fig_mvn.pdf")

print("all figures written.")
