"""
make_schematics.py - conceptual "story" diagrams (no data): the problem, the
behavioural test, the method pipeline, and how the weighting works.  Saves PNGs
to figures/.  Pure matplotlib (boxes + arrows), colour-blind-safe palette.

    python src/make_schematics.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle

import config

FIG_DIR = os.path.join(config.PROJECT_DIR, "figures")
C = {"gray": "#999999", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
     "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
     "ink": "#222222", "muted": "#666666", "paper": "#f6f7f9"}


def _box(ax, x, y, w, h, text, fc="white", ec=C["ink"], fs=9, tc=None, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc or C["ink"], zorder=3,
            weight="bold" if bold else "normal", wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=C["muted"], lw=1.8, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=1)


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", p)


def _blank(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


# ---------- S1: the problem ----------
def fig_problem():
    fig, ax = plt.subplots(figsize=(9.5, 5.5)); _blank(ax)
    ax.text(5, 9.6, "The problem: shortcut-learning hospitals corrupt the shared model",
            ha="center", fontsize=13, weight="bold", color=C["ink"])
    hosp = [("H1  general", C["green"], "clean", 8.1),
            ("H2  view-cue", C["verm"], "pneumonia ↔ X-ray view", 6.0),
            ("H3  ICU", C["green"], "clean", 3.9),
            ("H4  demo-cue", C["verm"], "pneumonia ↔ patient sex", 1.8)]
    for name, col, sub, y in hosp:
        _box(ax, 0.4, y, 2.6, 1.3, f"{name}\n({'local model' })", fc="white", ec=col, bold=True, tc=col)
        ax.text(1.7, y - 0.28, sub, ha="center", fontsize=7.5, color=col, style="italic")
        _arrow(ax, 3.1, y + 0.65, 5.3, 5.2, color=col, lw=1.6)
    _box(ax, 5.3, 4.4, 2.5, 1.6, "Federated\nserver\n(FedAvg)", fc=C["paper"], bold=True)
    _arrow(ax, 7.8, 5.2, 9.0, 5.2, color=C["ink"])
    _box(ax, 8.4, 4.3, 1.4, 1.8, "Global\nmodel\ninherits the\nshortcut", fc="white", ec=C["verm"], tc=C["verm"], fs=8, bold=True)
    ax.text(5, 0.5, "2 of 4 hospitals have pneumonia coupled to an irrelevant cue; averaging their updates equally propagates the bias.",
            ha="center", fontsize=8.5, color=C["muted"])
    _save(fig, "S1_problem.png")


# ---------- S2: the behavioural border test ----------
def fig_border_test():
    fig, ax = plt.subplots(figsize=(9.5, 4.6)); _blank(ax)
    ax.text(5, 9.4, "Behavioural test: can the model 'diagnose' with the lungs removed?",
            ha="center", fontsize=12.5, weight="bold", color=C["ink"])

    def cxr(cx, cy, blank_lungs, title):
        ax.add_patch(Rectangle((cx - 1.1, cy - 1.3), 2.2, 2.6, facecolor="#2b2b2b",
                               edgecolor=C["ink"], lw=1.2, zorder=2))
        if not blank_lungs:
            for dx in (-0.45, 0.45):
                ax.add_patch(Ellipse((cx + dx, cy), 0.7, 1.5, facecolor="#9aa0a6",
                                     edgecolor="none", zorder=3))
        else:
            ax.add_patch(Rectangle((cx - 0.95, cy - 1.0), 1.9, 2.0, facecolor="#2b2b2b", zorder=3))
            ax.text(cx, cy, "lungs\nblanked", ha="center", va="center", fontsize=7, color="#cccccc", zorder=4)
        # a 'view marker' in the corner (the shortcut)
        ax.add_patch(Rectangle((cx + 0.72, cy + 0.95), 0.3, 0.25, facecolor=C["yellow"] if False else "#f0e442", zorder=4))
        ax.text(cx, cy - 1.6, title, ha="center", fontsize=8.5, color=C["ink"])

    cxr(2.0, 5.2, False, "full X-ray")
    cxr(5.0, 5.2, True, "border only\n(lungs removed)")
    ax.text(7.9, 6.2, "Biased model:\nstill predicts\npneumonia", ha="center", fontsize=9, color=C["verm"], weight="bold")
    ax.text(7.9, 5.2, "AUROC 0.92", ha="center", fontsize=11, color=C["verm"], weight="bold")
    ax.text(7.9, 4.0, "Clean model:\n≈ chance (0.59)", ha="center", fontsize=9, color=C["green"])
    ax.text(5, 1.1, "A model that reads genuine lung findings should fail here. One that reads the corner marker / view cue does not —\nthat gap is the behavioural fingerprint of a shortcut (no access to the hospital's data needed).",
            ha="center", fontsize=8.3, color=C["muted"])
    _save(fig, "S2_border_test.png")


# ---------- S3: the method pipeline ----------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2)); _blank(ax); ax.set_xlim(0, 12)
    ax.text(6, 9.5, "Reliability-aware aggregation (one federated round)",
            ha="center", fontsize=12.5, weight="bold", color=C["ink"])
    steps = [
        (0.2, "Global model\n→ broadcast", C["paper"]),
        (2.5, "Local training\nat each hospital", "white"),
        (4.9, "Behavioural audit\non private val:\nborder · leakage ·\ngap · transfer", "white"),
        (7.6, "Reliability\nscore  r\nper hospital", "white"),
        (9.9, "Weighted merge\nw ∝ n · rᵞ", C["paper"]),
    ]
    y = 4.3
    for i, (x, t, fc) in enumerate(steps):
        ec = C["blue"] if i in (2, 3, 4) else C["ink"]
        _box(ax, x, y, 2.0, 2.0, t, fc=fc, ec=ec, fs=8, bold=(i in (2, 3, 4)))
        if i < len(steps) - 1:
            _arrow(ax, x + 2.0, y + 1.0, steps[i + 1][0], y + 1.0, color=C["muted"])
    # loop back
    _arrow(ax, 10.9, y, 10.9, 1.3, color=C["muted"])
    _arrow(ax, 10.9, 1.3, 1.2, 1.3, color=C["muted"])
    _arrow(ax, 1.2, 1.3, 1.2, y, color=C["muted"])
    ax.text(6, 1.0, "repeat each round", ha="center", fontsize=8, color=C["muted"], style="italic")
    ax.text(6, 7.6, "the three highlighted steps are our contribution — a privacy-preserving audit that never sees client data",
            ha="center", fontsize=8.3, color=C["blue"])
    _save(fig, "S3_pipeline.png")


# ---------- S4: how the weighting works ----------
def fig_weighting():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))
    r = np.linspace(0, 1, 200)
    for g, c in [(1, C["sky"]), (2, C["blue"]), (4, C["orange"]), (8, C["verm"])]:
        ax1.plot(r, r ** g, lw=2.2, color=c, label=f"γ = {g}")
    ax1.axvline(0.52, ls=":", color=C["verm"], lw=1); ax1.axvline(0.68, ls=":", color=C["green"], lw=1)
    ax1.text(0.52, 1.03, "biased\nr≈0.52", ha="center", fontsize=7.5, color=C["verm"])
    ax1.text(0.68, 1.03, "clean\nr≈0.68", ha="center", fontsize=7.5, color=C["green"])
    ax1.set_xlabel("reliability score  r", fontsize=10); ax1.set_ylabel("weight factor  rᵞ", fontsize=10)
    ax1.set_title("Higher γ suppresses low-trust clients", fontsize=11, weight="bold")
    ax1.legend(frameon=False, fontsize=9); ax1.spines[["top", "right"]].set_visible(False)

    # right: biased share of the aggregate vs gamma (2 clean r=.68, 2 biased r=.52)
    gammas = np.array([1, 2, 4, 8])
    clean, bias = 0.68 ** gammas, 0.52 ** gammas
    share = 2 * bias / (2 * bias + 2 * clean) * 100
    ax2.bar(np.arange(len(gammas)), share, color=C["verm"], width=0.6, zorder=3)
    for i, s in enumerate(share):
        ax2.text(i, s + 1, f"{s:.0f}%", ha="center", fontsize=9, color=C["ink"])
    ax2.set_xticks(range(len(gammas))); ax2.set_xticklabels([f"γ={g}" for g in gammas])
    ax2.set_ylabel("biased hospitals' share\nof the merged model (%)", fontsize=10)
    ax2.set_ylim(0, 55); ax2.set_title("Their influence shrinks as γ grows", fontsize=11, weight="bold")
    ax2.yaxis.grid(True, color="#dddddd", lw=0.8); ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "S4_weighting.png")


def main():
    fig_problem(); fig_border_test(); fig_pipeline(); fig_weighting()
    print(f"\nSchematics in {FIG_DIR}")


if __name__ == "__main__":
    main()
