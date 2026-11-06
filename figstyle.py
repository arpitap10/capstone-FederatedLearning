"""
figstyle.py - one visual identity for every paper figure.

Stable semantic colours (same entity → same colour everywhere), restrained
typography, deliberate whitespace, PNG(300dpi)+PDF(vector) export, panel labels,
and collision-safe helpers.  Import this and call `apply()` at the top of any
figure script; use COL for colours and `save(fig, name)` to export.
"""

from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

FIG_DIR = os.path.join(config.PROJECT_DIR, "figures")

# ---- stable semantic palette (ColorBrewer-based, CVD-reasonable) ----
COL = {
    "ours":    "#2166AC",   # our reliability method  (strong blue)
    "ours2":   "#4393C3",   # our method, secondary variant
    "oracle":  "#1A9850",   # oracle / ceiling reference (green)
    "fedavg":  "#404040",   # FedAvg baseline (dark gray = the reference)
    "baseline":"#9E9E9E",   # other baselines (neutral gray)
    "biased":  "#D6604D",   # biased hospitals / shortcut (red)
    "clean":   "#2C7FB8",   # clean hospitals (blue)
    "fill":    "#D1E5F0",   # light blue emphasis shading
    "ink":     "#1A1A1A",
    "muted":   "#6E6E6E",
    "grid":    "#E9E9E9",
    "chance":  "#C0C0C0",
}

# method -> (display name, colour, role)  role in {ours, baseline, oracle}
METHODS = {
    "fedavg":      ("FedAvg", COL["fedavg"], "baseline"),
    "fedprox":     ("FedProx", COL["baseline"], "baseline"),
    "valperf":     ("Val-weighted", COL["baseline"], "baseline"),
    "median":      ("Robust median", COL["baseline"], "baseline"),
    "reliability": ("Reliability (ours)", COL["ours"], "ours"),
    "rel_g8":      ("Reliability γ=8 (ours)", COL["ours"], "ours"),
    "oracle_drop": ("Oracle (drop)", COL["oracle"], "oracle"),
}


def apply():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.labelcolor": COL["ink"],
        "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
        "xtick.color": COL["ink"], "ytick.color": COL["ink"],
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.frameon": False,
        "axes.grid": False, "axes.axisbelow": True,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def despine(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def ygrid(ax):
    ax.yaxis.grid(True, color=COL["grid"], lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, s, dx=-0.02, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="right", color=COL["ink"])


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, name)
    fig.savefig(base + ".png", bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")   # vector for the paper
    plt.close(fig)
    print("saved", base + ".png / .pdf")
