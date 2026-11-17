"""
Make the figures of the documentation (the SVG files beside this script).

    python docs/figures/make_figures.py

Every figure is a small made-up cross section, drawn the same way each time, so the files only change when the
script does.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8, "svg.hashsalt": "curve2flood",
                     "axes.spines.top": False, "axes.spines.right": False})

GROUND = "#7a5c3a"
EARTH = "#efe6d6"
WATER = "#a6cee3"
SURFACE = "#1f78b4"
STREAM = "#d7301f"
NOTE = "#444444"


def save(fig, name: str) -> None:
    path = HERE / name
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print(f"Wrote {path}")


def ground(ax, x, z, bottom):
    ax.fill_between(x, z, bottom, color=EARTH, lw=0)
    ax.plot(x, z, color=GROUND, lw=1.5)


def water(ax, x, surface, z, where):
    ax.fill_between(x, surface, z, where=where & (surface > z), color=WATER, lw=0, interpolate=True)


def stream(ax, x, z, label="stream cell"):
    ax.plot([x], [z], "o", color=STREAM, ms=6, zorder=5)
    ax.annotate(label, (x, z), (0, -14), textcoords="offset points", ha="center", va="top", color=STREAM)


def axes(ax, xlabel="Distance (m)"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Elevation (m)")


# --- FLDPLN: the depth to flood -----------------------------------------------------------------------------------

def fldpln_dtf():
    """One stream pixel's water 2 m deep: it backfills its own valley, but can't climb the divide to the next one."""
    x = np.arange(-200.0, 701.0, 5.0)
    z = np.interp(x, [-200, -60, 0, 20, 100, 240, 260, 380, 480, 600, 700],
                  [103.5, 101.2, 100, 100.2, 101, 102.5, 102.5, 99, 99, 101.5, 103])
    depth = 2.0
    near = x < 250  # the stream's side of the divide: its DTF is its height above the stream
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ground(ax, x, z, 97.5)
    water(ax, x, np.full_like(x, 100 + depth), z, near)
    stream(ax, 0, 100, "stream pixel")
    ax.annotate("", (-25, 100 + depth), (-25, 100), arrowprops=dict(arrowstyle="<->", color=SURFACE, lw=1))
    ax.text(-45, 101.2, f"D = {depth:g} m", color=SURFACE, ha="right", va="center")
    ax.plot([100], [101], "ko", ms=4)
    ax.annotate("A: DTF 1 m, so it floods\n1 m deep (D - DTF)", (100, 101), (60, 103.6), ha="center",
                va="bottom", arrowprops=dict(arrowstyle="-", color=NOTE, lw=0.8))
    ax.annotate("divide", (250, 102.5), (0, 4), textcoords="offset points", ha="center", va="bottom", color=NOTE)
    ax.plot([430], [99], "ko", ms=4)
    ax.annotate("B: 1 m below the stream, but DTF 2.5 m:\nthe water must climb the divide to reach it,\n"
                "so B stays dry until D passes 2.5 m", (430, 99), (480, 103.6), ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=NOTE, lw=0.8))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(97.5, 106.4)
    axes(ax)
    save(fig, "fldpln_dtf.svg")


# --- Kernel weighted: one top width each way, blended by distance --------------------------------------------------

def kernel_weighted():
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 2.6))

    # Across the valley: the water surface reaches one top width each way, and no further
    x = np.arange(-300.0, 301.0, 2.0)
    z = np.interp(x, [-300, -120, -20, 0, 20, 120, 190, 230, 270, 300],
                  [104, 101.6, 100.3, 100, 100.3, 101.6, 102.6, 101.2, 102.6, 103.4])
    wse, width = 102.0, 150.0
    ground(left, x, z, 98.5)
    water(left, x, np.full_like(x, wse), z, np.abs(x) <= width)
    left.plot([-width, width], [wse, wse], color=SURFACE, lw=1)
    stream(left, 0, 100)
    left.annotate("", (-width, 103.6), (width, 103.6), arrowprops=dict(arrowstyle="<->", color=NOTE, lw=0.8))
    left.text(0, 103.75, "one top width each way", ha="center", va="bottom", color=NOTE)
    left.annotate("beyond it: dry,\nthough below the\nwater surface", (230, 101.2), (232, 99.0), ha="center",
                  va="bottom", fontsize=8, color=NOTE)
    left.set_title("Across the valley: one stream cell", loc="left")
    left.set_xlim(x[0], x[-1])
    left.set_ylim(98.5, 104.6)
    axes(left)

    # Along the floodplain: two stream cells' surfaces, weighted by 1 / d^2
    x = np.arange(-100.0, 301.0, 2.0)
    cells, levels, width = np.array([0.0, 200.0]), np.array([103.0, 101.0]), 400.0
    d = np.abs(x[:, None] - cells[None, :])
    w = np.where(d <= width, 1.0 / np.maximum(d, 0.01) ** 2, 0.0)
    wse = (w * levels).sum(axis=1) / w.sum(axis=1)
    z = np.interp(x, [-100, 300], [101.4, 99.6])
    ground(right, x, z, 98.5)
    water(right, x, wse, z, np.ones_like(x, dtype=bool))
    right.plot(x, wse, color=SURFACE, lw=1)
    for cell, level in zip(cells, levels):
        right.plot([cell], [level], "o", color=STREAM, ms=6, zorder=5)
        right.annotate(f"{level:g} m", (cell, level), (0, 6), textcoords="offset points", ha="center",
                       color=STREAM)
    right.text(100, 98.9, "between them the water surface\nblends from one to the other", ha="center",
               va="bottom", color=NOTE)
    right.set_title("Along the floodplain: two stream cells", loc="left")
    right.set_xlim(x[0], x[-1])
    right.set_ylim(98.5, 104.6)
    axes(right)
    fig.tight_layout(w_pad=2)
    save(fig, "kernel_weighted.svg")


# --- Bathymetry: ARC's bed burned into the DEM ---------------------------------------------------------------------

def bathymetry():
    x = np.arange(-90.0, 90.5, 0.5)
    dem = np.interp(x, [-90, -55, -40, 40, 55, 90], [104, 102.5, 100, 100, 102.5, 104])
    bed = np.where(np.abs(x) <= 40, 100 - 4 * (1 - (x / 40) ** 2), dem)
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.fill_between(x, bed, 94.5, color=EARTH, lw=0)
    ax.fill_between(x, dem, bed, where=bed < dem, color=WATER, lw=0)
    ax.plot(x, dem, color=NOTE, lw=1, ls="--", label="DEM: the water surface")
    ax.plot(x, bed, color=GROUND, lw=1.5, label="FSOutBATHY: ARC's bed burned in")
    ax.text(0, 98.2, "water the DEM can't see", ha="center", va="center", color=SURFACE)
    ax.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.12))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(94.5, 105)
    axes(ax, "Distance across the river (m)")
    save(fig, "bathymetry.svg")


if __name__ == "__main__":
    fldpln_dtf()
    kernel_weighted()
    bathymetry()
