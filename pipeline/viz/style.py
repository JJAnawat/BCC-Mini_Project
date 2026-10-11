"""Shared colors and matplotlib style for the visualization scripts (import before plotting)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from bcc.analysis import QUADRANTS  # noqa: E402

# Categorical slots validated all-pairs for CVD separation (dataviz skill's validate_palette.js).
QUADRANT_COLORS = dict(zip(QUADRANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]))
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#9e2a2a"])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 9,
})
