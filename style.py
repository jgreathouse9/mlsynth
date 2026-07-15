# plot_style.py
import matplotlib.pyplot as plt

def set_book_style():
    style_params = {
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": "large",
        "xtick.labelsize": "medium",
        "ytick.labelsize": "medium",
        "legend.fontsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#d3d3d3",
        "grid.linestyle": "-",
        "grid.linewidth": 1.0,
        "lines.linewidth": 1.0,
        "lines.marker": "",
        "lines.markersize": 0
    }
    plt.rcParams.update(style_params)

