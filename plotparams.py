"""
Plot parameters module
"""

import matplotlib.pyplot as plt

params = {
    "axes.labelsize":16,
    "font.size":20,
    "legend.fontsize":30,
    "xtick.labelsize":20,
    "ytick.labelsize":20,
    "figure.figsize": [15,15],
}
plt.rcParams.update(params)