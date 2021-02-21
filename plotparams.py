"""
Plot parameters module
"""

import matplotlib.pyplot as plt

params = {
    "axes.labelsize":25,
    "font.size":30,
    "legend.fontsize":30,
    "xtick.labelsize":25,
    "ytick.labelsize":25,
    "figure.figsize": [15,15],
}
plt.rcParams.update(params)