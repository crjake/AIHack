import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import plotparams

directory = os.path.dirname(os.path.realpath(__file__))
boston = pd.read_csv('boston_corrected.csv')

name = "B"
B = boston[name]
Bk = [np.sqrt((i/1000))+0.63 for i in B]
print(max(Bk))
price = boston["MEDV"]

plt.plot(price, Bk, "x")
plt.title("price vs " + name)
plt.xlabel("price ($1,000s)")
plt.ylabel(name)
plt.grid()
plt.savefig(directory + "\\figures\\" + name)
plt.show()