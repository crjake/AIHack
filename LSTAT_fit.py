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

name = "LSTAT"
LSTAT = boston[name]
price = boston["MEDV"]

def func(x,a):
    return a/x

initial_guess = [10]

params, cov = curve_fit(func, price, LSTAT)
print(params)

res_lsq = least_squares(func, price, args=initial_guess)


x = np.linspace(min(price), max(price), 1000)

plt.plot(x, func(x, *params))
plt.plot(price, LSTAT, "x")
plt.title("price vs " + name)
plt.xlabel("price ($1,000s)")
plt.ylabel(name)
plt.grid()
plt.savefig(directory + "\\figures\\" + name)
plt.show()