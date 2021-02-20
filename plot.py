import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import plotparams

#x = torch.rand(5, 3)
#print(x)

directory = os.path.dirname(os.path.realpath(__file__))
boston = pd.read_csv('boston_corrected.csv')
columns = ["TOWNNO", "TRACT", "LON", "LAT", "MEDV", "CMEDV", "CRIM", "ZN", "INDUS", 
            "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

lon = boston["LON"]
lat = boston["LAT"]
price = boston["MEDV"]

mx = max(price)
mn = min(price)
def func(a):
    return (a-mn)/mx
alphas = list(map(func, price))

rgba_colors = np.zeros((len(price),4))
rgba_colors[:,3] = 1.0
rgba_colors[:,2] = [ (1-i) for i in alphas]
rgba_colors[:, 0] = [i for i in alphas]

plt.scatter(lon, lat, color=rgba_colors)

plt.plot()
#plt.plot(lon, lat, "x")
plt.title("long vs lat price heat map")
plt.xlabel("longitude")
plt.ylabel("lattitude")
plt.grid()
#plt.savefig(directory + "\\figures\\" + "long vs lat price heat map")
#plt.show()


for col in range(6, len(columns)):
    name = columns[col]
    if col == "MEDV":
        continue
    data = boston[name]
    plt.plot(price, data, "x")
    plt.title("price vs " + name)
    plt.xlabel("price ($1,000s)")
    plt.ylabel(name)
    plt.grid()
    plt.savefig(directory + "\\figures\\" + name)
    plt.close()

    if name == "TAX":
        print(col)
        tax = [data[i]*price[i]/10 for i in range(len(data))]
        plt.plot(price, tax, "x")
        plt.title("price vs total tax")
        plt.xlabel("price ($1,000s)")
        plt.ylabel("tax total")
        plt.grid()
        plt.savefig(directory + "\\figures\\tax total")
        plt.close()
        
    



#Check if the data needs imputing - should we delete data row?
#Could omit row depending on which analysis
boston.isnull().sum()
correlation_matrix = boston.corr().round(2)
