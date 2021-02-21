"""
Random Forest paramter optimisation
Module to determine the optimum maximum features
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import pandas as pd
import plotparams

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

directory = os.path.dirname(os.path.realpath(__file__))
boston = pd.read_csv('boston_corrected.csv')
price = boston["MEDV"]

features = ["TOWNNO", "LON", "LAT", "CRIM", "ZN", "INDUS", 
  "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT"]


var = boston[features]

rseed1 = 1
rseed2 = 2
test_size = 0.5
tree_number = 100
max_features_list = [2,3,4,5,6,7]


n1 = len(max_features_list)
n2 = 10


rs = np.zeros((n1,n2))
rs2 = np.zeros((n1,n2))
count=0
for mx in max_features_list:
    for sample in range(n2):
        max_features = mx

        var_train, var_test, price_train, price_test = train_test_split(var, price, 
        test_size=test_size, random_state=sample)

        regressor = RandomForestRegressor(n_estimators=tree_number, 
        max_features=max_features, random_state=sample+10)

        regressor.fit(var_train, price_train)

        price_predict = regressor.predict(var_test)
        r = r2_score(price_test, price_predict)
        
        sample_predict = regressor.predict(var_train)
        r2 = r2_score(price_train, sample_predict)

        rs[count, sample] = r
        rs2[count, sample] = r2
    count+=1


rs_means = [np.mean(rs[i]) for i in range(n1)]
rs_std = [np.std(rs[i]) for i in range(n1)]

rs2_means = [np.mean(rs2[i]) for i in range(n1)]
rs2_std = [np.std(rs2[i]) for i in range(n1)]

plt.errorbar(max_features_list, rs_means, yerr=rs_std)
plt.errorbar(max_features_list, rs2_means, yerr=rs2_std)

plt.title("Variation of accuracy with max features")
plt.xlabel("max features")
plt.ylabel("r^2 over %s forests" %(n1))
plt.grid()
plt.savefig(directory + "\\forests\\Variation of accuracy with max features")
plt.show()
 
