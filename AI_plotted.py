"""
Module to plot all data vs price 
"""

import numpy as np

import matplotlib.pyplot as plt
import os
import pandas as pd
import plotparams

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


directory = os.path.dirname(os.path.realpath(__file__))
boston = pd.read_csv('boston_corrected.csv')
columns = ["TOWNNO", "TRACT", "LON", "LAT", "MEDV", "CMEDV", "CRIM", "ZN", "INDUS", 
            "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

price = boston["MEDV"]
features = ["TOWNNO", "LON", "LAT", "CRIM", "ZN", "INDUS", 
  "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT"]

var = boston[features]

var_train, var_test, price_train, price_test = train_test_split(var, price, 
test_size=0.4, random_state=1)

regressor = RandomForestRegressor(n_estimators= 400, min_samples_split= 2, 
                                  min_samples_leaf= 1, max_features= 'sqrt', 
                                  max_depth= None, bootstrap= False)

regressor.fit(var_train, price_train)

price_predict = regressor.predict(var_test)


plt.plot(price_test, price_predict, "x")
plt.plot([0,50] , [0, 50])
plt.title("AI accuracy on the test data")
plt.xlabel("Actual Price ($1,000s)")
plt.ylabel("Predicted Price ($1,000s)")
plt.grid()
plt.xlim([0,50])
plt.ylim([0,50])
plt.show()

plt.plot(price_train, regressor.predict(var_train), "x")
plt.plot([0,50] , [0, 50])
plt.title("AI accuracy on the training data")
plt.xlabel("Actual Price ($1,000s)")
plt.ylabel("Predicted Price ($1,000s)")
plt.xlim([0,50])
plt.ylim([0,50])
plt.show()
plt.grid()


    






#n_estimators= 400, min_samples_split= 2, min_samples_leaf= 1, max_features= 'sqrt', max_depth= None, bootstrap= False




#{'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}