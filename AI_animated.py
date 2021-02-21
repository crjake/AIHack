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

from matplotlib.animation import PillowWriter
from celluloid import Camera

params = {"figure.figsize": [45,15]}
plt.rcParams.update(params)


directory = os.path.dirname(os.path.realpath(__file__))
boston = pd.read_csv('boston_corrected.csv')
columns = ["TOWNNO", "TRACT", "LON", "LAT", "MEDV", "CMEDV", "CRIM", "ZN", "INDUS", 
            "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

price = boston["MEDV"]
features = ["TOWNNO", "LON", "LAT", "CRIM", "ZN", "INDUS", 
  "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT"]

var = boston[features]

fig, axs = plt.subplots(1,3)
camera = Camera(fig)

size_list = []
r_list = []

for size in np.linspace(0.1,0.9,50):
    var_train, var_test, price_train, price_test = train_test_split(var, price, 
    test_size=size, random_state=1)
    
    regressor = RandomForestRegressor(n_estimators= 400, min_samples_split= 2, 
                                      min_samples_leaf= 1, max_features= 'sqrt', 
                                      max_depth= None, bootstrap= False)
    
    regressor.fit(var_train, price_train)
    
    price_predict = regressor.predict(var_test)
    
    r = r2_score(price_test, price_predict)
    
    size_list.append(size)
    r_list.append(r)

    
    axs[0].plot(price_test, price_predict, "x", color='r')
    axs[0].plot([0,50] , [0, 50], color='b')
    axs[0].set_title("AI accuracy on the test data")
    axs[0].set_xlabel("Actual Price ($1,000s)")
    axs[0].set_ylabel("Predicted Price ($1,000s)")
    axs[0].grid(True)

    axs[1].plot(price_train, regressor.predict(var_train), "x", color='r')
    axs[1].plot([0,50] , [0, 50], color='b')
    axs[1].set_title("AI accuracy on the training data")
    axs[1].set_xlabel("Actual Price ($1,000s)")
    axs[1].set_ylabel("Predicted Price ($1,000s)")
    axs[1].grid(True)
    
    axs[2].plot(size_list, r_list, "-x", color="g")
    axs[2].set_title("r value on test data")
    axs[2].set_xlabel("Size of test data")
    axs[2].set_ylabel("r value")
    axs[2].grid(True)
    plt.close()
    
    if size == 0.9:
        plt.savefig(directory + "r variation with sample size")

    
    
    camera.snap()

anim = camera.animate(blit=False)
pillow = PillowWriter(fps=10)
filename = directory + "\\Animation.gif" 
anim.save(filename, writer=pillow)
    






#n_estimators= 400, min_samples_split= 2, min_samples_leaf= 1, max_features= 'sqrt', max_depth= None, bootstrap= False




#{'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}