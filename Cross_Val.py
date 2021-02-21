#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import pandas as pd
import plotparams
from sklearn import linear_model, tree, ensemble
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


directory = os.path.dirname(os.path.realpath(__file__))
boston = pd.read_csv('boston_corrected.csv')
price = boston["MEDV"]
features = ["TOWNNO", "LON", "LAT", "CRIM", "ZN", "INDUS", 
  "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT"]

var = boston[features]


# tree number
base_n_estimators = 100
base_max_features = 5
test_size = 0.4


kf = KFold(n_splits=5, shuffle=True, random_state=42)


# var_train, var_test, price_train, price_test = train_test_split(var, price, 
#         test_size=test_size)

kf.split(var, price)



def rmse(score):
    rmse = np.sqrt(-score)
    print(f'rmse= {"{:.2f}".format(rmse)}')

# estimators = [50, 100, 150, 200, 250, 300, 350]

# print(var_train)
# print(price_train)

# for count in estimators:
#     score = cross_val_score(ensemble.RandomForestRegressor(n_estimators= count, random_state= 42), var, price, cv= kf, scoring="neg_mean_squared_error")
#     print(f'For estimators: {count}')
#     rmse(score.mean())
    
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}    
    

# for count in n_estimators:
#     score = cross_val_score(ensemble.RandomForestRegressor(n_estimators= count, random_state= 42), var, price, cv= kf, scoring="neg_mean_squared_error")
#     print(f'number of estimators: {count}')
#     rmse(score.mean())
    
# for max_features_i in max_features:
#     score = cross_val_score(ensemble.RandomForestRegressor(random_state= 42, max_features = max_features_i), var, price, cv= kf, scoring="neg_mean_squared_error")
#     print(f'max features: {max_features_i}')
#     rmse(score.mean())

rf = RandomForestRegressor()

var_train, var_test, price_train, price_test = train_test_split(var, price, 
        test_size=test_size, random_state=5)

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(var_train, price_train)

# print(rf_random.best_params_)
    
# for max_depth_i in max_depth:
#     score = cross_val_score(ensemble.RandomForestRegressor(random_state= 42, max_depth = max_depth_i), var, price, cv= kf, scoring="neg_mean_squared_error")
#     print(f'For max_depth: {max_depth_i}')
#     rmse(score.mean())    

# for min_samples_split_i in n_estimators:
#     score = cross_val_score(ensemble.RandomForestRegressor(random_state= 42, min_samples_split = min_samples_split_i), var, price, cv= kf, scoring="neg_mean_squared_error")
#     print(f'For minimum samples split: {min_samples_split_i}')
#     rmse(score.mean())     

# for min_samples_leaf_i in min_samples_leaf:
#     score = cross_val_score(ensemble.RandomForestRegressor(random_state= 42, min_samples_leaf = min_samples_leaf_i), var, price, cv= kf, scoring="neg_mean_squared_error")
#     print(f'For minimum samples leaf: {min_samples_leaf_i}')
#     rmse(score.mean())     
    
# for bootstrap_i in bootstrap:
#     score = cross_val_score(ensemble.RandomForestRegressor(random_state= 42, bootstrap = bootstrap_i), var, price, cv= kf, scoring="neg_mean_squared_error")
#     print(f'For bootstrap: {bootstrap_i}')
#     rmse(score.mean())      

# {'n_estimators': 400,
#  'min_samples_split': 2,
#  'min_samples_leaf': 1,
#  'max_features': 'sqrt',
#  'max_depth': None,
#  'bootstrap': False}

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = RandomForestRegressor(n_estimators = base_n_estimators, max_features = base_max_features, random_state = 42)
base_model.fit(var_train, price_train)
base_accuracy = evaluate(base_model, var_train, price_train)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, var_train, price_train)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

    
    