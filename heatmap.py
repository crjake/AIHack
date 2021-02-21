#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import os

directory = os.path.dirname(os.path.realpath(__file__))
boston = pd.read_csv('boston_corrected.csv')
price = boston["MEDV"]

sns.heatmap(boston.corr())

