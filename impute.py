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

s = boston.isnull()#.sum()
correlation_matrix = boston.corr().round(2)

print(s)