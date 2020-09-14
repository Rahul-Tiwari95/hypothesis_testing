# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:39:12 2020

@author: rahul
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

Startups = pd.read_csv("E:\\Data Science\\Data Sheet\\50_Startups.csv")


model = Sequential()
model.add(Dense(12,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="mean_squared_error",optimizer = "adam", metrics = ['mse'])

column_names = list(Startups.columns)
predictor = column_names[0:3]
target = column_names[3]

first_model = model
first_model.fit(np.array(Startups[predictor]),np.array(Startups[target]),epochs=10)
pred_train = first_model.predict(np.array(Startups[predictor]))
pred_train = pd.Series([i[0]for i in pred_train])
rmse_values = np.sqrt(np.mean((pred_train-Startups[target])**2))

import matplotlib.pyplot as plt
plt.plot(pred_train,Startups[target],"bo")
np.corrcoef(pred_train,Startups[target])

from ann_visualizer.visualize import ann_viz
ann_viz(first_model, title="Neural Network")

