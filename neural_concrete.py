# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 22:43:57 2020

@author: rahul
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

Concrete = pd.read_csv("E:\\Data Science\\Data Sheet\\concrete.csv")

model_data = Sequential()
model_data.add(Dense(50,input_dim=8,activation="relu"))
model_data.add(Dense(40,activation="relu"))
model_data.add(Dense(20,activation="relu"))
model_data.add(Dense(1,kernel_initializer="normal"))
model_data.compile(loss="mean_squared_error",optimizer = "adam", metrics = ['mse'])

column_names = list(Concrete.columns)
predictor = column_names[0:8]
target = column_names[8]

first_model = model_data
first_model.fit(np.array(Concrete[predictor]),np.array(Concrete[target]),epochs=10)
pred_train = first_model.predict(np.array(Concrete[predictor]))
pred_train = pd.Series([i[0]for i in pred_train])
rmse_values = np.sqrt(np.mean((pred_train-Concrete[target])**2))

import matplotlib.pyplot as plt
plt.plot(pred_train,Concrete[target],"bo")
np.corrcoef(pred_train,Concrete[target])

from ann_visualizer.visualize import ann_viz

ann_viz(first_model, title="Neural Network")

