# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:03:35 2020

@author: rahul
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

forestfires = pd.read_csv("E:\\Data Science\\Data Sheet\\forestfires.csv")

model_data = Sequential()
model_data.add(Dense(50,input_dim=8,activation="relu"))
model_data.add(Dense(40,activation="relu"))
model_data.add(Dense(20,activation="relu"))
model_data.add(Dense(1,kernel_initializer="normal"))
model_data.compile(loss="mean_squared_error",optimizer = "adam", metrics = ['mse'])

column_names = list(forestfires.columns)
predictor = column_names[2:10]
target = column_names[10]

first_model = model_data
first_model.fit(np.array(forestfires[predictor]),np.array(forestfires[target]),epochs=10)
pred_train = first_model.predict(np.array(forestfires[predictor]))
pred_train = pd.Series([i[0]for i in pred_train])
rmse_values = np.sqrt(np.mean((pred_train-forestfires[target])**2))

import matplotlib.pyplot as plt
plt.plot(pred_train,forestfires[target],"bo")
np.corrcoef(pred_train,forestfires[target])

from ann_visualizer.visualize import ann_viz
ann_viz(first_model, title="Neural Network")
