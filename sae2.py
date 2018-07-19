#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:56:01 2018

@author: shaydineen
"""

import pandas as pd
import numpy as np
import math as math

dataset = pd.read_csv('ETF_Opportunity_Set/SPY/spy.csv')
data = dataset.iloc[:,1:].values
data = np.array(data)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

data_scaled = sc.fit_transform(data)

from sklearn.model_selection import train_test_split
x_train_scaled, x_test_scaled = train_test_split(data_scaled, test_size = .2)

from keras.models import Model, Input
from keras.layers import Dense

input_shape = Input(shape = (data_scaled.shape[1], ))

encoded_dim = Dense(units = 12, activation= 'sigmoid')(input_shape)
decoded_dim = Dense(units = data_scaled.shape[1], activation='sigmoid')(encoded_dim)

ae = Model(input_shape, decoded_dim)
ae.compile(optimizer = 'adam', loss = 'mean_squared_error')
ae.fit(x = x_train_scaled, y = x_train_scaled, epochs = 100, batch_size = 32 )

predict = ae.predict(x_test_scaled)

import keras.backend as K

hidden_layer_function = K.function([ae.layers[0].input],
                                 [ae.layers[1].output])

hidden_layer_output = hidden_layer_function([x_test_scaled])[0]

output = hidden_layer_function([data_scaled])[0]

filepath = 'ETF_Opportunity_Set/SPY/spy_ae.csv'

df = pd.DataFrame(output)

df.to_csv(filepath, index=False)
