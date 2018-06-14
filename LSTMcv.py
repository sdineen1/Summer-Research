#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:52:34 2018

@author: shaydineen
"""
# =============================================================================
# Importing the dataset
# =============================================================================
import numpy as np
import pandas as pd

#dataset = pd.read_excel('SAEoutputSP500.xlsx')
dataset = pd.read_csv('WaveletOutput.csv')
#dataset = dataset.iloc[:,2:].values

dataset = np.array(dataset)




# =============================================================================
# Scaling the dataset
# =============================================================================
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

dataset_scaled = sc.fit_transform(dataset)

# =============================================================================
# Building the Model
# =============================================================================

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def compile_regressor(units, shape, dropout_rate, optim):
    
    regressor = Sequential()
    
    regressor.add(LSTM(units=units, return_sequences=True, input_shape = (shape.shape[1], shape.shape[2])))
    regressor.add(Dropout(dropout_rate))
    
    regressor.add(LSTM(units = units, return_sequences = True))
    regressor.add(Dropout(dropout_rate))

    regressor.add(LSTM(units = units, return_sequences = True))
    regressor.add(Dropout(dropout_rate))
    
    regressor.add(LSTM(units = units, return_sequences = True))
    regressor.add(Dropout(dropout_rate))

    regressor.add(LSTM(units = units))
    regressor.add(Dropout(dropout_rate))
    
    regressor.add(Dense(1))
    
    regressor.compile(optimizer=optim, loss = 'mean_squared_error')
    
    
    return regressor

def train_regressor(compiled_regressor , X_train, y_train, epochs, batch_size):
    
        regressor=compiled_regressor
        regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
        return regressor


time_steps = 90
features = int(dataset_scaled.shape[1])
X = []
y = []

#So this sets up our correct X and y vectors
for i in range(time_steps, int(len(dataset_scaled))):
    X.append(dataset[i-time_steps:i,0:features]) #dataset_scaled
    y.append(dataset[i,0]) #dataset_scaled

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], features))

training_set_size = int(len(X)*.25)
test_size = int(.25*training_set_size)

correlations = []

'''for i in range (0, int(len(X))-(training_set_size+test_size),test_size):
    
    X_train = X[i:i+training_set_size, :, :]
    y_train = y[i:i+training_set_size]
    X_test = X[i+training_set_size:i+training_set_size+test_size, :, :]
    y_test = y[i+training_set_size: i + training_set_size + test_size]


    shape_train = (X_train[0], features)
    
    regressor = compile_regressor(units = 200, shape = X_train, dropout_rate = .2, optim = 'adam')
    regressor = train_regressor(compiled_regressor = regressor, X_train = X_train, y_train = y_train, epochs = 100 , batch_size = 60)
    
    predicted = regressor.predict(X_test)
    predicted = predicted[:,0]
    correl = np.corrcoef(predicted, y_test)
    correlations.append(correl[0,1])


filepath = 'LSTMcorrelations.csv'
df = pd.DataFrame(correlations)
df.to_csv(filepath, index=False)'''
X_train_size = int(len(X)*.8)

X_train = X[0:X_train_size,1:]
y_train = y[0:X_train_size]
X_test = X[X_train_size:,1:]
y_test = y[X_train_size:]

regressor = compile_regressor(units = 200, shape = X_train, dropout_rate = .2, optim = 'adam')
regressor = train_regressor(compiled_regressor = regressor, X_train = X_train, y_train = y_train, epochs = 100 , batch_size = 60)

predicted = regressor.predict(X_test)
predicted = predicted[:,0]
correl = np.corrcoef(predicted, y_test)


filepath = 'WLSTM.csv'
df = pd.DataFrame(predicted)
df.to_csv(filepath, index=False)

        