#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:32:14 2018

@author: shaydineen
"""

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr


dataset = pd.read_csv('ETF_Opportunity_Set/SameTrain/tlt.csv')
dataset = dataset.iloc[:,1:].values



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




def X_y_variable_selection(time_steps, data_scaled, num_feature):
    #assumes the variable that we are trying to predict is in the 0 index
    #index of variable is the variable that you want X_train to have
    X = []
    y = []
    
    for i in range(time_steps, int(len(data_scaled))):
        X.append(data_scaled[i-time_steps:i,1])           
        y.append(data_scaled[i,0])                           

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    
    return X, y

def scale_variable(dataset, index_of_variable):
    
    sc = MinMaxScaler(feature_range=(0, 1))
    data = np.zeros(shape = (len(dataset), 2))
    data[:,0] = dataset[:,0]
    data[:,1] = dataset[:,index_of_variable]
    
    data_scaled = sc.fit_transform(data)
    
    return data_scaled


def feature_correl(X, y, train_size, test_size):
    
    
     
    correl = []
    
    
    
    for i in range (0, 3):
        
        X_train = X[0:training_set_size, :, :]
        y_train = y[0:training_set_size]
        X_test = X[training_set_size:training_set_size+test_size, :, :]
        y_test = y[training_set_size: training_set_size + test_size]
    
        regressor = compile_regressor(units = 200, shape = X_train, dropout_rate = .2, optim = 'adam')
        regressor = train_regressor(compiled_regressor = regressor, X_train = X_train, y_train = y_train, epochs = 100 , batch_size = 60)
    
        predicted = regressor.predict(X_test)
        predicted = predicted[:,0]
        
        r = pearsonr(predicted, y_test)
        
        correl.append(r)

    return correl





time_steps = 90
features = int(dataset_scaled.shape[1])
#training_set_size = int(len(X)*.25) - correct train and test sizes used by Bao, Yue, and Rao. two years is 25% of the data
#test_size = int(.25*training_set_size) - correct train and test sizes used by Bao, Yue, and Rao. 25% of two year is 6 months of test 

training_set_size = int(len(dataset)*.80)
test_size = int(.2*training_set_size)

correlations = []

for j in range(0,dataset.shape[1]):
    #needs scaler objet in for loop
    data_scaled = scale_variable(dataset = dataset, index_of_variable = j)
    
    X, y = X_y_variable_selection(time_steps = time_steps, data_scaled = data_scaled, num_feature = features) 
    
    r = feature_correl(X = X, y = y, train_size = training_set_size, test_size = test_size)
    
    correlations.append(r)


correlations = np.hstack(correlations)
correlations = np.reshape(correlations, newshape = (3, -1))


filepath = 'ETF_Opportunity_Set/ETFCorrelations.csv'
df = pd.DataFrame(correlations)
df.to_csv(filepath, index=False)

