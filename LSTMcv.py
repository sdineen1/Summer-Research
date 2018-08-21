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

#dataset = pd.read_excel('SP500.xlsx')
#dataset = pd.read_csv('WaveletOutput.csv')
dataset = pd.read_csv('ETF_Opportunity_Set/SameTrain/spy.csv')
dataset = dataset.iloc[:,1:].values

null = pd.isnull(dataset[:,0])

#dataset = np.array(dataset)

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





#This sets up our correct X and y vectors
def X_y_vectors(time_steps, data_scaled, num_feature):
    #assumes the variable that we are trying to predict is in the 0 index
    X = []
    y = []
    
    for i in range(time_steps, int(len(data_scaled))):
        X.append(data_scaled[i-time_steps:i,0:num_feature])           
        y.append(data_scaled[i,0])                           

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], num_feature))
    
    return X, y

def X_y_variable_selection(time_steps, data_scaled, num_feature, index_of_variable):
    #assumes the variable that we are trying to predict is in the 0 index
    #index of variable is the variable that you want X_train to have
    X = []
    y = []
    
    for i in range(time_steps, int(len(data_scaled))):
        X.append(data_scaled[i-time_steps:i,index_of_variable:index_of_variable+1])           
        y.append(data_scaled[i,0])                           

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    
    return X, y


def sliding_window(X, y, train_size, test_size):
    #train_size and test_size are ints
    
    predictions = []
    actual_price = []
    
    #This is the sliding-window for-loop.  The for-loop starts at 0 and goes to legnth of X minus the size of the additive total of the size of the trainign and sets
    #i is then incremented by the size of the test set
    for i in range (0, int(len(X))-(training_set_size+test_size),test_size):
        
        print(i)
        X_train = X[i:i+training_set_size, :, :]
        y_train = y[i:i+training_set_size]
        X_test = X[i+training_set_size:i+training_set_size+test_size, :, :]
        y_test = y[i+training_set_size: i + training_set_size + test_size]

    
        regressor = compile_regressor(units = 200, shape = X_train, dropout_rate = .2, optim = 'adam')
        regressor = train_regressor(compiled_regressor = regressor, X_train = X_train, y_train = y_train, epochs = 100 , batch_size = 60)
    
        predicted = regressor.predict(X_test)
        predicted = predicted[:,0]
        predictions.append([predicted])
        
        actual_price.append(y_test)
        

    return predictions, actual_price





time_steps = 90
features = int(dataset_scaled.shape[1])
#training_set_size = int(len(X)*.25) - correct train and test sizes used by Bao, Yue, and Rao. two years is 25% of the data
#test_size = int(.25*training_set_size) - correct train and test sizes used by Bao, Yue, and Rao. 25% of two year is 6 months of test 



X, y = X_y_vectors(time_steps = time_steps, data_scaled = dataset, num_feature = features) 
#training_set_size = int(len(X)*.80)
#test_size = int(.2*training_set_size)

training_set_size = 2000
test_set_size = 250

y_hat, y = sliding_window(X = X, y = y, train_size = training_set_size, test_size = test_set_size)
y_hat, y = np.array(y_hat), np.array(y)
y_hat, y = np.reshape(y_hat, newshape = (-1, 1)), np.reshape(y, newshape = (-1 , 1))

predict_dataset_like = np.zeros(shape=(len(y_hat), dataset.shape[1]))
predict_dataset_like[:,0] = y_hat[:,0]
real_predicted = sc.inverse_transform(predict_dataset_like)[:,0]

actual_prices = y[:,0]

y = np.column_stack((real_predicted, actual_prices))

filepath = 'ETF_Opportunity_Set/LSTMoutput.csv'
df = pd.DataFrame(y)
df.to_csv(filepath, index=False)

'''X_train_size = int(len(X)*.8)

X_train = X[0:X_train_size,:]
y_train = y[0:X_train_size]
X_test = X[X_train_size:,:]
y_test = y[X_train_size:]

regressor = compile_regressor(units = 200, shape = X_train, dropout_rate = .2, optim = 'adam')
regressor = train_regressor(compiled_regressor = regressor, X_train = X_train, y_train = y_train, epochs = 150 , batch_size = 60)

predicted = regressor.predict(X_test)
predicted = predicted[:,0]
correl = np.corrcoef(predicted, y_test)


filepath = 'WLSTM.csv'
df = pd.DataFrame(predicted)
df.to_csv(filepath, index=False)'''

        