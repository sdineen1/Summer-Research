#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:56:01 2018

@author: shaydineen
"""

import pandas as pd
import numpy as np
import math as math

dataset = pd.read_csv('ETF_Opportunity_Set/SameTrain/iyr.csv')
data = dataset.iloc[:,1:].values
data = np.array(data)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

data_scaled = sc.fit_transform(data)

#from sklearn.model_selection import train_test_split
#x_train_scaled, x_test_scaled = train_test_split(data_scaled, test_size = .2)

from keras.models import Model, Input
from keras.layers import Dense
import keras.backend as K


def build_sae(x_train_scaled, X_train_and_test):
    #x_train scaled should be the data that the LSTM will be trained and x-test should be the data that the LSTM is tested on
    input_shape = Input(shape = (X_train_and_test.shape[1], ))
    
    encoded_dim = Dense(units = 9, activation= 'sigmoid')(input_shape)
    decoded_dim = Dense(units = data_scaled.shape[1], activation='sigmoid')(encoded_dim)
    
    ae = Model(input_shape, decoded_dim)
    ae.compile(optimizer = 'adam', loss = 'mean_squared_error')
    ae.fit(x = x_train_scaled, y = x_train_scaled, epochs = 100, batch_size = 32 )
        
    
    hidden_layer_function = K.function([ae.layers[0].input],
                                     [ae.layers[1].output])
    
    hidden_layer_output = hidden_layer_function([x_train_scaled])[0]
    
    input_shape2 = Input(shape = (hidden_layer_output.shape[1], ))
    encoded2 = Dense(units = 7, activation = 'sigmoid')(input_shape2)
    decoded2 = Dense(units = hidden_layer_output.shape[1], activation = 'sigmoid')(encoded2)
    
    ae2 = Model(input_shape2, decoded2)
    ae2.compile(optimizer = 'adam', loss = 'mean_squared_error')
    ae2.fit(x = hidden_layer_output, y = hidden_layer_output, epochs = 100, batch_size = 32)
    
    sae_input = Input(shape = (X_train_and_test.shape[1], ))
    sae_first_encoded = Dense(units = 9, activation = 'sigmoid')(sae_input)
    sae_second_encoded = Dense(units = 7, activation = 'sigmoid')(sae_first_encoded)
    sae = Model(sae_input, sae_second_encoded)
    
    ae_weights = ae.layers[1].get_weights()
    ae2_weights = ae2.layers[1].get_weights()
    
    sae.layers[1].set_weights(ae_weights)
    sae.layers[2].set_weights(ae2_weights)
    
    output = sae.predict(X_train_and_test)
    
    return output

'''def SAE_train_test_vectors(data_scaled, timesteps, train_size):
    #sets up the proper X vectors to be fed into the SAE'''
      

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


def sliding_window(data_scaled, train_size, test_size, time_steps, data):
    #train_size and test_size are ints
    
    predictions = []
    actual_price = []
    run = 1 
    #This is the sliding-window for-loop.  The for-loop starts at 0 and goes to legnth of X minus the size of the additive total of the size of the trainign and sets
    #i is then incremented by the size of the test set
    for i in range (0, int(len(data_scaled))-(train_size+test_size+time_steps),test_size):
        
        
        print(run)
        sae_train = data_scaled[i:i+train_size+time_steps, :] #the training data for the SAE
        
        x = data_scaled[i:i+train_size+test_size+time_steps, :]
        
        output = build_sae(x_train_scaled = sae_train, X_train_and_test = x)
        appended_closing_prices = np.zeros(shape = (output.shape[0],8))
        appended_closing_prices[:,0] = data[i:i+train_size+test_size+time_steps,0] #changing the last thing from 0 to 1 appends the opening price
        appended_closing_prices[:,1:8]= output[:,:]
        scaler = MinMaxScaler()
        appended_closing_prices_scaled = scaler.fit_transform(appended_closing_prices)
        
        X, y = X_y_vectors(time_steps = time_steps, data_scaled = appended_closing_prices_scaled, num_feature = 8)
        
        X_train = X[:train_size, :, :]
        y_train = y[:train_size]
        X_test = X[train_size:train_size+test_size, :, :]
        y_test = y[train_size: train_size + test_size]

    
        regressor = compile_regressor(units = 200, shape = X_train, dropout_rate = .2, optim = 'adam')
        regressor = train_regressor(compiled_regressor = regressor, X_train = X_train, y_train = y_train, epochs = 100 , batch_size = 60)
    
        predicted = regressor.predict(X_test)
        predict_dataset_like = np.zeros(shape=(len(predicted), appended_closing_prices_scaled.shape[1]))
        predict_dataset_like[:,0] = predicted[:,0]
        real_predicted = scaler.inverse_transform(predict_dataset_like)[:,0]

        #predicted = predicted[:,0]
        predictions.append([real_predicted])
        
        y = appended_closing_prices[time_steps+train_size:train_size+test_size+time_steps,0] #0 here indicates whatever value we appened to the first columns whether that be the closing or opening price etc.
        
        actual_price.append(y)
        
        run = run+1
        

    return predictions, actual_price

training_set_size = 2000
test_set_size = 250
time_steps = 90

y_hat, y = sliding_window(data_scaled, train_size = training_set_size, test_size = test_set_size, time_steps = time_steps, data = data)
y_hat, y = np.array(y_hat), np.array(y)
y_hat, y = np.reshape(y_hat, newshape = (-1, 1)), np.reshape(y, newshape = (-1 , 1))

#predict_dataset_like = np.zeros(shape=(len(y_hat), dataset.shape[1]))
#predict_dataset_like[:,0] = y_hat[:,0]
#real_predicted = sc.inverse_transform(predict_dataset_like)[:,0]

#actual_prices = dataset[len(dataset)-len(real_predicted):,0]

y = np.column_stack((y_hat, y))

filepath = 'ETF_Opportunity_Set/LSTMoutput.csv'
df = pd.DataFrame(y)
df.to_csv(filepath, index=False)
    
