#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:23:50 2018

@author: shaydineen
"""

import pandas as pd
import numpy as np


SAP = pd.read_excel('Data/UltimateDataSet2000.xlsx')
data = SAP.iloc[:, 1:].values #The first two values are date and time
data = data[0:2516,:]


# =============================================================================
# Step 2- Feature Scaling
# ============================================================================= 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

X_train, X_test = train_test_split(data, test_size=.4, random_state=None )

#Normalizing the data
'''sc = MinMaxScaler(feature_range=(0,1)) #not going to scale the data

X_train_Scaled = sc.fit_transform(X_train)
X_test_Scaled = sc.transform(X_test)
data_scaled = sc.transform(data)'''

# =============================================================================
# Building the SAE
# =============================================================================
from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers 
from keras import backend as K

def RMSE(predict, ground_truth):
    
    rmse = sqrt(mean_squared_error(predict, ground_truth))
    return rmse


#nneds to be fixed. Error is in using output for training
def build_SAE(layers, data, activation, regularizer, batch_size, epochs, optim):
    
    #List that will keep track of the RMSE for each AE
    rmse = []
    
    #Building the first AE
    #The shape of the first AE corresponds to layers @ 0
    shape1 = layers[0]
    input_shape1 = Input(shape = (shape1, ))
    
    
    encoded1 = Dense(units = layers[1], activation = activation, activity_regularizer = regularizers.l1(.05) )(input_shape1)
    decoded1 = Dense(units = shape1, activation = activation, activity_regularizer = regularizers.l1(.05))(encoded1)
    
    ae1 = Model(input_shape1, decoded1)
    
    ae1.compile(optimizer = optim, loss = 'mean_squared_error')
    ae1.fit(x = data, y = data, batch_size = batch_size, epochs = epochs)
    
    hidden_output1 = K.function([ae1.layers[0].input], [ae1.layers[1].output])
    output1 = hidden_output1([data])[0]
    
    rmse1 = RMSE(ae1.predict(data), data)
    rmse.append(rmse1)
    
    #Building the second AE
    shape2 = layers[1]
    input_shape2 = Input(shape= (shape2, ))
    
    encoded2 = Dense(units = layers[2], activation = activation, activity_regularizer = regularizers.l1(.025))(input_shape2)
    decoded2 = Dense(units = shape2, activation = activation, activity_regularizer = regularizers.l1(.025))(encoded2)
    
    ae2 = Model(input_shape2, decoded2)
    
    ae2.compile(optimizer = optim, loss = 'mean_squared_error')
    ae2.fit(x = output1, y = output1, batch_size = batch_size, epochs = epochs)
    
    hidden_output2 = K.function([ae2.layers[0].input], [ae2.layers[1].output])
    output2 = hidden_output2([output1])[0]
    
    rmse2 = RMSE(ae2.predict(output1), output1)
    rmse.append(rmse2)
    
    #Building the third AE
    shape3 = layers[2]
    input_shape3 = Input(shape= (shape3, ))
    
    encoded3 = Dense(units = layers[3], activation = activation, activity_regularizer = regularizers.l1(.0125))(input_shape3)
    decoded3 = Dense(units = shape3, activation = activation, activity_regularizer = regularizers.l1(.0125))(encoded3)
    
    ae3 = Model(input_shape3, decoded3)
    
    ae3.compile(optimizer = optim, loss = 'mean_squared_error')
    ae3.fit(x = output2, y = output2, batch_size = batch_size, epochs = epochs)
    
    hidden_output3 = K.function([ae3.layers[0].input], [ae3.layers[1].output])
    output3 = hidden_output3([output2])[0]
    
    rmse3 = RMSE(ae3.predict(output2), output2)
    rmse.append(rmse3)
    
    #Building the fourth AE
    shape4 = layers[3]
    input_shape4 = Input(shape= (shape4, ))
    
    encoded4 = Dense(units = layers[4], activation = activation, activity_regularizer = regularizers.l1(.0075))(input_shape4)
    decoded4 = Dense(units = shape4, activation = activation, activity_regularizer = regularizers.l1(.0075))(encoded4)
    
    ae4 = Model(input_shape4, decoded4)
    
    ae4.compile(optimizer = optim, loss = 'mean_squared_error')
    ae4.fit(x = output3, y = output3, batch_size = batch_size, epochs = epochs)

    
    hidden_output4 = K.function([ae4.layers[0].input], [ae4.layers[1].output])
    output4 = hidden_output4([output3])[0]
    
    rmse4 = RMSE(ae4.predict(output3), output3)
    rmse.append(rmse4)
    
    ae1_weights = ae1.layers[1].get_weights() 
    ae2_weights = ae2.layers[1].get_weights() 
    ae3_weights = ae3.layers[1].get_weights() 
    ae4_weights = ae4.layers[1].get_weights() 

    #these are the wieghts that connect the last hidden layer to the output layer
    ae4_weights_output = ae4.layers[2].get_weights()

    #Creating a model that pools all of learned weights together 
    #Wasn't sure if I should add the outputs of the fourth sae to the model
    sae_input = Input(shape=(layers[0], )) #
    sae_en1 = Dense(units = layers[1], activation=activation, activity_regularizer=regularizers.l1(.05))(sae_input)
    sae_en2 = Dense(units = layers[2], activation=activation, activity_regularizer=regularizers.l1(.025))(sae_en1)
    sae_en3 = Dense(units = layers[3], activation=activation, activity_regularizer=regularizers.l1(.0125))(sae_en2)
    sae_en4 = Dense(units = layers[4], activation=activation, activity_regularizer=regularizers.l1(.0075))(sae_en3)
    #the same reguklarizer sparsity parameter nneds to be chanfe for easch lasyer 

    sae = Model(sae_input, sae_en1)
    sae.layers[1].set_weights(ae1_weights)
    sae.layers[2].set_weights(ae2_weights)
    sae.layers[3].set_weights(ae3_weights)
    sae.layers[4].set_weights(ae4_weights)
    
    
    rmse = np.array(rmse)
    
    
    return sae#, rmse
    
 #Should I train the AE on the whole dataset or use a train and test set   


'''
shape1 = layers[0]
input_shape1 = Input(shape = (shape1, ))
    
    
encoded1 = Dense(units = layers[1], activation = 'sigmoid', activity_regularizer = regularizers.l1(.05) )(input_shape1)
decoded1 = Dense(units = shape1, activation = 'sigmoid', activity_regularizer = regularizers.l1(.05))(encoded1)
    
ae1 = Model(input_shape1, decoded1)
    
ae1.compile(optimizer = 'adam', loss = 'mean_squared_error')
ae1.fit(x = data_scaled, y = data_scaled, batch_size = 60, epochs = 100)
    
hidden_output1 = K.function([ae1.layers[0].input], [ae1.layers[1].output])
output1 = hidden_output1([data])[0]
ae1_weights = ae1.layers[1].get_weights()

sae_input = Input(shape = (shape1, ))
en = Dense(units = layers[1], activation = 'sigmoid', activity_regularizer = regularizers.l1(.05))(sae_input)
sae = Model(sae_input, en)
sae.set_weights(ae1_weights)

predict = sae.predict(data_scaled)

filepath = 'SAEoutput.csv'
#filepath2= 'SAE_RMSE.csv'
df = pd.DataFrame(predict)
#df2 = pd.DataFrame(rmse)
#df2.to_csv(filepath2, index=False)
df.to_csv(filepath, index=False)'''


def SAE_one_layer(layers, rho, data, optim, epochs, batch_size):
    
    error = []
    
    for j in range(len(rho)):
        for i in range(len(layers)):
            input_shape = Input(shape = (data.shape[1], ))
            encoding = Dense(units = layers[i], activation ='sigmoid', activity_regularizer = regularizers.l1(rho[j]))(input_shape)
            decoding = Dense(units = data.shape[1], activation = 'sigmoid', activity_regularizer = regularizers.l1(rho[j]))(encoding)
            
            ae = Model(input_shape, decoding)
            ae.compile(optimizer = optim, loss = 'mean_squared_error')
            ae.fit(x = data, y = data, epochs = epochs , batch_size = batch_size)
            
            predict = ae.predict(data)
            reconstruction_error= mean_squared_error(data, predict)
            
            error.append([reconstruction_error, layers[i], rho[j]])

    return error


layers = [18, 22, 26, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72]
rho = [.5, .25, .125 , .1, .075, .05, .0375, .01875, .009375]


#Creating the input shape

def one_hidden_layer_AE(data, X_train):
    
    input_shape = Input(shape = (data.shape[1], ))
    
    
    encoding1 = Dense(units = 7, activation = 'sigmoid')(input_shape)
    decoding1 = Dense(units = data.shape[1], activation = 'sigmoid')(encoding1)
    
    sae1 = Model(input_shape, decoding1)
    sae1.compile(optimizer = 'adam', loss = 'mean_squared_error')
    sae1.fit(x = X_train, y = X_train, batch_size = 60, epochs = 500)
    
    hidden_output_func = K.function([sae1.layers[0].input], [sae1.layers[1].output])
    output = hidden_output_func([data])[0]
    
    return output, hidden_output_func
    #sae1_output = sc.inverse_transform(predict)
    
#def two_hidden_layer_AE(data, X_train):
    
input_shape = Input(shape = (data.shape[1], ))

encoding1 = Dense(units = 9, activation = 'sigmoid')(input_shape)
decoding1 = Dense(units  = data.shape[1], activation = 'sigmoid')(encoding1)

ae1 = Model(input_shape, decoding1)
ae1.compile(optimizer = 'adam', loss = 'mean_squared_error')
ae1.fit(x = X_train, y = X_train, batch_size = 60, epochs = 100)

hidden_output1 = K.function([ae1.layers[0].input], 
                            [ae1.layers[1].output])
output1 = hidden_output1([X_train])[0]

input_second_hlayer = Input(shape = (output1.shape[1], ))
encoding2 = Dense(units = 7, activation = 'sigmoid')(input_second_hlayer)
decoding2 = Dense(units = output1.shape[1], activation = 'sigmoid')(encoding2)

ae2 = Model(input_second_hlayer, decoding2)
ae2.compile(optimizer = 'adam', loss = 'mean_squared_error')
ae2.fit(x = output1, y = output1, batch_size = 60, epochs = 100)

ae1_weights = ae1.layers[1].get_weights()
ae2_weights = ae2.layers[1].get_weights()

sae2_input = Input(shape = (data.shape[1], ))
encoding2 = Dense(units = 9, activation = 'sigmoid')(sae2_input)
decoding2 = Dense(units = 7, activation = 'sigmoid')(encoding2)

sae2 = Model(sae2_input, decoding2)

sae2.layers[1].set_weights(ae1_weights)
sae2.layers[2].set_weights(ae2_weights)

output = sae2.predict(data)
    
    #return output, sae2

#output, sae = two_hidden_layer_AE(data = data, X_train = X_train)

#compressed_data = hidden_output_func([data])[0]
outputs = sae2.predict(data)
filepath = 'SAEoutput.csv'
df = pd.DataFrame(outputs)
df.to_csv(filepath, index=False)
            
            

            
           
