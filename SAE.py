#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:23:50 2018

@author: shaydineen
"""

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

SAP = pd.read_excel('SP500.xlsx')
data = SAP.iloc[:, 2:].values #The first two values are date and time



# =============================================================================
# Step 2- Feature Scaling
# ============================================================================= 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

X_train, X_test = train_test_split(data, test_size=.4, random_state=None )

#Normalizing the data
sc = MinMaxScaler(feature_range=(0,1))

X_train_Scaled = sc.fit_transform(X_train)
X_test_Scaled = sc.transform(X_test)
data_scaled = sc.transform(data)

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
    sae_en2 = Dense(units = layers[2], activation=activation, activity_regularizer=regularizers.l1(.05))(sae_en1)
    sae_en3 = Dense(units = layers[3], activation=activation, activity_regularizer=regularizers.l1(.05))(sae_en2)
    sae_en4 = Dense(units = layers[4], activation=activation, activity_regularizer=regularizers.l1(.05))(sae_en3)
    #the same reguklarizer sparsity parameter nneds to be chanfe for easch lasyer 

    sae = Model(sae_input, sae_en4)
    sae.layers[1].set_weights(ae1_weights)
    sae.layers[2].set_weights(ae2_weights)
    sae.layers[3].set_weights(ae3_weights)
    sae.layers[4].set_weights(ae4_weights)
    
    
    rmse = np.array(rmse)
    
    
    return sae, rmse
    
 #Should I train the AE on the whole dataset or use a train and test set   
    
layers = [data_scaled.shape[1], 22, 26, 30, 34]
regularizer1 = np.random.uniform(.05,10e-4)
regularizer2 = np.random.uniform(.05,10e-4)
regularizer3 = np.random.uniform(.05,10e-4)
regularizer4 = np.random.uniform(.05,10e-4)


regularizers_input = [.05, 
                .025, 
                .0125, 
                .05]

sae, rmse = build_SAE(layers=layers, data=data_scaled, activation = 'sigmoid', regularizer = regularizers_input, batch_size=30, epochs=750, optim='adam' )

predict = sae.predict(data_scaled)

filepath = 'SAEoutput.csv'
filepath2= 'SAE_RMSE.csv'
df = pd.DataFrame(predict)
df2 = pd.DataFrame(rmse)
df2.to_csv(filepath2, index=False)
df.to_csv(filepath, index=False)
