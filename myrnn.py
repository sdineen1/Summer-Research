#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:56:44 2018

@author: shaydineen
"""

#Part 1- Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')  #this turns the data into a dateframe.  Keras only accepts inputs to Neural Nets as array so we will have to change it
training_set = dataset_train.iloc[:,1:2].values #we need to do 1:2 even though 2 is icluded because we eventually want to create a numpy array column and this imsures us that we will get that. .values turns it from a dataframe to a numpy array

#Feature Scaling
#normailisation is preffered to standardization when there is a sigmoid function in your output layer
from sklearn.preprocessing import MinMaxScaler #MinMaxScalar is the normalization class

sc= MinMaxScaler(feature_range=(0,1)) #we are going to use the deafault arguements. All new scaled stock prices will be between 0 and 1 
training_set_scaled=sc.fit_transform(training_set) #in fit_transform, fit means that it will get the minimum and maximum values in the training set and then transform means that it will apply the normalization to each datapoint 

#Creating a datastructure with 60 time steps and 1 output. A timestep means that at time t, the RNN is going to look at the stock prices at t-60 and then it will try and predict the stock price at t+1
#for each financial day, X_train will contain the 60 previous financial records before that day and y_train will contain the finacial stock price for the next day
X_train =[]
y_train=[]

#since we are finding t-60 through t, we can only start at our 60th day
for i in range(60,1258): #1257 is the number we go up to, but we say go up to 1258 because the upper bound is excluded
     X_train.append(training_set_scaled[i-60:i,0])#there is only one column in training_set_scaled so we put 0
     y_train.append(training_set_scaled[i,0]) #it is i because X_train takes all the stock prices up to i but excludes i so technically i is the next day

#X_train and y_train are currently lists so we need to convert them in numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)   

#reshaping the dimension that we are adding is the unit, that is the number of predictiors that we can use to predict what we want.  In this case the google stock price at t+1
X_train= np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #first arguement is the numpy array that we want to reshpae. The second arguement is the new shape we want our numpy array to have. For the second arguement we use parenthesis because we have to input three values
#^^^For the second arguement is is reccommended that we go to https://keras.io/layers/recurrent/ and see exactly the input shape that the lastm expects. The first arguement is batch_size which correspends to the total number of stock prices from 2012 to 2016 that we are training on, the second arguement is the number of timesteps (60), and the third arguement, input_dim, corresponds to the additional indicators that we are putting in
#X_train.shape[0], X_train.shape[1] are merely the dimensions of our dataset, we could have put in 1198 and 60 respectivley instead. 1 indicates the number of indicators that we are using and in this case it is just 1

#building the RNN

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#Initalizing the RNN
regressor= Sequential()

#Adding the first layer and adding some dropout regularization
regressor.add(LSTM(units= 50, return_sequences= True, input_shape=(X_train.shape[1], 1)  ))  #The first arguement is the number of units or the number of memory cells that we want our LSTM to have.  The second arguement is return_sequenses which we have to set to true because we are building a stacked LSTM.  Whenever we add an additionasl layer we have to set this second arguement to true
#The third arguement is input shape which is exactly the shape of X_train that we creating in the last step of the data preprossesing.  For this last arguement we don't have to include all three dimensions.  We can exclude the first one because the LSTM already takes it into accound.  We just jave to define the number of time-steps and the number of indicators that we are using
#Even though our lstm will have a high dimensionality because it is stacked, we can continue to increase the dimensionality of our model by having a large number of neurons in each of the layers.  We need a high dimensionality so that our LSTM is able to find patterns within the data

#Adding Dropout
regressor.add(Dropout(.2)) #.2 is the dropoout

#Adding the second layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(.2))

#Adding the third layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(.2))

#Adding the fourth layer
regressor.add(LSTM(units=50)) #Since return_sequence default is False we don;'t  need to set it true because our fourth layer is our last layer
regressor.add(Dropout(.2))

#Adding the output layer
regressor.add(Dense(units=1))

#Compiling the model
regressor.compile(optimizer='adam', loss='mean_squared_error') #even though rmsprop is better for RNNs, the instructor found better results using the adam optimizer

#Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

#Part 3- Making the Predictions
dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values

#Getting the predicted stock price of 2017

#to be able to get the 60 previous days of stock prices we will need both the test and the training set
dataset_total= pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0) #contains both the traning and test set. concat is the function that allows us to concatinate (combine) our datasets
#the first arguement is the datframes that we want to concatinate. The dataset_train and the dataset_test contain all the variables but we just want the open values so we type them in the square bracket
#second arguement is the axis that we want to combine on.  We can either combine on the columns or the rows. Since we want to combine the rows we will do axis=0 (verticle axis is 0 and horizontal axis is 1)

inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values #len(dataset_total) will give us the stock price in our total dataset and subtracting len(dataset_test) will give us the stock price on our first day of our test set and then subtracting 60 (our time step) will give us the lower bound that we need
#the colon just means it will go to the end (the upper bound)^^^
#.values to make it a numpy array

#since we didn't use the iloc function from pandas, our data may not be the right shape so we need to use the reshape function from numpy to do this
inputs=inputs.reshape(-1,1) #to reshape it properly we use -1 and +1

#we need to scale the inputs now. WE ONY SCLAE THE INPUTS NOT THE TEST VALUES. WE KEEP THE TEST VALUES THE WAY THEY ARE
inputs=sc.transform(inputs) #we dont do fit_transform because our scale object (sc) is already fit for the training set and we want to use that same fit (because the lstm needs the same scaling to properly work) for the test set 

#Now we need to put it into a dimension that the lstm can understand
X_test=[]
#we dont need a y_test anymore because we aren't doing anymore training so we don't need the groiund truth variables
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
    
X_test= np.array(X_test)

X_test= np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

y_pred=regressor.predict(X_test)

deScaled_y_pred= sc.inverse_transform(y_pred) #de scales it back into readable values

#visualizing the results
plt.plot(real_stock_price, color='black', label='Real Google Stock Price')
plt.plot(deScaled_y_pred, color='red', label='Predicted Google Stock Price')
plt.title('Google Stock Price Predictio]n') #the title of our chart
plt.xlabel('Time') #label of x axis
plt.ylabel('Stock Price') #label of y axis
plt.legend() #we don't use any input because this is just to include the legends in the chart
plt.show() #this displays the graph

#ACCORDING TO THE MATHEMATICAL BROWNIAN MOTION, FUTURE STOCK PRICES ARE INDEPENDENT FROM THE PAST

#For Regression, the way to evaluate the model performance is with a metric called RMSE (Root Mean Squared Error). It is calculated as the root of the mean of the squared differences between the predictions and the real values.
#consider dividing this RMSE by the range of the Google Stock Price values of January 2017 (that is around 800) to get a relative error, as opposed to an absolute error. It is more relevant since for example if you get an RMSE of 50, then this error would be very big if the stock price values ranged around 100, but it would be very small if the stock price values ranged around 10000.
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

#Parameter Tuning for Regression is the same as Parameter Tuning for Classification which you learned in Part 1 - Artificial Neural Networks, the only difference is that you have to replace:
#scoring = 'accuracy'  
#by:
#scoring = 'neg_mean_squared_error' 
#in the GridSearchCV class parameters.









#WILL DO LATER
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:3].values

X_train =[]
y_train=[]

for i in range(60,1258):
    X_train.append(training_set[i-60:i,0:2])

X_train=np.array(X_train)

#reshaping the multi feature X_train
X_train=np.reshape(X_train,(X_train.shape[0], X_train.shape[1],2))










import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
 
# Importing Training Set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
 
cols = list(dataset_train)[1:3]
 
#Preprocess data for training by removing all commas
 
dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0,len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(",","")
 
dataset_train = dataset_train.astype(float)
 
 
training_set = dataset_train.as_matrix() # Using multiple predictors.
 

 
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
 
n_future = 1  # Number of days you want to predict into the future
n_past = 60  # Number of past days you want to use to predict the future
 
for i in range(n_past, len(training_set) - n_future + 1):
    X_train.append(training_set[i - n_past:i, 0:2])

 
X_train = np.array(X_train)
 
    

