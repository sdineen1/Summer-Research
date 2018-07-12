
# =============================================================================
# Step 1- Importing the Data 
# =============================================================================

import numpy as np
import pandas as pd
import math as math

#data = pd.read_excel('Data/UltimateDataSet2000.xlsx')

#data = data.iloc[:,2:].values

#Turning the pandas dataframe into numpy array
data = pd.read_csv('eem_wavelet.csv', engine= 'python', encoding = 'ASCII')
data = data.iloc[:,1:].values
data = np.array(data)
data = data[0:2516,:]


#In the paper, Bao, Yue, and Rao s training set consisted of 80% of the data while the CV and test sets each consisted of 10% of the data
#training_size = int(math.floor(len(data)*.5)) #going to use .6 because i trained the SAE on 60% of the data
#cv_size = math.ceil(len(data)*.1)
#test_size = math.ceil(len(data)*.1)
training_size = 2061

training_set = data[:training_size] #does it make a difference if I put an 



#don;t need to preprocess after the autoencoder 

# =============================================================================
# Step 2- Data Preprocessing
# =============================================================================

from sklearn.preprocessing import MinMaxScaler

#Normalizing the data as opposed to Standardizing it
#The TYPE of the data preporocessing is supposed to have a neglible difference on performance 
sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []

time_steps = 90 #arbitraily set the # of timesteps to 60.  The paper does not specify the # number of timesteps that they used, however; 
#based on  the way they trained their model the # number of time_steps that they used is between 0 and 720 (not very helpful)

for i in range (time_steps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-time_steps:i, :])   #training_set_scaled
    y_train.append(training_set_scaled[i, 0])                   #training_set_scaled

X_train , y_train = np.array(X_train), np.array(y_train) #Transforiming the list objects into numpy arrays 

X_train = np.reshape(X_train , (X_train.shape[0], X_train.shape[1], X_train.shape[2])) #Reshaping into a 3rd degree tensor that the Keras LSTM expects


# =============================================================================
# Step 3- Building the LSTM
# =============================================================================

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

dropout_rate =.2 #Arbitrarily set dropout rate to .2.

#In the paper, they specify that they used a LSTM with 5 hidden layers but failed to mention the number of units in each hidden layer
regressor = Sequential()

#Adding the first LSTM layer
regressor.add(LSTM(units = 200, return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(dropout_rate))

#Adding the second LSTM layer
regressor.add(LSTM(units=200, return_sequences=True))
regressor.add(Dropout(dropout_rate))

#Adding the third LSTM layer
regressor.add(LSTM(units=200, return_sequences=True))
regressor.add(Dropout(dropout_rate))

#Adding the fourth LSTM layer
regressor.add(LSTM(units=200, return_sequences=True))
regressor.add(Dropout(dropout_rate))

#Adding the 5th LSTM layer
regressor.add(LSTM(units=200, return_sequences=True))
regressor.add(Dropout(dropout_rate))

#EXPERIMENTAL 6th LSTM LAYER
regressor.add(LSTM(units=200))
regressor.add(Dropout(dropout_rate))

#Output layer
regressor.add(Dense(units=1))


# =============================================================================
# Step 4- Compiling and Fitting the Model
# =============================================================================

#In the paper they specified that they used 5000 epochs and a batch size of 60
epochs = 100
batch_size = 60
learning_rate = .05

#I used the adam optimizer but in the paper they used some other type of optimizer
regressor.compile(optimizer= 'adam', loss= 'mean_squared_error')

regressor.fit(X_train, y_train, epochs=epochs, batch_size= batch_size )

# =============================================================================
# Step 5- Predictions
# =============================================================================

test_size = int(len(data)-training_size)
inputs = data[len(data)-test_size-time_steps:]
inputs = sc.transform(inputs)
X_test = []
for i in range (time_steps , len(inputs)):
    X_test.append(inputs[i-time_steps:i, :])
    
X_test = np.array(X_test)
y_test = inputs[time_steps:len(inputs),0]

y_pred = regressor.predict(X_test)

# =============================================================================
# Step 6- Exporting the files 
# =============================================================================

y_pred = np.array(y_pred)

predict_dataset_like = np.zeros(shape=(len(y_pred), data.shape[1]))
predict_dataset_like[:,0] = y_pred[:,0]
real_predicted = sc.inverse_transform(predict_dataset_like)[:,0]

actual_prices = data[len(data)-len(real_predicted):,0]


y_pred = y_pred[:,0]
y = np.column_stack((real_predicted, actual_prices))

y = np.column_stack((y_pred, y_test, real_predicted, actual_prices))
filepath = 'LSTMoutput.csv'

df = pd.DataFrame(y)

df.to_csv(filepath, index=False)



