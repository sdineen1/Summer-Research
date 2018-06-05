
# =============================================================================
# Step 1- Importing the Data 
# =============================================================================

import numpy as np
import pandas as pd
import math as math

data = pd.read_excel('SAEoutputSP500.xlsx')
data = np.array(data)

training_size = math.floor(len(data)*.8)
cv_size = math.ceil(len(data)*.1)
test_size = math.ceil(len(data)*.1)

training_set = data[:training_size, :]
cv_set = data[training_size:training_size+cv_size, :]
test_set = data[training_size+cv_size:,:]


# =============================================================================
# Step 2- Data Preprocessing
# =============================================================================

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []

time_steps = 60

for i in range (time_steps, len(training_set_scaled)):
    X_train.append(training_set[i-time_steps:i, 0:11])
    y_train.append(training_set[i, 0])

X_train , y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train , (X_train.shape[0], X_train.shape[1], 11))


# =============================================================================
# Step 3- Building the LSTM
# =============================================================================

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

dropout_rate =.2

regressor = Sequential()

regressor.add(LSTM(units = 100, return_sequences=True, input_shape = (X_train.shape[1], 11)))
regressor.add(Dropout(dropout_rate))

regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(dropout_rate))

regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(dropout_rate))

regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(dropout_rate))

regressor.add(LSTM(units=100))
regressor.add(Dropout(dropout_rate))

regressor.add(Dense(units=1))


# =============================================================================
# Step 4- Compiling and Fitting the Model
# =============================================================================

epochs = 500
batch_size = 60

regressor.compile(optimizer= 'adam', loss= 'mean_squared_error')

regressor.fit(X_train, y_train, epochs=epochs, batch_size= batch_size )



