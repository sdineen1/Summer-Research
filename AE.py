#python <name of file>
#screen program useful for long experiments
# =============================================================================
# Step 1- Importing the Data
# =============================================================================

import pandas as pd
import numpy as np

SAP = pd.read_excel('SP500.xlsx')
data = SAP.iloc[:, 2:].values #The first two values are date and time

#DO I INCLUDE CLOSING PRICE IN THE ENCODING? I am afraid that by including closing price in the SAE, the closing price will be encoded in the output 
#^^I.e by encoding the closing price, would the LSTM then have acess to the closing price that it is trying to predict?
    #In regards to the above, I don't believe the LSTM will have acess to that information.  Each input vector is the information that happened at day t.  The info at day t does not include t+1 closing price, only t's closing price.  
    #When we set up the LSTM, the decoded stock price is appended to its respective day t. The lstm then uses the info at day t to predict day t+1.  Day t's encoded representation does not contain info on t+1 closing price

# =============================================================================
# Step 2- Feature Scaling
# ============================================================================= 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(data, test_size=.4, random_state=None )

#Normalizing the data
sc = MinMaxScaler(feature_range=(0,1))

X_train_Scaled = sc.fit_transform(X_train)
X_test_Scaled = sc.transform(X_test)
data_scaled = sc.transform(data)




# =============================================================================
# Step 3A- Training the First AE layer
# =============================================================================

from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from math import sqrt

#Initialize encoding dimension set to 10 for compression ratio of 2.2
encoding_dim=10 
shape = X_train_Scaled.shape[1]
input_shape=Input(shape=(shape,))

#Creating the Encoder and Decoder Objects
#In the paper, they never specifically state that they used sparsity, however; in the overview of their SAE model, they discuss a sparseity parameter but neglect to mention what type of 'sparsity' they used or the values of this parameter 
#Big takeaway is that they must have used a sparse AE because the AE would just cheat in its four stacked layers of 10 nodes each if it wasn't sparse. "The goal is to have the least amount of neurons fire in any given instance" -Ian Goodfellow
encoded = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(input_shape) #L1 vs l2 vs l1_l2?
decoded = Dense(shape, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(encoded)

#Building the AE
ae = Model(input_shape, decoded)
#In the paper, they built their own optimizer, however; after doing my own reseach, it makes more sense to use an optimizer that uses an adaptive learning rate method
ae.compile(optimizer='adam', loss='mean_squared_error')

filepath_ae1 = 'ae1.hdf5'
checkpoint_ae1 = ModelCheckpoint(filepath =filepath_ae1, monitor = 'loss', save_best_only=True, verbose=1, mode='min') #correct setting mode to min for mean_squared_error
ae.fit(X_train_Scaled, X_train_Scaled, epochs=200, batch_size=32, callbacks=[checkpoint_ae1]) #monitor=loss?

#Building a function that returns the raw outputs of the FIRST hidden layer
get_hidden_layer_output = K.function([ae.layers[0].input], 
                                     [ae.layers[1].output])

#Raw output from the first hidden layer (used to train second hidden layer)
hidden_layer_output1 = get_hidden_layer_output([X_train_Scaled])[0] #raw outputs from the hidden layers used to train the second hidden layer

#Finding the RMSE on the traing set
ae1_train_predict = ae.predict(X_train_Scaled)
RMSE_train= sqrt(mean_squared_error(X_train_Scaled, ae1_train_predict))

#Finding the RMSE for the test set (slightly useless as we aren't even using this)
ae1_test_predict= ae.predict(X_test_Scaled)
RMSE_test = sqrt(mean_squared_error(X_test_Scaled, ae1_test_predict))


# =============================================================================
# Steb 3B- Training the Second AE layer
# =============================================================================

hidden_shape_raw = hidden_layer_output1.shape[1] #the shape of the output layer from the first AE that will be used as the input for the second AE
input_shape2 = Input(shape=(hidden_shape_raw,))
encoded2 = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(input_shape2)
decoded2 = Dense(hidden_shape_raw, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(encoded2)

ae2 = Model(input_shape2, decoded2)
ae2.compile(optimizer='adam', loss='mean_squared_error')

filepath_ae2 = 'ae2.hdf5'
checkpoint_ae2 = ModelCheckpoint(filepath=filepath_ae2, monitor= 'loss', verbose=1, mode='min', save_best_only=True)
ae2.fit(hidden_layer_output1, hidden_layer_output1, epochs=200, batch_size=32, callbacks=[checkpoint_ae2]) #using the output of the first AE to train the second AE
get_second_hidden_layer_output= K.function([ae2.layers[0].input], 
                                           [ae2.layers[1].output])
hidden_layer_output2 = get_second_hidden_layer_output([hidden_layer_output1])[0]

#Calculating the RMSE for the SECOND hidden layer 
ae2_train_predict = ae2.predict(hidden_layer_output1)

RMSE_train2 = sqrt(mean_squared_error(hidden_layer_output1, ae2_train_predict))


# =============================================================================
# Step 3C- Training the Third AE
# =============================================================================

hidden_raw_shape3 = hidden_layer_output2.shape[1]
input_shape3 = Input(shape=(hidden_raw_shape3, ))

encoded3 = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(input_shape3)
decoded3 = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(encoded3)

ae3 = Model(input_shape3, decoded3)
ae3.compile(optimizer='adam', loss='mean_squared_error')

filepath_ae3= 'ae3.hdf5'
checkpoint_ae3 = ModelCheckpoint(filepath=filepath_ae3, verbose=1, monitor='loss', mode='min', save_best_only=True)
ae3.fit(hidden_layer_output2, hidden_layer_output2, epochs=200, batch_size=32, callbacks= [checkpoint_ae3]) #using the output from the second AE to train the third AE
get_third_layer_output = K.function([ae3.layers[0].input], 
                                    [ae3.layers[1].output])
hidden_layer_output3= get_third_layer_output([hidden_layer_output2])[0]

ae3_train_predict = ae3.predict(hidden_layer_output2)
RMSE_train3= sqrt(mean_squared_error(hidden_layer_output2, ae2_train_predict))


# =============================================================================
# Step 3D- Training the Fourth AE
# =============================================================================

hidden_raw_shape4 = hidden_layer_output3.shape[1]
input_shape4 = Input(shape=(hidden_raw_shape4, ))

encoded4 = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(input_shape4)
decoded4 = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(encoded4)

ae4 = Model(input_shape4, decoded4)
ae4.compile(optimizer='adam', loss='mean_squared_error')

filepath_ae4 = 'ae4.hdf5'
checkpoint_ae4 = ModelCheckpoint(filepath=filepath_ae4, verbose=2, save_best_only=True, monitor='loss', mode='min')
ae4.fit(hidden_layer_output3, hidden_layer_output3, epochs=200, batch_size=32, callbacks=[checkpoint_ae4]) #using the output from the third AE to train the fourth AE
get_fourth_layer_output = K.function([ae4.layers[0].input], 
                                     [ae4.layers[1].output])

hidden_layer_output4 = get_fourth_layer_output([hidden_layer_output3])[0]

ae4_train_predict = ae4.predict(hidden_layer_output3)
RMSE_train4 = sqrt(mean_squared_error(hidden_layer_output3, ae4_train_predict))


# =============================================================================
# Step 4- Putting the SAE Together
# =============================================================================

#Weight matrices from trained AEs
ae1_weights = ae.layers[1].get_weights() #22 nodes to 10 nodes
ae2_weights = ae2.layers[1].get_weights() #10 nodes to 10 nodes
ae3_weights = ae3.layers[1].get_weights() #10 nodes to 10 nodes
ae4_weights = ae4.layers[1].get_weights() #10 nodes to 10 nodes

#these are the wieghts that connect the last hidden layer to the output layer
ae4_weights_output = ae4.layers[2].get_weights()

#Creating a model that pools all of learned weights together 
#Wasn't sure if I should add the outputs of the fourth sae to the model
sae_input = Input(shape=(X_train.shape[1], ))
encoder1 = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(sae_input)
encoder2 = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(encoder1)
encoder3 = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(encoder2)
encoder4 = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(encoder3)
#the same reguklarizer sparsity parameter nneds to be chanfe for easch lasyer 

sae = Model(sae_input, encoder4)
sae.layers[1].set_weights(ae1_weights)
sae.layers[2].set_weights(ae2_weights)
sae.layers[3].set_weights(ae3_weights)
sae.layers[4].set_weights(ae4_weights)


# =============================================================================
# Step 5- Saving the Model
# =============================================================================




# =============================================================================
# Step 6- Fine Tuning the SAE
# =============================================================================

#The paper makes no refrence if they trained the compiled SAE after initializing the weights at each layer although this is common practice 


encode_X_train = sae.predict(X_train)
new_data = sae.predict(data_scaled)


df = pd.DataFrame(new_data)

filepath = 'SAEoutputSP500.xlsx'
df.to_excel(filepath, index=False)






