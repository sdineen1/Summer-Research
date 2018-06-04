
# =============================================================================
# Step 1- Importing the Data
# =============================================================================

import pandas as pd
import numpy as np

SAP = pd.read_excel('SP500.xlsx')
data = np.array(SAP)

# =============================================================================
# Step 2- Feature Scaling
# =============================================================================

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(data, test_size=.4, random_state=None )

sc = MinMaxScaler(feature_range=(0,1))

X_train_Scaled = sc.fit_transform(X_train)
X_test_Scaled = sc.transform(X_test)

#X_train_Scaled = np.reshape()


# =============================================================================
# Step 3A- Defing the AE Class
# =============================================================================

















# =============================================================================
# Step 3A- Building the First AE Layer
# =============================================================================

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers

encoding_layers= [22,10,10,10,10]
batch_size = 32
nb_classes = 10
epochs=200

trained_encoders = []
X_train_tmp = X_train_Scaled
for n_in, n_out in zip (encoding_layers[:-1], encoding_layers[1:]):
    print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
    
    ae= Sequential()
    encoder= Sequential([Dense(n_out, input_dim=n_in, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))])
    decoder= Sequential([Dense(n_in, activation='sigmoid', actvity_regularizer=regularizers.l1(10e-5))])
    ae.add(encoder, decoder, output_reconstruction=False, tie_weights=True)
    
    ae.compile(optimizer='adam', loss='mean_squared_error')
    ae.fit(X_train, X_train, batch_size=batch_size, nb_epoch=epochs)
    
    trained_encoders.append(ae.layers[0].encoder)
    
    X_train_tmp = ae.predict(X_train_tmp)


















from keras.layers import Dense, Input
from keras.models import Model
from keras import regularizers

encoding_dim=10 
shape = X_train_Scaled.shape[1]
input_shape=Input(shape=(shape,))

encoded = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(input_shape)
decoded = Dense(shape, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))(encoded)

ae = Model(input_shape, decoded)
ae.compile(optimizer='adam', loss='mean_squared_error')
ae.fit(X_train_Scaled, X_train_Scaled, epochs=200, batch_size=21)

ae1_predicted= ae.predict(X_test_Scaled)

RMSE = 0
for i in range(len(X_test)):
    diff = (ae1_predicted[i]-X_test[i])
    square_diff = np.power(diff,2)
    RMSE += np.sqrt(square_diff)
RMSE_mean = RMSE/len(X_test)
encoded2 = ae.layers[0]

'''encoder=(input_shape, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layer[-1]
decoder=Model(encoded_input, decoder_layer(encoded_input))

aeL1=Model(input_shape, decoded)'''



