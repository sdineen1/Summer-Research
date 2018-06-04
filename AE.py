
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
for n_in, n_out in zip (encoding_layers[:-1], encoding_layers[1:]):
    print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
    
    ae= Sequential()
    encoder= Sequential([Dense(, input_dim=n_in, activation='sigmoid', activity_regularizer=regularizers.l1(10e-5))])
    decoder= Sequential([Dense(n_out, activation='sigmoid', actvity_regularizer=regularizers.li(10e-5))])
    ae.add(encoder, decoder, output_reconstruction=False, tie_weights=True)
    
    ae.compile(optimizer='adam', loss='mean_squared_error')
    ae.fit(X_train, X_train, batch_size=batch_size, nb_epoch=epochs)
    
    trained_encoders.append(ae.layers[0].encoder)
    


















'''encoding_dim=10 

shape = X_train_Scaled.shape[1]
input_shape=Input(shape=(shape,))

enocded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_shape)
decoded = Dense(shape, activity_regularizer=regularizers.l1(10e-5))(encoded)

encoder=(input_shape, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layer[-1]
decoder=Model(encoded_input, decoder_layer(encoded_input))

aeL1=Model(input_shape, decoded)'''



