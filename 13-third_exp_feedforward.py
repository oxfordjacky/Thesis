import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import *
from keras import backend as K

# Define hyperparameters for the dense layers
learning_rate = 0.001
hidden_dim = 100
hidden_dim_2 = 50
hidden_dim_3 = 25

def build_model(title_input, text_input):
    # Define the 2 data input: title and text
    inp_title = Input(shape=(title_input[1],))
    inp_text = Input(shape=(text_input[1],))
    
    # Define the first layers of the NN
    title = Dense(hidden_dim, activation="relu")(inp_title)
    text = Dense(hidden_dim, activation="relu")(inp_text)
    
    # Define the concatenation layer
    x = concatenate([title, text], axis=-1)
    # Add Dropout
    x = Dropout(0.5)(x)
    # Define the second layers of the NN
    x = Dense(hidden_dim_2, activation="relu")(x)
    x = Dropout(0.5)(x)
    # Define the third layers of the NN
    x = Dense(hidden_dim_3, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[inp_title, inp_text], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
    return model
