from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.callbacks import *

def build_model(title_input, text_input):
    # Define the 2 data input: title and text
    inp_title = Input(shape=(title_input[1], title_input[2]))
    inp_text = Input(shape=(text_input[1], text_input[2]))
    
    # Define the LSTM layers for title input
    title = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inp_title)
    title = Attention(title_input[1])(title)
    title = Dense(hidden_dim, activation="relu")(title)
                      
    # Define the LSTM layers for text input
    text = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inp_text)
    text = Attention(text_input[1])(text)
    text = Dense(hidden_dim, activation="relu")(text)
    
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
