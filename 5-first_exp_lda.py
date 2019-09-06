# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras import optimizers
from keras.callbacks import *

# Tensorflow
import tensorflow as tf

# SKLearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
import scipy as sp

# Import the csv as df
sp500_dict = {}
for i in range(1,4):
    sp500_dict[i] = pd.read_csv("/kaggle/input/sp500_lda/sp500_lda2_{}.csv".format(i))

# Split the dataset into training and test sets
sp500_train_dict = {}
sp500_test_dict = {}
for i in range(1,4):
    train_size = int(len(sp500_dict[i]) * 0.9)
    sp500_train_dict[i] = sp500_dict[i][:train_size]
    sp500_train_dict[i].reset_index(drop=True,inplace=True)
    sp500_test_dict[i] = sp500_dict[i][train_size:]
    sp500_test_dict[i].reset_index(drop=True,inplace=True)
    
# Define the data input for the training set and test set
X_train_dict = {}
X_test_dict = {}
for i in range(1,4):
    X_train_dict[i] = sp500_train_dict[i]["Titles"]
    X_test_dict[i] = sp500_test_dict[i]["Titles"]
    
# Featurise the data input with TF-IDF
for i in range(1,4):
    tfidf = TfidfVectorizer(max_features = 10000, sublinear_tf=True)
    tfidf.fit(X_train_dict[i])
    X_train_dict[i] = tfidf.transform(X_train_dict[i])
    X_test_dict[i] = tfidf.transform(X_test_dict[i])
    
# Standardise the data input by the mean and standard deviation of its training set
for i in range(1,4):
    scaler = StandardScaler()
    scaler.fit(X_train_dict[i].toarray())
    X_train_dict[i] = scaler.transform(X_train_dict[i].toarray())
    X_test_dict[i] = scaler.transform(X_test_dict[i].toarray())
    
# Perform PCA on the training set and apply it on the test set
for i in range(1,4):
    pca = PCA(n_components=1000)
    pca.fit(X_train_dict[i])
    X_train_dict[i] = pca.transform(X_train_dict[i])
    X_test_dict[i] = pca.transform(X_test_dict[i])
    print(np.sum(pca.explained_variance_ratio_))
print(X_train_dict[2].shape)
print(X_test_dict[2].shape)

# Define the labels for both training and testing
y_train_dict = {}
y_test_dict = {}
for i in range(1,4):
    y_train_dict[i] = sp500_train_dict[i]["Daily_Direction"]
    y_test_dict[i] = sp500_test_dict[i]["Daily_Direction"]
    
def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

# Define the matthews correlation for self-usage
def matthews_correlation_self(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)

    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + 1e-7)
    
# Define hyperparameters for the dense layers
learning_rate = 0.001
hidden_dim = 100
hidden_dim_2 = 50
hidden_dim_3 = 25

# Define the model architecture for TFIDF vectors
def build_model():
    model = Sequential()
    model.add(Dense(hidden_dim, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dim_2, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dim_3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
    return model
    
# Split the training set further into training and valdiation set
X_train_fin_dict = {}
X_val_dict = {}
y_train_fin_dict = {}
y_val_dict = {}
for i in range(1,4):
    train_size = X_train_dict[i].shape[0] - X_test_dict[i].shape[0]
    X_train_fin_dict[i] = X_train_dict[i][:train_size]
    X_val_dict[i] = X_train_dict[i][train_size:]
    y_train_fin_dict[i] = np.array(y_train_dict[i][:train_size])
    y_val_dict[i] = np.array(y_train_dict[i][train_size:])
    
# Define the number of runs for each scenario
R = 30
# Define the number of epoch for each run
E = 50

# Define history matrices to save down both training and validation results
history_train_dict = {}
history_val_dict = {}
for i in range(1,4):
    history_train_dict[i] = np.zeros((E, R))
    history_val_dict[i] = np.zeros((E, R))
    
# Define result matrices to hold the test results
MCC_dict = {}
ACC_dict = {}
for i in range(1,4):
    MCC_dict[i] = np.zeros(R)
    ACC_dict[i] = np.zeros(R)
    
# Kickstart the runs
for t in range(1,4):
    print("Start Topic = {}".format(t))
    # Define the data input, validation set and test set
    X_train_fin = X_train_fin_dict[t]
    y_train_fin = y_train_fin_dict[t]
    X_val = X_val_dict[t]
    y_val = y_val_dict[t]
    X_test = X_test_dict[t]
    y_test = y_test_dict[t]
    for i in range(R):
        print("Start Run = {}".format(i))
        # Instantiate a model
        K.clear_session()
        model = build_model()
        
        # Define callback
        callbacks = [ModelCheckpoint("weights_dense_{}.h5".format(i), monitor='val_matthews_correlation', 
                                      save_best_only=True, save_weights_only=True, mode='max')]
    
        # Start training
        model.fit(X_train_fin, y_train_fin, batch_size=50, epochs=E, validation_data=[X_val, y_val], 
                  callbacks=callbacks, shuffle=False, verbose=True)
    
        # Save down the training history
        history_train_dict[t][:,i] = model.history.history["matthews_correlation"]
        history_val_dict[t][:,i] = model.history.history["val_matthews_correlation"]
    
        # Load the best weights saved by the checkpoint
        model.load_weights('weights_dense_{}.h5'.format(i))
    
        # Use the model for prediction on the test set
        predict_test = np.reshape((model.predict(X_test) >= 0.5).astype(int), (X_test.shape[0],))
        MCC_dict[t][i] = matthews_correlation_self(y_test, predict_test)
        ACC_dict[t][i] = np.sum(y_test == predict_test) / len(y_test)
        print(MCC_dict[t][i])
        
# Export the results as csv
for i in range(1,4):
    np.savetxt("history_train_lda_{}.csv".format(i), history_train_dict[i], delimiter=",")
    np.savetxt("history_val_lda_{}.csv".format(i), history_val_dict[i], delimiter=",")
    np.savetxt("MCC_lda_{}.csv".format(i), MCC_dict[i], delimiter=",")
    np.savetxt("ACC_lda_{}.csv".format(i), ACC_dict[i], delimiter=",")
    
# Define the labels for both training and testing
y_train_dict = {}
y_test_dict = {}
for i in range(1,4):
    y_train_dict[i] = sp500_train_dict[i]["3_Class"]
    y_train_dict[i] = np.eye(3)[y_train_dict[i]]
    y_test_dict[i] = sp500_test_dict[i]["3_Class"]
    y_test_dict[i] = np.eye(3)[y_test_dict[i]]
print(y_train_dict[1].shape)
print(y_test_dict[1].shape)

def matthews_correlation(y_true, y_pred):
    # Calculate the MCC for multi-class classification
    #Calculate the total number of samples
    s = K.sum(y_true)
    #Calculate the number of times each class occurred
    t = K.sum(y_true, axis=0)
    #Allocate the predicted class based on the max probability
    predict_argmax = K.argmax(y_pred, axis=1)
    #Create a one hot matrix based on the predicted class
    encoded = K.one_hot(predict_argmax, 3)
    #Calculate the number of times each class was predicted
    p = K.sum(encoded, axis=0)
    #Calculate the product of the actual class vector and predicted class vector
    actual_predict = t * p
    #Square the actual class vector and the predicted class vector
    t_sq = K.square(t)
    p_sq = K.square(p)
    #Calculate the total no. of samples correctly predicted
    c = K.sum(y_true * encoded)
    #Calculate the numerator and the denominator
    numerator = c * s - K.sum(actual_predict)
    denominator = K.sqrt((K.square(s) - K.sum(p_sq)) * (K.square(s) - K.sum(t_sq)))
    # Calculate the multi-class MCC
    MCC = numerator / (denominator + K.epsilon())
    return MCC
    
# Define the model architecture for TFIDF vectors
def build_model():
    model = Sequential()
    model.add(Dense(hidden_dim, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dim_2, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dim_3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[matthews_correlation])
    return model
    
# Split the training set further into training and valdiation set
X_train_fin_dict = {}
X_val_dict = {}
y_train_fin_dict = {}
y_val_dict = {}
for i in range(1,4):
    train_size = X_train_dict[i].shape[0] - X_test_dict[i].shape[0]
    X_train_fin_dict[i] = X_train_dict[i][:train_size]
    X_val_dict[i] = X_train_dict[i][train_size:]
    y_train_fin_dict[i] = np.array(y_train_dict[i][:train_size])
    y_val_dict[i] = np.array(y_train_dict[i][train_size:])
    
# Define the number of runs for each scenario
R = 30
# Define the number of epoch for each run
E = 50

# Define history matrices to save down both training and validation results
history_train_dict = {}
history_val_dict = {}
for i in range(1,4):
    history_train_dict[i] = np.zeros((E, R))
    history_val_dict[i] = np.zeros((E, R))
    
# Define result matrices to hold the test results
MCC_dict = {}
ACC_dict = {}
for i in range(1,4):
    MCC_dict[i] = np.zeros(R)
    ACC_dict[i] = np.zeros(R)
    
# Kickstart the runs
for t in range(1,4):
    print("Start Topic = {}".format(t))
    # Define the data input, validation set and test set
    X_train_fin = X_train_fin_dict[t]
    y_train_fin = y_train_fin_dict[t]
    X_val = X_val_dict[t]
    y_val = y_val_dict[t]
    X_test = X_test_dict[t]
    y_test = y_test_dict[t]
    for i in range(R):
        print("Start Run = {}".format(i))
        # Instantiate a model
        K.clear_session()
        model = build_model()
        
        # Define callback
        callbacks = [ModelCheckpoint("weights_dense_{}.h5".format(i), monitor='val_matthews_correlation', 
                                      save_best_only=True, save_weights_only=True, mode='max')]
    
        # Start training
        model.fit(X_train_fin, y_train_fin, batch_size=50, epochs=E, validation_data=[X_val, y_val], 
                  callbacks=callbacks, shuffle=False, verbose=True)
    
        # Save down the training history
        history_train_dict[t][:,i] = model.history.history["matthews_correlation"]
        history_val_dict[t][:,i] = model.history.history["val_matthews_correlation"]
    
        # Load the best weights saved by the checkpoint
        model.load_weights('weights_dense_{}.h5'.format(i))
    
        # Use the model for prediction on the test set
        predict_test = np.argmax(model.predict(X_test), axis=1)
        actual_test = np.argmax(y_test, axis=1)
        MCC_dict[t][i] = matthews_corrcoef(actual_test, predict_test)
        ACC_dict[t][i] = np.sum(actual_test==predict_test) / len(y_test)
        predict_test_3 = np.eye(3)[predict_test]
        print(np.sum(predict_test_3, axis=0))
        print(MCC_dict[t][i])
        
# Export the results as csv
for i in range(1,4):
    np.savetxt("history_train_lda_{}.csv".format(i), history_train_dict[i], delimiter=",")
    np.savetxt("history_val_lda_{}.csv".format(i), history_val_dict[i], delimiter=",")
    np.savetxt("MCC_lda_{}.csv".format(i), MCC_dict[i], delimiter=",")
    np.savetxt("ACC_lda_{}.csv".format(i), ACC_dict[i], delimiter=",")
