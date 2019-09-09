import numpy as np # linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model
from keras.layers import *

# Import the news csv as df
sp500 = pd.read_csv("/kaggle/input/thesissp500lda-t1-titles-v3/sp5001_t1_v2/sp5001_t1_v2.csv")
sp500.fillna(value="", inplace=True)
sp500["Percent_Change"] = sp500["Daily_Change"] / sp500["Open"]
us_yield = pd.read_csv("/kaggle/input/thesis-us-yield/us_yield1/us_yield1.csv")
vix = pd.read_csv("/kaggle/input/thesis-vix/vix3/vix3.csv")
print(vix.head())

# Split the dataset into training and test sets
train_size = int(len(sp500) * 0.9)
sp500_train = sp500[:train_size]
sp500_train.reset_index(drop=True,inplace=True)
sp500_test = sp500[train_size:]
sp500_test.reset_index(drop=True,inplace=True)

# Define the data input for the training set and test set
X_train = sp500_train["Titles"]
X_test = sp500_test["Titles"]

# Import the vocab csv as df
vocab = pd.read_csv("/kaggle/input/thesis-ner-sentiment-vocab/vocab.csv")

# Convert the vocab df to a dictionary
keys = list(vocab["Vocab"])
values = list(vocab.index)
dictionary = dict(zip(keys, values))

# Featurise the data input with TF-IDF
tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary=dictionary, ngram_range=(1,5))
tfidf.fit(X_train)
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)
print(X_train.shape)
print(X_test.shape)

# Standardise the data input by the mean and standard deviation of its training set
scaler = StandardScaler()
scaler.fit(X_train.toarray())
X_train_stand = scaler.transform(X_train.toarray())
X_test_stand = scaler.transform(X_test.toarray())
print(X_train_stand.shape)
print(X_test_stand.shape)

# Perform PCA on the training set and apply it on the test set
pca = PCA(n_components=700)
pca.fit(X_train_stand)
X_train_pca = pca.transform(X_train_stand)
X_test_pca = pca.transform(X_test_stand)
print(X_train_pca.shape)
print(X_test_pca.shape)

# Calculate the total explained variance ratio
explained_variance = np.sum(pca.explained_variance_ratio_)
print(explained_variance)

# Stack the training data and the test data back together along the batch dimension
X = np.concatenate((X_train_pca, X_test_pca), axis=0)
print(X.shape)

# Define the 1 day data input
X_1d = X[37:]
# Define the label for the dataset
y = np.array(sp500["Daily_Direction"])[37:]

# Define the price change history array
X_history = np.zeros((len(X_1d),37))
for i in range(1300):
    X_history[i,:] = np.array(sp500["Percent_Change"])[i:(i+37)]
print(X_history.shape)

# Define the price change history array for US 2 year yield
X_2yr = np.zeros((len(X_1d),37))
for i in range(1300):
    X_2yr[i,:] = np.array(us_yield["2 Yr diff"] / us_yield["prior 2 Yr"])[i:(i+37)]

# Define the price change history array for US 10 year yield
X_10yr = np.zeros((len(X_1d),37))
for i in range(1300):
    X_10yr[i,:] = np.array(us_yield["10 Yr diff"] / us_yield["prior 10 Yr"])[i:(i+37)]

# Define the price change history array for VIX
X_vix = np.zeros((len(X_1d),37))
for i in range(1300):
    X_vix[i,:] = np.array(vix["Daily_Change"] / vix["VIX Open"])[i:(i+37)]
    
# Define the 1 week data input in the form of 3-d tensor
X_1w = np.zeros((len(X_1d),7,700))
for i in range(30, 1330):
    X_1w[(i-30),:,:] = X[i:(i+7)] 
    
# Define the 1 month data input in the form of 3-d tensor
X_1m = np.zeros((len(X_1d),30,700))
for i in range(1300):
    X_1m[i,:,:] = X[i:(i+30)]
    
# Expand X_1w and X_1m to 4-d tensors
X_1w_4d = np.expand_dims(X_1w, axis=-1)
X_1m_4d = np.expand_dims(X_1m, axis=-1)
print(X_1w_4d.shape)

# Define hyperparameters for the dense layers
learning_rate = 0.001
hidden_dim = 100
hidden_dim_1 = 100
hidden_dim_2 = 50
hidden_dim_3 = 25

# Define the CNN model
def model_cnn(d_input, w_input, m_input, h_input, us2_input, us10_input, vix_input):
    # Define the 3 data input: 1d, 1w and 1m
    inp_d = Input(shape=(d_input[1],))
    inp_w = Input(shape=(w_input[1], w_input[2], w_input[3]))
    inp_m = Input(shape=(m_input[1], m_input[2], w_input[3]))
    inp_h = Input(shape=(h_input[1],))
    inp_us2 = Input(shape=(us2_input[1],))
    inp_us10 = Input(shape=(us10_input[1],))
    inp_vix = Input(shape=(vix_input[1],))
    
    # Define the first layer of the NN
    w = Conv2D(filters=1, kernel_size=(w_input[1],1), activation="relu")(inp_w)
    m = Conv2D(filters=1, kernel_size=(m_input[1],1), activation="relu")(inp_m)

    # Flatten both output from CNN
    w = Flatten()(w)
    #w = Dropout(0,5)(w)
    m = Flatten()(m)
    #m = Dropout(0.5)(m)
    
    # Define the second layer NN
    d = Dense(hidden_dim, activation="relu")(inp_d)
    w = Dense(hidden_dim, activation="relu")(w)
    m = Dense(hidden_dim, activation="relu")(m)
    h = Dense(5, activation="relu")(inp_h)
    us2 = Dense(5, activation="relu")(inp_us2)
    us10 = Dense(5, activation="relu")(inp_us10)
    vix = Dense(5, activation="relu")(inp_vix)
    
    # Concatenate the output from the first dense layer
    x = concatenate([d, w, m, h, us2, us10, vix], axis=-1)
    # Add Dropout layer
    x = Dropout(0.5)(x)
    
    # Feed the concatenation into the next layers
    x = Dense(hidden_dim_1, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(hidden_dim_2, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(hidden_dim_3, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[inp_d, inp_w, inp_m, inp_h, inp_us2, inp_us10, inp_vix], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
    return model
