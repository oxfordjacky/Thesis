import pandas as pd
pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/"
folder = "EDA/"
file = "df_all.csv"

# Define the columns required
columns = ["url","title","section","published"]

# Import the csv with all the sections
df = pd.read_csv(directory + folder + file, usecols=columns)
print(df.head())

# Export this as csv
folder = "Data/SP500/"
df.to_csv(directory + folder + "df_title.csv", index=False)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "df_title.csv"

# Reimport the csv with all the titles only
df = pd.read_csv(directory + folder + file)

# Remove the duplicated rows of titles
title = df["title"]
mask = title.duplicated().apply(lambda x: not x)
df_net = df[mask]

# Export the latest df as csv
df_net.to_csv(directory + folder + "df_title1.csv", index=False)

from datetime import timedelta, datetime

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "df_title1.csv"

# Reimport the csv with all the titles where duplicates are removed
df = pd.read_csv(directory + folder + file)

# Set published and date columns to date format in the news dataframe
df["published"] = pd.to_datetime(df.published)

# Sort the df by published in ascending order
df.sort_values(by ="published", inplace=True)

# Define a calendar for DST from 2013 to 2019 
us_day_save = {2013: [datetime(2013,3,10,7,0,0),datetime(2013,11,3,6,0,0)],
              2014: [datetime(2014,3,9,7,0,0),datetime(2014,11,2,2,6,0)],
              2015: [datetime(2015,3,8,7,0,0),datetime(2015,11,1,2,6,0)],
              2016: [datetime(2016,3,13,7,0,0),datetime(2016,11,6,6,0,0)],
              2017: [datetime(2017,3,12,7,0,0),datetime(2017,11,5,6,0,0)],
              2018: [datetime(2018,3,11,7,0,0),datetime(2018,11,4,6,0,0)],
              2019: [datetime(2019,3,10,7,0,0),datetime(2019,11,3,6,0,0)]}


# Add an empty column which will host whether it is standard or daylight saving time
df["us_eastern_time"] = ""

# Populate this column of US Eastern time (New York)
N = len(df)
for i in range(N):
    Date = df.loc[i]["published"]
    Year = Date.year
    day_save = us_day_save[Year]
    if Date >= day_save[0] and Date <= day_save[1]:
        df["us_eastern_time"].loc[i] = df["published"].loc[i] - timedelta(hours = 4)
    else:
        df["us_eastern_time"].loc[i] = df["published"].loc[i] - timedelta(hours = 5)
    if i % 100 == 0:
        print(i)

# Inspect the df one more time
print(df.head())

# Export this df as csv
df.to_csv(directory + folder + "df_title2.csv", index = False)

import pandas as pd         
import string
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "df_title2.csv"

# Reimport the latest csv
df = pd.read_csv(directory + folder + file)

# Define the translation mapping to remove punctuation
punct = dict.fromkeys(string.punctuation, " ")
punct["“"] = " "
punct["”"] = " "
punct["‘"] = " "
punct["’"] = " "
translator = str.maketrans(punct)

# Define a set of stop words
stops = set(stopwords.words("english"))
stops.remove("not")

# Define a function to clean the text for sentiment analysis
def clean_text(my_string):
   # Expand some of the contraction words
   my_string = re.sub(r"No\.", "number", my_string)
   my_string = re.sub(r"M\&A", "Merger and Acquisition", my_string)
    
   # Convert words to lower case and split them
   my_string = my_string.lower()
   
   # Clean the text
   my_string = re.sub(r"\'s ", "  ", my_string)
   my_string = re.sub(r"\’s ", "  ", my_string)
   my_string = re.sub(r"\'ve", " have ", my_string)
   my_string = re.sub(r"n\'t", " not ", my_string)
   my_string = re.sub(r"i\'m", "i am ", my_string)
   my_string = re.sub(r"\'re", " are ", my_string)
   my_string = re.sub(r"\'d", " would ", my_string)
   my_string = re.sub(r"\'ll", " will ", my_string)
   my_string = re.sub(r"'t", " not ", my_string)
   my_string = re.sub(r"'cause", " because ", my_string)
   my_string = re.sub(r",000","000", my_string)
   my_string = re.sub(r"u\.s\.", " american ", my_string)
   my_string = re.sub(r"u\.n\.", " unitednation ", my_string)
   my_string = re.sub(r"\.\d", r"\.", my_string)
   my_string = re.sub(r"\+\d", r" positive \d", my_string)
   my_string = re.sub(r"un-", "un", my_string)
   my_string = re.sub(r"write-", "write", my_string)
   my_string = re.sub(r"over-", "over", my_string)

   # Remove punctuations
   my_string = my_string.translate(translator)
   
   # Remove stop words
   my_string = my_string.split()
   my_string = [w for w in my_string if not w in stops]
   my_string = " ".join(my_string)

   # Clean the text further
   #Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}"," ",my_string)
   #Replace k by 000
   my_string = re.sub(r"(\d+)(k)", r"\g<1> thousand", my_string)
   
   # Stemming
   stemmer = SnowballStemmer("english")
   my_string = my_string.split()
   stemmed_words = [stemmer.stem(word) for word in my_string]
   my_string = " ".join(stemmed_words)
   return my_string

# Clean the titles
df["clean_title"] = df["title"].apply(lambda x: clean_text(x))

# Calculate the length of the cleaned titles
df["title_length"] = df["clean_title"].str.split().str.len()

# Export the dataframe as csv
df.to_csv(directory + folder + "df_title3.csv", index=False)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import pandas as pd
from datetime import timedelta, datetime
import numpy as np

pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "sp500.csv"

# Import the SP500 time series
sp500 = pd.read_csv(directory + folder + file)
sp500["Date"] = pd.to_datetime(sp500.Date)

# Sort the dataframe in chronological order
sp500.sort_values(by ="Date", inplace=True)

# Plot the closing price time series 
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(sp500["Date"], sp500["Adj Close"])
ax.set_title("S&P500 Adj Close from Dec13 to Apr19", fontsize=16)
monthyearFmt = mdates.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(monthyearFmt)
ax.xaxis.set_major_locator(mdates.MonthLocator((6,12)))
plt.xticks(rotation=90)
fig.savefig(directory + folder + "sp500.png")
plt.legend()
plt.show()

# Add the daily change column
sp500["Daily_Change"] = sp500["Adj Close"] - sp500["Open"]

# Add the daily direction column
sp500["Daily_Direction"] = (sp500["Daily_Change"] >= 0) * 1

# Define a function for adding in the cut-off time given dates
def cut_off_time(Date, Hour, Minute):
    cutoff = datetime(Date.year,Date.month,Date.day,Hour,Minute)
    return cutoff

# Add in the cutoff time column
sp500["Cutoff"] = sp500["Date"].apply(lambda x: cut_off_time(x,Hour=9,Minute=25))

# Add a column to indicate if a trading day falls into training or test set
N = len(sp500)
Train = int(N * 0.9)
sp500["Split"] = ""
sp500["Split"][:Train] = "train"
sp500["Split"][Train:] = "test"

# Export this S&P 500 df as csv
sp500.to_csv(directory + folder + "sp5001.csv", index=False) 

import pandas as pd
pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"

# Import both the price csv and the title csv
df = pd.read_csv(directory + folder + "df_title3.csv")
sp500 = pd.read_csv(directory + folder + "sp5001.csv")

# Make date columns date_time format
df["us_eastern_time"] = pd.to_datetime(df.us_eastern_time)
sp500["Date"] = pd.to_datetime(sp500.Date)
sp500["Cutoff"] = pd.to_datetime(sp500.Cutoff)

# Create a new column that will contain the following trading day
df["trading_day"] = ""

# Define a mechanism to align dates of the two dataframes
outer_loop = len(sp500)
inner_loop = len(df)
#Current index in the news dataframe
current_step = 0
#Loop the price dataframe
for outer in range(outer_loop-1):
    #Current date and next date in the price dataframe
    date_current = sp500["Cutoff"].loc[outer]
    date_next = sp500["Cutoff"].loc[outer+1]
    trading_day = sp500["Date"].loc[outer+1]
    while (df["us_eastern_time"].loc[current_step] <= date_current):
        current_step += 1
    while (df["us_eastern_time"][current_step] > date_current and df["us_eastern_time"][current_step] <= date_next):
        df["trading_day"][current_step] = trading_day
        current_step += 1
        if current_step == inner_loop:
            break
    print(outer)

# Export the updated news df as csv
df.to_csv(directory + folder + "df_title4.csv", index=False) 

import pandas as pd
import numpy as np

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"

# re-import both the latest price csv and the title csv
df = pd.read_csv(directory + folder + "df_title7.csv")
sp500 = pd.read_csv(directory + folder + "sp5001.csv")

df["trading_day"] = pd.to_datetime(df.trading_day)
sp500["Date"] = pd.to_datetime(sp500.Date)

# Add 2 columns which will be populated in S&P500 df
sp500["No_of_News"] = ""
sp500["Total_Len"] = ""

# Count no. of articles within daily date buckets
N = len(sp500)
for i in range(N):
    current_date = sp500["Date"][i]
    dummy = df[df["trading_day"] == current_date] 
    sp500["No_of_News"][i] = len(dummy)
    sp500["Total_Len"][i] = np.sum(dummy["title_length"].values)
    if i % 100 == 0:
        print(i)
print(sp500.head())

# Print the statistics of total news lengths
print(sp500["Total_Len"].astype(float).describe())
print(sp500["No_of_News"].astype(float).describe())

# Export this partially populated df as csv
sp500.to_csv(directory + folder + "sp5002.csv", index=False)

# Re-import the price csv and the news csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
df = pd.read_csv(directory + folder + "df_title4.csv")
sp500 = pd.read_csv(directory + folder + "sp5002.csv")
df["trading_day"] = pd.to_datetime(df.trading_day)
sp500["Date"] = pd.to_datetime(sp500.Date)

# Remove null values in cutoff_time
df = df[df["trading_day"].isnull().apply(lambda x: not x)]
df.reset_index(drop = True, inplace = True)
print(df.head())
print(sp500.head())

# Add the dataset split in the news df
train_thres = sp500[sp500["Split"] == "train"]["Date"].iloc[-1]

df["split"] = ""
df["split"][df["trading_day"] <= train_thres] = "train"
df["split"][df["trading_day"] > train_thres] = "test"

# Export this updated df as csv
df.to_csv(directory + folder + "df_title5.csv", index=False)

# Re-import the price csv and the news csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
df = pd.read_csv(directory + folder + "df_title7.csv")
sp500 = pd.read_csv(directory + folder + "sp5002.csv")

# Set both dates columns to datetime format
df["trading_day"] = pd.to_datetime(df.trading_day)
sp500["Date"] = pd.to_datetime(sp500.Date)

# Add 2 empty columns that need to be populated in the price df
sp500["Titles"] = ""
sp500["Title_Check"] = "" 

# Populate these empty columns
outer_loop = len(sp500)
inner_loop = len(df)
separator = " "
#Current index in the news dataframe
current_step = 0
#Loop the price dataframe
for outer in range(outer_loop):
   current_date = sp500["Date"][outer]
   daily_titles = []
   no_news = sp500["No_of_News"][outer]
   if no_news > 0:
      while (df["trading_day"][current_step] == current_date):
           title = df["clean_title"][current_step]
           daily_titles.append(title)
           current_step += 1
           if current_step == inner_loop:
              break
      sp500["Titles"][outer] = separator.join(daily_titles)
   if outer % 100 == 0:
      print(outer)

# Add a column that checks total length of concatenated titles
sp500["Title_Check"] = sp500["Titles"].str.split().str.len()

# Export this dataframe as csv
sp500.to_csv(directory + folder + "sp5003.csv", index = False)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
sp500 = pd.read_csv(directory + folder + "sp5003.csv")

# Set a min threshold of news
min_thres = 8

# Remove the trading days with less than the min threshold of news
sp500 = sp500[sp500["No_of_News"] >= min_thres]
print(len(sp500))

# Export the price df as csv
sp500.to_csv(directory + folder + "sp5004.csv", index = False)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

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
sp500 = pd.read_csv("../input/thesissp500all-titles-v2/sp5004 v2.csv")

# Split the dataset into training and test sets
train_size = int(len(sp500) * 0.9)
sp500_train = sp500[:train_size]
sp500_train.reset_index(drop=True,inplace=True)
sp500_test = sp500[train_size:]
sp500_test.reset_index(drop=True,inplace=True)

# Define the data input for the training set and test set
X_train = sp500_train["Titles"]
X_test = sp500_test["Titles"]

# Featurise the data input with TF-IDF
tfidf = TfidfVectorizer(max_features = 10000, sublinear_tf=True)
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
pca = PCA(n_components=1000)
pca.fit(X_train_stand)
X_train_pca = pca.transform(X_train_stand)
X_test_pca = pca.transform(X_test_stand)
print(X_train_pca.shape)
print(X_test_pca.shape)

# Define the labels for both training and testing
y_train = sp500_train["Daily_Direction"]
y_test = sp500_test["Daily_Direction"]

# Calculate the total explained variance ratio
explained_variance = np.sum(pca.explained_variance_ratio_)
print(explained_variance)

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
train_size = X_train_pca.shape[0] - X_test_pca.shape[0]
X_train_pca_f = X_train_pca[:train_size]
X_val_pca = X_train_pca[train_size:]
y_train_fin = np.array(y_train[:train_size])
y_val = np.array(y_train[train_size:])
print(X_train_pca_f.shape)
print(X_val_pca.shape)
print(X_test_pca.shape)

# Define the number of runs for each scenario
R = 30
# Define the number of epoch for each run
E = 50

# Define history matrices to save down both training and validation results
history_train = np.zeros((E, R))
history_val = np.zeros((E, R))

# Define result matrices to hold the test results
MCC = np.zeros(R)
ACC = np.zeros(R)

# Kickstart the runs
for i in range(R):
    print("Start Run = {}".format(i))
    # Instantiate a model
    K.clear_session()
    model = build_model()
    
    # Define callback
    callbacks = [ModelCheckpoint("weights_dense_{}.h5".format(i), monitor='val_matthews_correlation', 
                                 save_best_only=True, save_weights_only=True, mode='max')]
    
    # Start training
    model.fit(X_train_pca_f, y_train_fin, batch_size=50, epochs=E, validation_data=[X_val_pca, y_val], 
              callbacks=callbacks, shuffle=False, verbose=True)
    
    # Save down the training history
    history_train[:,i] = model.history.history["matthews_correlation"]
    history_val[:,i] = model.history.history["val_matthews_correlation"]
    
    # Load the best weights saved by the checkpoint
    model.load_weights('weights_dense_{}.h5'.format(i))
    
    # Use the model for prediction on the test set
    predict_test = np.reshape((model.predict(X_test_pca) >= 0.5).astype(int), (X_test_pca.shape[0],))
    MCC[i] = matthews_correlation_self(y_test, predict_test)
    ACC[i] = np.sum(y_test == predict_test) / len(y_test)
    print(MCC[i])
    
# Export the results as csv
np.savetxt("history_train.csv", history_train, delimiter=",")
np.savetxt("history_val.csv", history_val, delimiter=",")
np.savetxt("MCC.csv", MCC, delimiter=",")
np.savetxt("ACC.csv", ACC, delimiter=",")

# Define its 1 SD move in the training set extracted locally
SD = 15.4761215678

# Define a function to conver the daily changes into the 3 classes
def three_class(daily, SD):
    if daily > SD:
        daily_class = 2
    elif daily < -1 * SD:
        daily_class = 0
    else:
        daily_class = 1
    return daily_class
    
# Allocate the 3 classes based on the 1 SD extracted
sp500["3_Class"] = sp500["Daily_Change"].apply(lambda x: three_class(x, SD))

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
    
# Kickstart the runs
for i in range(R):
    print("Start Run = {}".format(i))
    # Instantiate a model
    K.clear_session()
    model = build_model()
    
    # Define callback
    callbacks = [ModelCheckpoint("weights_dense_{}.h5".format(i), monitor='val_matthews_correlation', 
                                 save_best_only=True, save_weights_only=True, mode='max')]
    
    # Start training
    model.fit(X_train_pca_f, y_train_fin, batch_size=50, epochs=E, validation_data=[X_val_pca, y_val], 
              callbacks=callbacks, shuffle=False, verbose=True)
    
    # Save down the training history
    history_train[:,i] = model.history.history["matthews_correlation"]
    history_val[:,i] = model.history.history["val_matthews_correlation"]
    
    # Load the best weights saved by the checkpoint
    model.load_weights('weights_dense_{}.h5'.format(i))
    
    # Use the model for prediction on the test set
    predict_test = np.argmax(model.predict(X_test_pca), axis=1)
    actual_test = np.argmax(y_test, axis=1)
    MCC[i] = matthews_corrcoef(actual_test, predict_test)
    ACC[i] = np.sum(actual_test==predict_test) / len(y_test)
    predict_test_3 = np.eye(3)[predict_test]
    print(np.sum(predict_test_3, axis=0))
    print(MCC[i])
    
# Export the results as csv
np.savetxt("history_train_all_3.csv", history_train, delimiter=",")
np.savetxt("history_val_all_3.csv", history_val, delimiter=",")
np.savetxt("MCC_all_3.csv", MCC, delimiter=",")
np.savetxt("ACC_all_3.csv", ACC, delimiter=",")
