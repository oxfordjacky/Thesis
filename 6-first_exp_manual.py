import pandas as pd
pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "df_title.csv"

# Import the csv with all the sections
df = pd.read_csv(directory + folder + file)

# Import the csv that contains the manual selected sections
section = pd.read_csv(directory + folder + "section.csv")

# Define a pandas series for the sections in the title df
df_section = df["section"]
df1 = df[df_section.isin(section.Section.unique())]

# Export the dataframe as csv
df1.to_csv(directory + folder + "df_manual.csv", index=False

import pandas as pd         
import string
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "df_manual.csv"

# Import the csv with the manually selected sections
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
stops.remove("no")

# Define a function to clean the text for sentiment analysis
def clean_text(my_string):
   # Expand some of the contraction words
   my_string = re.sub(r"No\.", "number", my_string)
   my_string = re.sub(r"M\&A", "merger acquisition", my_string)
    
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
   my_string = re.sub(r"\s{2,}", " ", my_string)
   #Replace k by 000
   my_string = re.sub(r"(\d+)(k)", r"\g<1> thousand", my_string)
   #Remove any numbers
   my_string = re.sub(r"[\d]+", " ", my_string)
   
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
df.to_csv(directory + folder + "df_manual1.csv", index=False)

import pandas as pd

pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "df_manual1.csv"

# Reimport the csv with all the titles only
df = pd.read_csv(directory + folder + file)

# Remove the duplicated rows of titles
title = df["clean_title"]
mask = title.duplicated().apply(lambda x: not x)
df_net = df[mask]

# Export the latest df as csv
df_net.to_csv(directory + folder + "df_manual2.csv", index=False)

import pandas as pd

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "df_manual2.csv"

# Import the csv with the manually selected sections
df = pd.read_csv(directory + folder + file)
# Set the index to its url
df.set_index("url", drop=True, inplace=True)

# Import the csv with all the sections
df_all = pd.read_csv(directory + folder + "df_title5.csv")
df_all.set_index("url", drop=True, inplace=True)

# Set the US Eastern time and trading day to datetime format
df_all["us_eastern_time"] = pd.to_datetime(df_all.us_eastern_time)
df_all["trading_day"] = pd.to_datetime(df_all.trading_day)

# Align the date and time between the two df
df["us_eastern_time"] = df_all["us_eastern_time"]
df["trading_day"] = df_all["trading_day"]

# Separate the df into null and non-null
df_null = df[df["trading_day"].isnull()]
df_notnull = df[df["trading_day"].isnull().apply(lambda x: not x)]

# Export both df as csv
df_null.to_csv(directory + folder + "df_manual2_null.csv", index=True)
df_notnull.to_csv(directory + folder + "df_manual2_notnull.csv", index=True)

import pandas as pd         
import string
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "df_title5.csv"

# Import the df with topic models
df = pd.read_csv(directory + folder + file)
print(df.head())

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
stops.remove("no")

# Define a function to clean the text for sentiment analysis
def clean_text(my_string):
   # Expand some of the contraction words
   my_string = re.sub(r"No\.", "number", my_string)
   my_string = re.sub(r"M\&A", "merger acquisition", my_string)
   my_string = re.sub(r"£", " pound ", my_string)
    
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
   my_string = re.sub(r"\s{2,}", " ", my_string)
   #Replace k by 000
   my_string = re.sub(r"(\d+)(k)", r"\g<1> thousand", my_string)
   #Remove any numbers
   my_string = re.sub(r"[\d]+", " ", my_string)
   
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
df.to_csv(directory + folder + "df_title6.csv", index=False)

pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "df_title6.csv"

# Reimport the csv with all the titles only
df = pd.read_csv(directory + folder + file)

# Remove the duplicated rows of titles
title = df["clean_title"]
mask = title.duplicated().apply(lambda x: not x)
df_net = df[mask]

# Export the latest df as csv
df_net.to_csv(directory + folder + "df_title7.csv", index=False)

from datetime import timedelta, datetime

pd.set_option('display.max_columns', 500)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
file = "df_manual2_null.csv"

# Reimport the csv with all the titles only
df = pd.read_csv(directory + folder + file)

# Make published the datetime formate
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

def time_convert(date_gmt, calendar, standard_diff):
    Year = date_gmt.year
    day_save = calendar[Year]
    if date_gmt >= day_save[0] and date_gmt <= day_save[1]:
        date_us = date_gmt - timedelta(hours=(standard_diff - 1))
    else:
        date_us = date_gmt - timedelta(hours=standard_diff)
    return date_us

check = df["published"].loc[0]
print(check)
print(time_convert(check, us_day_save, 5))

# Populate this column of US Eastern time (New York)
df["us_eastern_time"] = df["published"].apply(lambda x: time_convert(x, us_day_save, 5))    

# Export the updated df as csv
df.to_csv(directory + folder + "df_manual2_null1.csv", index=False)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"

# Import both the price csv and the title csv
df = pd.read_csv(directory + folder + "df_manual2_null1.csv")
sp500 = pd.read_csv(directory + folder + "sp5001.csv")

# Make date columns date_time format
df["us_eastern_time"] = pd.to_datetime(df.us_eastern_time)
sp500["Date"] = pd.to_datetime(sp500.Date)
sp500["Cutoff"] = pd.to_datetime(sp500.Cutoff)

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
    if current_step == inner_loop:
            break
    while (df["us_eastern_time"].loc[current_step] <= date_current):
        current_step += 1
        if current_step == inner_loop:
            break
    while (df["us_eastern_time"][current_step] > date_current and df["us_eastern_time"][current_step] <= date_next):
        df["trading_day"][current_step] = trading_day
        current_step += 1
        if current_step == inner_loop:
            break
    print(outer)

# Export the updated news df as csv
df.to_csv(directory + folder + "df_manual2_null2.csv", index=False)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"

# Import both manual df
df_null = pd.read_csv(directory + folder + "df_manual2_null2.csv")
df_notnull = pd.read_csv(directory + folder + "df_manual2_notnull.csv")

# Combine the two df together
df = pd.concat([df_null, df_notnull], axis=0)
df.sort_values(by ="us_eastern_time", inplace=True)

# Export the combined df as csv
df.to_csv(directory + folder + "df_manual3.csv", index=False)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"

# Import the latest manual df
df = pd.read_csv(directory + folder + "df_manual3.csv")

# Remove null values in cutoff_time
df = df[df["trading_day"].isnull().apply(lambda x: not x)]

# Export this updated df as csv
df.to_csv(directory + folder + "df_manual4.csv", index=False)

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"

# re-import both the latest price csv and the title csv
df = pd.read_csv(directory + folder + "df_manual4.csv")
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
sp500.to_csv(directory + folder + "sp500_manual.csv", index=False)

# Re-import the price csv and the news csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
df = pd.read_csv(directory + folder + "df_manual4.csv")
sp500 = pd.read_csv(directory + folder + "sp500_manual.csv")

# Set both dates columns to datetime format
df["trading_day"] = pd.to_datetime(df.trading_day)
sp500["Date"] = pd.to_datetime(sp500.Date)

# Add a column in the price df that will contain the concatenated titles
sp500["Titles"] = ""

# Populate this column
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
sp500.to_csv(directory + folder + "sp500_manual1.csv", index = False)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
sp500 = pd.read_csv(directory + folder + "sp500_manual1.csv")

# Set a min threshold of news
min_thres = 8

# Remove the trading days with less than the min threshold of news
sp500 = sp500[sp500["No_of_News"] >= min_thres]
print(len(sp500))

# Export the price df as csv
sp500.to_csv(directory + folder + "sp500_manual2.csv", index = False)

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
sp500 = pd.read_csv("/kaggle/input/sp500_manual2/sp500_manual2.csv")

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

# Calculate the total explained variance ratio
explained_variance = np.sum(pca.explained_variance_ratio_)
print(explained_variance)

# Define the labels for both training and testing
y_train = sp500_train["Daily_Direction"]
y_test = sp500_test["Daily_Direction"]

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
confusion = np.zeros((R, 2, 2))

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
    confusion[i,:,:] = confusion_matrix(y_test, predict_test)
    print(MCC[i])
    
# Export the results as csv
np.savetxt("history_train_manual.csv", history_train, delimiter=",")
np.savetxt("history_val_manual.csv", history_val, delimiter=",")
np.savetxt("MCC_manual.csv", MCC, delimiter=",")
np.savetxt("ACC_manual.csv", ACC, delimiter=",")
np.savetxt("confusion_mean.csv", confusion_mean, delimiter=",")
np.savetxt("confusion_std.csv", confusion_std, delimiter=",")

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
confusion = np.zeros((R, 3, 3))

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
    confusion[i,:,:] = confusion_matrix(actual_test, predict_test)
    print(np.sum(predict_test_3, axis=0))
    print(MCC[i])
    
# Compute the mean and std of the confusion matrix
confusion_mean = np.mean(confusion, axis=0)
confusion_std = np.std(confusion, axis=0)

# Export the results as csv
np.savetxt("history_train_manual_3.csv", history_train, delimiter=",")
np.savetxt("history_val_manual_3.csv", history_val, delimiter=",")
np.savetxt("MCC_manual_3.csv", MCC, delimiter=",")
np.savetxt("ACC_manual_3.csv", ACC, delimiter=",")
np.savetxt("confusion_mean_3.csv", confusion_mean, delimiter=",")
np.savetxt("confusion_std_3.csv", confusion_std, delimiter=",")



