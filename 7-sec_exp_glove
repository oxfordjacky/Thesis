import pandas as pd
from nltk.stem import SnowballStemmer
import numpy as np
pd.set_option('display.max_columns', 500)

# Re-import the price csv (LDA Topic 1)
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
sp500 = pd.read_csv(directory + "sp500_lda2_1.csv")
print(sp500.head())

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}
for titles in sp500["Titles"]:
    for word in titles.split():
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1

print("Size of Vocabulary:", len(word_counts))

# Create a list of tuples of unique words and their corresponding frequencies
word_counts_list = [(key,value) for key, value in word_counts.items()]

# Sort the list of word_counts in descending order
word_counts_list = sorted(word_counts_list, key=lambda x: x[1], reverse=True)
# Print the top 100 words
print(word_counts_list[:100])

# Define a snowball stemmer object for English
stemmer = SnowballStemmer("english")

# Load GloVe's embeddings
embeddings_index = {}
with open(directory + folder + "glove.840B.300d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = stemmer.stem(values[0])
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))

import numpy as np
import json

# Set the base directory and folder
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"

# Find the number of words that are missing from GloVe
missing_words = 0
for word, count in word_counts.items():
    if word not in embeddings_index:
       missing_words += 1
missing_ratio = round(missing_words/len(word_counts),4)*100
print("Number of words missing from GloVe:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

# Limit the vocab that we will use to words that are in GloVe
#dictionary to convert words to integers
vocab_to_int = {}
value = 0
for word, count in word_counts.items():
    if word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word
    
# Measure the usage ratio in the vectorisation
usage_ratio = round(len(vocab_to_int) / len(word_counts),2)*100

print("Total Number of Unique Words:", len(word_counts))
print("Number of Words we will use:", len(vocab_to_int))
print("Percent of Words we will use: {}%".format(usage_ratio))

# Define the embedding dimension
embedding_dim = 300
# Define the number of unique words in the embedding layer
nb_words = len(vocab_to_int)
# Create a word embedding matrix for the embedding layer
word_embedding_matrix = np.zeros((nb_words, embedding_dim))
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
        
# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))

# Export the word_embedding matrix as csv
np.savetxt(directory + folder + "word_embedding_matrix.csv", 
           word_embedding_matrix, delimiter=",")

# Export the vocab dictionary as json file
with open(directory + folder + "vocab_to_int.json", 'w') as fp:
    json.dump(vocab_to_int, fp)
    
# Set the base directory and folder
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"

# Re-import the word embedding matrix
word_embedding_matrix = np.genfromtxt(directory + folder + "word_embedding_matrix.csv",
                                      delimiter=",")
print(np.shape(word_embedding_matrix))

# Re-import the vocab_to_int json file as dictionary
with open(directory + folder + "vocab_to_int.json") as f:
     vocab_to_int = json.loads(f.read())

print(len(vocab_to_int))

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
sp500 = pd.read_csv(directory + "sp500_lda2_1.csv")

# Split the dataset into training and test sets
train_size = int(len(sp500) * 0.9)
sp500_train = sp500[:train_size]
sp500_train.reset_index(drop=True, inplace=True)
sp500_test = sp500[train_size:]
sp500_test.reset_index(drop=True, inplace=True)

# Define the data input for the training set and test set
X_train = sp500_train["Titles"]
X_test = sp500_test["Titles"]

# Featurise the data input with TF-IDF
tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary=vocab_to_int)
tfidf.fit(X_train)
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)
print(X_train.shape)
print(X_test.shape)

# Try one trading day as an example of getting an embedding index
example = X_train.getrow(0)

# Index of the example where the coefficient is non-zero
index = example.indices

# Corresponding word embedding matrix
example_embedding = word_embedding_matrix[index]

# TF-IDF scores of the corresponding non-zero entries
tf_score = example.data 

# Transpose the word embedding matrix
embedding_t = example_embedding.transpose()

# Calculate the word embedding vector weighed by the tf-idf scores
embed_vector = embedding_t @ tf_score

print(tf_score)

import pandas as pd
import numpy as np

# Define the directories for importing the base csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"

# Re-import both the latest price csv and the title csv
df = pd.read_csv(directory + "df_topic_1.csv")
sp500 = pd.read_csv(directory + "sp5001.csv")

# Set the date columns to datetime format
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

# Print the statistics of total news lengths
print(sp500["Total_Len"].astype(float).describe())
print(sp500["No_of_News"].astype(float).describe())

# Export this partially populated df as csv
sp500.to_csv(directory + folder + "sp5001_t1.csv", index=False)

# Re-import the price csv and the news csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
df = pd.read_csv(directory + "df_topic_1.csv")
sp500 = pd.read_csv(directory + folder + "sp5001_t1.csv")

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
      while (df["trading_day"][current_step]==current_date):
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
sp500.to_csv(directory + folder + "sp5001_t1_v1.csv", index=False)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
sp500 = pd.read_csv(directory + folder + "sp5001_t1_v3.csv")

# Re-import the vocab_to_int json file as dictionary
with open(directory + folder + "vocab_to_int.json") as f:
     vocab_to_int = json.loads(f.read())
     
# Count the total number of words and unknown words in our corpus
word_count = 0
unk_count = 0
for i in range(len(sp500)):
    title = sp500["Titles"][i]
    for word in title.split():
        word_count += 1
        if word not in vocab_to_int:
            unk_count += 1
    if i % 100 == 0:
        print(i)

# Calculate the unknown percentage
unk_percent = round(unk_count/word_count,4)*100
print("Total number of words in the news titles:", word_count)
print("Total number of UNKs in the news titles:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))


