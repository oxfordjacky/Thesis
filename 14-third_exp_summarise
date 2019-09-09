import pandas as pd

pd.set_option('display.max_columns', 500)

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentner_t1.csv")

import string
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

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
   my_string = re.sub(r"No\.", " number ", my_string)
   my_string = re.sub(r"M\&A", " merger acquisition ", my_string)
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

# Define a function to clean the text for sentiment words and US entities
def ner_text(my_string):
    
   # Clean the text
   my_string = re.sub(r"Amazon\.com", " Amazon ", my_string)
   my_string = re.sub(r"AT&T-Time Warner", " AT&T Time Warner ", my_string)
   my_string = re.sub(r"Conoco", " ConocoPhillips ", my_string)
   my_string = re.sub(r"Exxon Mobil", " Exxon ", my_string)
   my_string = re.sub(r"ExxonMobil", " Exxon ", my_string)
   my_string = re.sub(r"Democrats", " Democrats ", my_string)
   my_string = re.sub(r"Federal Reserve", " Fed ", my_string)
   my_string = re.sub(r"Citi", " Citigroup ", my_string)
   my_string = re.sub(r"GE", " General Electric ", my_string)
   my_string = re.sub(r"Goldman Sachs", " Goldman ", my_string)
   my_string = re.sub(r"Alphabet", " Google ", my_string)
   my_string = re.sub(r"JP Morgan", " JPMorgan ", my_string)
   my_string = re.sub(r"McDonald's", " McDonald ", my_string)
   my_string = re.sub(r"McDonald’s", " McDonald ", my_string)
   my_string = re.sub(r"Coca-Cola", " CocaCola ", my_string)
   my_string = re.sub(r"Moody's", " Moody ", my_string)
   my_string = re.sub(r"Moody’s", " Moody ", my_string)
   my_string = re.sub(r"N\.Y\.", " New York ", my_string)
   my_string = re.sub(r"NY", " New York ", my_string)
   my_string = re.sub(r"New York Times", " NYT ", my_string)
   my_string = re.sub(r"Republicans", " Republican ", my_string)
   my_string = re.sub(r"Treasury's", " Treasury ", my_string)
   my_string = re.sub(r"Treasury’s", " Treasury ", my_string)
   my_string = re.sub(r"Treasuries", " Treasury ", my_string)
   my_string = re.sub(r"Wall St", " Wall Street ", my_string)
   my_string = re.sub(r"Wal-Mart", " Walmart ", my_string)
   my_string = re.sub(r"Wells Fargo's", " Wells Fargo ", my_string)
   my_string = re.sub(r"Wells Fargo’s", " Wells Fargo ", my_string)
   my_string = re.sub(r"WSJ", " Wall Street Journal ", my_string)
   
   # Convert all US related terms to "American"
   my_string = re.sub(r"U\.S\.", " American ", my_string)
   my_string = re.sub(r"America", " American ", my_string)
   my_string = re.sub(r"American n", " American ", my_string)
   my_string = re.sub(r"United States", " American ", my_string)
   my_string = re.sub(r"U\.S", " American ", my_string)
   my_string = re.sub(r"US", " American ", my_string)
   
   # Convert words to lower case
   my_string = my_string.lower()
   
   # Remove punctuations
   my_string = my_string.translate(translator)
   
   # Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}", " ", my_string)
   
   # Remove one "American" if it is duplicated
   my_string = re.sub(r"american american", " american ", my_string)
   
   # Remove any more than 1 consecutive white space
   my_string = re.sub(r"\s{2,}", " ", my_string)
   return my_string
   
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy

pd.set_option('display.max_columns', 500)

# Load a language model
nlp = spacy.load("en_core_web_lg")

# Import the df with only LDA topic 1 and titles
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sentner_t1.csv")

# Import the vocab 
vocab = pd.read_csv(directory + folder + "vocab.csv")

# Convert the vocab df to a dictionary
keys = list(vocab["Vocab"])
values = list(vocab.index)
dictionary = dict(zip(keys, values))

# Create 4 new columns for summarisation purposes
df["summary"] = ""
df["ner_summary"] = ""
df["ner_number"] = ""
df["comb_summary"] = ""

# Define the number of sentences to be extracted
N = 5

# Start the loop for summarisation
for i in range(len(df)):
    # Create an example
    ex = df["text"].loc[i]
    # Tokenise the article using the language model loaded
    doc = nlp(ex)
    # Break the article into individual sentences
    sentence = list(doc.sents)
    # Convert each of the entries back to string
    sent_str = [sent.text for sent in sentence]
    # Create a df based on the sentence list
    df_sent = pd.DataFrame(sent_str, columns=["sentence"], index=list(range(len(sent_str))))
    # Clean the text
    df_sent["cleaned_sent"] = df_sent["sentence"].apply(lambda x: clean_text(x))
    # Featurise the individual sentences with TF-IDF
    tfidf = TfidfVectorizer(sublinear_tf=True)
    tfidf.fit(df_sent["cleaned_sent"])
    X = tfidf.transform(df_sent["cleaned_sent"]).toarray()
    X_score = np.sum(X, axis=1)
    # Find the index with the top N TF-IDF scores
    N_min = min(N, len(X))
    ind = np.sort(np.argpartition(X_score, -N_min)[-N_min:])
    # Find the sentences with the top scores
    df_filter = df_sent["cleaned_sent"].loc[ind]
    # Join the sentences together to form a single paragraph
    df["summary"].loc[i] = " ".join(list(df_filter))
    
    # Clean the text (NER)
    df_sent["ner_sent"] = df_sent["sentence"].apply(lambda x: ner_text(x))
    # Featurise the individual sentences with TF-IDF (NER)
    tfidf_ner = TfidfVectorizer(sublinear_tf=True, vocabulary=dictionary, ngram_range=(1,5))
    tfidf_ner.fit(df_sent["ner_sent"])
    X_ner = tfidf_ner.transform(df_sent["ner_sent"]).toarray()
    X_score_ner = np.sum(X_ner, axis=1)
    # Find the index with the top N TF-IDF scores (include those > 0)
    Zero = np.sum((X_score_ner > 0))
    N_min_ner = min(N, Zero)
    ind_ner = np.sort(np.argpartition(X_score_ner, -N_min_ner)[-N_min_ner:])
    df["ner_number"].loc[i] = N_min_ner
    # Find the sentences with the top scores
    df_filter_ner = df_sent["cleaned_sent"].loc[ind_ner]
    df["ner_summary"].loc[i] = " ".join(list(df_filter_ner))
    
    # Find the combined summary
    ind_comb = np.sort(np.array(list(set(ind).union(set(ind_ner)))))
    df_filter_combined = df_sent["cleaned_sent"].loc[ind_comb]
    df["comb_summary"].loc[i] = " ".join(list(df_filter_combined))
    print(i)

# Export the updated df as csv
df.to_csv(directory + folder + "df_sum_t1.csv", index=False)

df = pd.read_csv(directory + folder + "df_sum_t1.csv")

# Calculate the length of the cleaned titles
df["comb_len"] = df["comb_summary"].str.split().str.len()

# Export the dataframe as csv
df.to_csv(directory + folder + "df_sum_t1_1.csv", index=False)

# Import the df with main text and topic 1 only
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sum_t1_1.csv")
print(df.head())

# Re-import both the latest price csv and the title csv
sp500 = pd.read_csv(directory + "sp5001_sd.csv")

# Set the date columns to date_time format
df["trading_day"] = pd.to_datetime(df.trading_day)
df["us_eastern_time"] = pd.to_datetime(df.us_eastern_time)
sp500["Date"] = pd.to_datetime(sp500.Date)

# Sort df by its US Eastern Time
df.sort_values(by ="us_eastern_time", inplace=True)

# Reset the index of the df
df.reset_index(inplace=True, drop=True)

# Add 2 columns which will be populated in S&P500 df
sp500["No_of_News"] = ""
sp500["Total_Len"] = ""
sp500["Title_Len"] = ""

# Count no. of articles within daily date buckets
N = len(sp500)
for i in range(N):
    current_date = sp500["Date"][i]
    dummy = df[df["trading_day"] == current_date] 
    sp500["No_of_News"][i] = len(dummy)
    sp500["Total_Len"][i] = np.sum(dummy["comb_len"].values)
    sp500["Title_Len"][i] = np.sum(dummy["title_length"].values)
    if i % 100 == 0:
        print(i)
print(sp500.head())

# Print the statistics of total news lengths
print(sp500["Total_Len"].astype(float).describe())
print(sp500["No_of_News"].astype(float).describe())

# Export this partially populated df as csv
sp500.to_csv(directory + folder + "sp500_sum.csv", index=False)

import pandas as pd

# Re-import the price csv and the news csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
df = pd.read_csv(directory + folder + "df_sum_t1_1.csv")
sp500 = pd.read_csv(directory + folder + "sp500_sum.csv")

# Set both dates columns to datetime format
df["trading_day"] = pd.to_datetime(df.trading_day)
sp500["Date"] = pd.to_datetime(sp500.Date)

# Add a column in the price df that will contain the concatenated titles
sp500["Title"] = ""

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
      sp500["Title"][outer] = separator.join(daily_titles)
   if outer % 100 == 0:
      print(outer)

# Add a column that checks total length of concatenated titles
sp500["Title_Check"] = sp500["Title"].str.split().str.len()

# Add a column in the price df that will contain the concatenated titles
sp500["Text"] = ""

#Current index in the news dataframe
current_step = 0
#Loop the price dataframe
for outer in range(outer_loop):
   current_date = sp500["Date"][outer]
   daily_titles = []
   no_news = sp500["No_of_News"][outer]
   if no_news > 0:
      while (df["trading_day"][current_step]==current_date):
           title = df["ner_summary"][current_step]
           daily_titles.append(title)
           current_step += 1
           if current_step == inner_loop:
              break
      sp500["Text"][outer] = separator.join(daily_titles)
   if outer % 100 == 0:
      print(outer)

# Add a column that checks total length of concatenated titles
sp500["Text_Check"] = sp500["Text"].str.split().str.len()

# Check there is no null value
print(sp500.isnull().sum())

# Export this dataframe as csv
sp500.to_csv(directory + folder + "sp500_sum1.csv", index=False)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
sp500 = pd.read_csv(directory + folder + "sp500_sum1.csv")

# Set a min threshold of news
min_thres = 8

# Remove the trading days with less than the min threshold of news
sp500 = sp500[sp500["No_of_News"] >= min_thres]
print(len(sp500))

# Export the price df as csv
sp500.to_csv(directory + folder + "sp500_sum2.csv", index = False)

import pandas as pd
pd.set_option('display.max_columns', 500)

# Re-import the price csv (LDA Topic 1)
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
sp500 = pd.read_csv(directory + folder + "sp500_sum2.csv")
print(sp500.head())

# Find the number of times each word was used and the size of the vocabulary (Titles)
word_counts_title = {}
for titles in sp500["Title"]:
    for word in titles.split():
        if word not in word_counts_title:
            word_counts_title[word] = 1
        else:
            word_counts_title[word] += 1

print("Size of Vocabulary:", len(word_counts_title))

# Create a list of tuples of unique words and their corresponding frequencies
word_counts_title_list = [(key,value) for key, value in word_counts_title.items()]

# Sort the list of word_counts in descending order
word_counts_title_list = sorted(word_counts_title_list, key=lambda x: x[1], reverse=True)
# Print the top 100 words
print(word_counts_title_list[:100])

# Find the number of times each word was used and the size of the vocabulary (Text)
word_counts_text = {}
for titles in sp500["Text"]:
    for word in titles.split():
        if word not in word_counts_text:
            word_counts_text[word] = 1
        else:
            word_counts_text[word] += 1

print("Size of Vocabulary:", len(word_counts_text))

# Create a list of tuples of unique words and their corresponding frequencies
word_counts_text_list = [(key,value) for key, value in word_counts_text.items()]

# Sort the list of word_counts in descending order
word_counts_text_list = sorted(word_counts_text_list, key=lambda x: x[1], reverse=True)
# Print the top 100 words
print(word_counts_text_list[:100])

import pandas as pd
from nltk.stem import SnowballStemmer
import numpy as np
pd.set_option('display.max_columns', 500)

# Set the base folder for the GloVe embedding

# Define a snowball stemmer object for English
stemmer = SnowballStemmer("english")

# Load GloVe's embeddings
embeddings_index = {}
with open(directory + folder + "glove.6B.50d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = stemmer.stem(values[0])
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))

# Set the base directory and folder
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"

# Find the number of words that are missing from GloVe
missing_words_title = 0
for word, count in word_counts_title.items():
    if word not in embeddings_index:
       missing_words_title += 1
missing_ratio_title = round(missing_words_title/len(word_counts_title),4)*100
print("Number of words missing from GloVe:", missing_words_title)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio_title))

# Limit the vocab that we will use to words that are in GloVe
#dictionary to convert words to integers
vocab_to_int_title = {}
value = 0
for word, count in word_counts_title.items():
    if word in embeddings_index:
        vocab_to_int_title[word] = value
        value += 1

# Dictionary to convert integers to words
int_to_vocab_title = {}
for word, value in vocab_to_int_title.items():
    int_to_vocab_title[value] = word
    
# Measure the usage ratio in the vectorisation
usage_ratio_title = round(len(vocab_to_int_title) / len(word_counts_title),2)*100

print("Total Number of Unique Words:", len(word_counts_title))
print("Number of Words we will use:", len(vocab_to_int_title))
print("Percent of Words we will use: {}%".format(usage_ratio_title))

# Define the embedding dimension
embedding_dim = 50
# Define the number of unique words in the embedding layer
nb_words = len(vocab_to_int_title)
# Create a word embedding matrix for the embedding layer
word_embedding_title = np.zeros((nb_words, embedding_dim))
for word, i in vocab_to_int_title.items():
    if word in embeddings_index:
        word_embedding_title[i] = embeddings_index[word]
        
# Check if value matches len(vocab_to_int)
print(len(word_embedding_title))

# Export the word_embedding matrix as csv
np.savetxt(directory + folder + "word_embedding_title.csv", 
           word_embedding_title, delimiter=",")

# Export the vocab dictionary as json file
with open(directory + folder + "vocab_to_int_title.json", 'w') as fp:
    json.dump(vocab_to_int_title, fp)
    
# Set the base directory and folder
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"

# Find the number of words that are missing from GloVe
missing_words_text = 0
for word, count in word_counts_text.items():
    if word not in embeddings_index:
       missing_words_text += 1
missing_ratio_text = round(missing_words_text/len(word_counts_text),4)*100
print("Number of words missing from GloVe:", missing_words_text)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio_text))

# Limit the vocab that we will use to words that are in GloVe
#dictionary to convert words to integers
vocab_to_int_text = {}
value = 0
for word, count in word_counts_text.items():
    if word in embeddings_index:
        vocab_to_int_text[word] = value
        value += 1

# Dictionary to convert integers to words
int_to_vocab_text = {}
for word, value in vocab_to_int_text.items():
    int_to_vocab_text[value] = word
    
# Measure the usage ratio in the vectorisation
usage_ratio_text = round(len(vocab_to_int_text) / len(word_counts_text),2)*100

print("Total Number of Unique Words:", len(word_counts_text))
print("Number of Words we will use:", len(vocab_to_int_text))
print("Percent of Words we will use: {}%".format(usage_ratio_text))

# Define the embedding dimension
embedding_dim = 50
# Define the number of unique words in the embedding layer
nb_words = len(vocab_to_int_text)
# Create a word embedding matrix for the embedding layer
word_embedding_text = np.zeros((nb_words, embedding_dim))
for word, i in vocab_to_int_text.items():
    if word in embeddings_index:
        word_embedding_text[i] = embeddings_index[word]
        
# Check if value matches len(vocab_to_int)
print(len(word_embedding_text))

# Export the word_embedding matrix as csv
np.savetxt(directory + folder + "word_embedding_text.csv", 
           word_embedding_text, delimiter=",")

# Export the vocab dictionary as json file
with open(directory + folder + "vocab_to_int_text.json", 'w') as fp:
    json.dump(vocab_to_int_text, fp)
    
# Set the max length of the title embedding matrix
max_len_title = np.max(sp500["Title_Check"])

# Create a zero word embedding matrix for title concatenation
N = len(sp500)
embed_title = np.zeros((N, max_len_title, 50))

# Populate this word embedding matrix
for i in range(N):
   #Define the example and its length
   example = sp500["Title"].loc[i]
   length = sp500["Title_Check"].loc[i]

   # Populate the embedding matrix
   for ind, word in enumerate(example.split()):
       if word in vocab_to_int_title:
           embed_index = vocab_to_int_title[word]
           embed_title[i,ind-length,:] = word_embedding_title[embed_index]
   if i % 100 == 0:
       print(i)
       
 import tensorly as tl
from tensorly.decomposition._tucker import partial_tucker
import numpy as np

# Split the embedding matrix into training and test
# Split the dataset into training and test sets
train_size = int(len(embed_title) * 0.9)
X_title_train = embed_title[:train_size]
X_title_test = embed_title[train_size:]

# Define the tensor PCA class
class TensorPCA:
    def __init__(self, ranks, modes):
        self.ranks = ranks
        self.modes = modes

    def fit(self, tensor):
        self.core, self.factors = partial_tucker(tensor, modes=self.modes, ranks=self.ranks)
        return self

    def transform(self, tensor):
        return tl.tenalg.multi_mode_dot(tensor, self.factors, modes=self.modes, transpose=True)
    
# Perform tucker decomposition on the word dimension
pc = 550
tpca = TensorPCA(ranks=[pc], modes=[1])
tpca.fit(X_title_train)
X_title_train_tuck = tpca.transform(X_title_train)
X_title_test_tuck = tpca.transform(X_title_test)
print(X_title_train_tuck.shape)
print(X_title_test_tuck.shape)

# Export the results as csv
np.save(directory + folder + "X_title_train_tuck_{}".format(pc), X_title_train_tuck)
np.save(directory + folder + "X_title_test_tuck_{}".format(pc), X_title_test_tuck)

# Calculate the variance preserved by the core tensor
Variance = (np.linalg.norm(X_title_train_tuck) / np.linalg.norm(X_title_train))**2
print(Variance)

# Set the max length of the title embedding matrix
max_len_text = np.max(sp500["Text_Check"])

# Create a zero word embedding matrix for text concatenation
N = int(len(sp500))
embed_text = np.zeros((N, max_len_text, 50), dtype=np.float32)

# Populate this word embedding matrix
for i in range(N):
   #Define the example and its length
   example = sp500["Text"].loc[i]
   length = sp500["Text_Check"].loc[i]

   # Populate the embedding matrix
   for ind, word in enumerate(example.split()):
       if word in vocab_to_int_text:
           embed_index = vocab_to_int_text[word]
           embed_text[i,ind-length,:] = word_embedding_text[embed_index]
   if i % 100 == 0:
       print(i)
       
import tensorly as tl
from tensorly.decomposition._tucker import partial_tucker
import numpy as np

# Split the embedding matrix into training and test
# Split the dataset into training and test sets
train_size = int(len(embed_text) * 0.9)
X_text_train = embed_text[:train_size]
X_text_test = embed_text[train_size:]

# Define the tensor PCA class
class TensorPCA:
    def __init__(self, ranks, modes):
        self.ranks = ranks
        self.modes = modes

    def fit(self, tensor):
        self.core, self.factors = partial_tucker(tensor, modes=self.modes, ranks=self.ranks)
        return self

    def transform(self, tensor):
        return tl.tenalg.multi_mode_dot(tensor, self.factors, modes=self.modes, transpose=True)
    
# Perform tucker decomposition on the word dimension
pc = 4500
tpca = TensorPCA(ranks=[pc], modes=[1])
tpca.fit(X_text_train)
X_text_train_tuck = tpca.transform(X_text_train)
X_text_test_tuck = tpca.transform(X_text_test)
print(X_text_train_tuck.shape)
print(X_text_test_tuck.shape)

# Export the results as csv
np.save(directory + folder + "X_text_train_tuck_{}".format(pc), X_text_train_tuck)
np.save(directory + folder + "X_text_test_tuck_{}".format(pc), X_text_test_tuck)

# Calculate the variance preserved by the core tensor
Variance = (np.linalg.norm(X_text_train_tuck) / np.linalg.norm(X_text_train))**2
print(Variance)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
sp500 = pd.read_csv(directory + folder + "sp500_sum2.csv")

# Re-import the vocab_to_int json file as dictionary
with open(directory + folder + "vocab_to_int_title.json") as f:
     vocab_to_int = json.loads(f.read())


# Count the total number of words and unknown words in our corpus
word_count = 0
unk_count = 0
for i in range(len(sp500)):
    title = sp500["Title"][i]
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

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Main Text/"
sp500 = pd.read_csv(directory + folder + "sp500_sum2.csv")

# Re-import the vocab_to_int json file as dictionary
with open(directory + folder + "vocab_to_int_text.json") as f:
     vocab_to_int = json.loads(f.read())

# Count the total number of words and unknown words in our corpus
word_count = 0
unk_count = 0
for i in range(len(sp500)):
    title = sp500["Text"][i]
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

print(np.max(sp500["Title_Check"]))
print(np.min(sp500["Title_Check"]))
print(np.max(sp500["Text_Check"]))
print(np.min(sp500["Text_Check"]))

# Split the dataset into training and test sets
train_size = int(len(embed_title) * 0.9)
X_title_train = embed_title[:train_size]
X_title_test = embed_title[train_size:]

# Load the title embedding
X_title_train_tuck = np.load(directory + folder + "X_title_train_tuck_550.npy")

# Calculate the variance preserved
Variance = (np.linalg.norm(X_title_train_tuck) / np.linalg.norm(X_title_train))**2
print(Variance)

# Split the dataset into training and test sets
train_size = int(len(embed_text) * 0.9)
X_text_train = embed_text[:train_size]
X_text_test = embed_text[train_size:]

# Load the title embedding
X_text_train_tuck = np.load(directory + folder + "X_text_train_tuck_4500.npy")

# Calculate the variance preserved
Variance = (np.linalg.norm(X_text_train_tuck) / np.linalg.norm(X_text_train))**2
print(Variance)
