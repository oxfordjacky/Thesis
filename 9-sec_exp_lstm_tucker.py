import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
pd.set_option('display.max_columns', 500)

# Re-import the price csv
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/SP500/"
folder = "Model/"
sp500 = pd.read_csv(directory + folder + "sp5001_sd2_t1_3.csv")

# Set a min threshold of news
min_thres = 8
# Remove the trading days with no of news less than the threshold
sp500 = sp500[sp500["No_of_News"] >= min_thres]
# Re-set the index of the df
sp500.reset_index(drop=True, inplace=True)

# Replace NaN with empty space
sp500.fillna(value="", inplace=True)

# Split the dataset into training and test sets
train_size = int(len(sp500) * 0.9)
sp500_train = sp500[:train_size]
sp500_train.reset_index(drop=True,inplace=True)
sp500_test = sp500[train_size:]
sp500_test.reset_index(drop=True,inplace=True)

# Define the data input for the training set and test set
X_train = sp500_train["NER_Titles"]
X_test = sp500_test["NER_Titles"]

# Import the vocab csv as df
vocab = pd.read_csv(directory + folder + "vocab.csv")

# Convert the vocab df to a dictionary
keys = list(vocab["Vocab"])
values = list(vocab.index)
dictionary = dict(zip(keys, values))

# Featurise the data input with TF-IDF
tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary=dictionary, ngram_range=(1,5))
tfidf.fit(X_train)

# Define two np arrays, one for hosting the training data and the other for test data
X_train_inter = np.zeros((len(X_train), 24, len(dictionary)))
X_test_inter = np.zeros((len(X_test), 24, len(dictionary)))

# Populate the data matrix for training and testing
for j in range(1, 25):
    X_train_inter[:,j-1,:] = tfidf.transform(sp500_train["NER_Titles_{}".format(j)]).toarray()
    X_test_inter[:,j-1,:] = tfidf.transform(sp500_test["NER_Titles_{}".format(j)]).toarray()
    
import tensorly as tl
from tensorly.decomposition._tucker import partial_tucker
import numpy as np

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
    
# Perform tucker decomposition on the feature axis
pc = 700
tpca = TensorPCA(ranks=[pc], modes=[2])
tpca.fit(X_train_inter)
X_train_tucker = tpca.transform(X_train_inter)
X_test_tucker = tpca.transform(X_test_inter)
print(X_train_tucker.shape)
print(X_test_tucker.shape)

# Export the results as csv
np.save(directory + folder + "X_train_tucker_{}".format(pc), X_train_tucker)
np.save(directory + folder + "X_test_tucker_{}".format(pc), X_test_tucker)

# Calculate the variance preserved by the core tensor
Variance = (np.linalg.norm(X_train_tucker) / np.linalg.norm(X_train_inter))**2
print(Variance)

