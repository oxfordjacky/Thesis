import pandas as pd
import spacy

pd.set_option('display.max_columns', 500)

# Import the news csv with titles only
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
df = pd.read_csv(directory + folder + "df_title7.csv")

# Load a language model
nlp = spacy.load("en_core_web_lg")

# Define a function to extract an entity type based on the NLP model
def extract_entity(text, ent_type):
    # Tokenise the text using the language model
    doc = nlp(text)
    # Extract the name entities of the text
    doc_ents = doc.ents
    # Create a list of tuples for the word and label pairs
    labels = [(x.text, x.label_) for x in doc_ents]
    # Create a list of name entities for a specific name entity type
    label_type = [T for T, L in labels if L==ent_type]
    # Join the labels to form a string
    label_combine = ";".join(label_type)
    return label_combine

# Define the entity types required
entity_type = ["GPE", "NORP", "ORG"]

# Create additional columns to host the specific name entities extracted
for ent_type in entity_type:
    df[ent_type] = df["title"].apply(lambda x: extract_entity(x, ent_type))

# Export the updated df as csv
df.to_csv(directory + folder + "df_ner.csv", index=False)

# Import the news csv with titles only
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"
df = pd.read_csv(directory + folder + "df_ner.csv")

# Define a list of entity types
entity_type = ["GPE", "NORP", "ORG"]

# Define a dictionary of df
df_dict = {}
for entity in entity_type:
    df_dict[entity] = df[df[entity].isnull().apply(lambda x: not x)]
    
# Export all the df as csv
for entity in entity_type:
    df_dict[entity].to_csv(directory + folder + "df_ner_{}.csv".format(entity), 
                           index=False)

# Import the news csv with titles only and GPE is non-null
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"

# Define a list of entity types
entity_type = ["GPE", "NORP", "ORG"]

# Import the df with different entity types as a dictionary of df
df_dict = {}
for entity in entity_type:
    df_dict[entity] = pd.read_csv(directory + folder + "df_ner_{}.csv".format(entity))

# Define a dictionary to hold entity values under different entity types
entity_df = {}
for entity in entity_type:
    # Define the required df
    df = df_dict[entity]
    # Define a pandas series with entity type only
    df_entity = df[entity]
    # Define a list of unique GPE values
    df_entity_unique = df_entity.unique()
    # Define the full list of all unique entity values
    full_list = ";".join(list(df_dict[entity][entity])).split(";")
    
    # Define a list to hold individual values
    individual_list = []
    count = []
    for item in df_entity_unique:
        individual = item.split(";")
        for individual_item in individual:
            individual_list.append(individual_item)

    # Get the unique values of individual list
    ind_uniq = list(set(individual_list))
    
    # Calculate the term frequencies for each unique value
    for value in ind_uniq:
        count.append(full_list.count(value))

    # Convert the list to a pandas dataframe
    entity_df[entity] = pd.DataFrame({entity: ind_uniq, "term_freq": count})

# Export all the df as csv
for entity in entity_type:
    entity_df[entity].to_csv(directory + folder + "ner_{}.csv".format(entity), 
                             index=False)
                             
import pandas as pd
import re
import numpy as np

pd.set_option('display.max_columns', 500)

# Import the news csv with titles only and GPE is non-null
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"

# Define a list of entity types
entity_type = ["GPE", "NORP", "ORG"]

# Import the df with different entity types as a dictionary of df
entity_df = {}
for entity in entity_type:
    entity_df[entity] = pd.read_csv(directory + folder + "ner_{}.csv".format(entity))

# Define doc freq columns
for entity in entity_type:
    entity_df[entity]["doc_freq"] = entity_df[entity]["term_freq"]

# Split the entity_df into 1 and non-1
for entity in entity_type:
    entity_df[entity][entity_df[entity]["term_freq"]==1].to_csv(directory + folder + "ner_{}_1.csv".format(entity), 
                                                                index=False)
    entity_df[entity][entity_df[entity]["term_freq"]!=1].to_csv(directory + folder + "ner_{}_not_1.csv".format(entity),
                                                                index=False)
 
# Import the news csv with titles only and GPE is non-null
directory = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/"
folder = "SP500/"

# Define a list of entity types
entity_type = ["GPE", "NORP", "ORG"]

# Import the df with different entity types as a dictionary of df
df_dict = {}
for entity in entity_type:
    df_dict[entity] = pd.read_csv(directory + folder + "df_ner_{}.csv".format(entity))
entity_df = {}
for entity in entity_type:
    entity_df[entity] = pd.read_csv(directory + folder + "ner_{}_not_1.csv".format(entity))

# Define an empty column to hold the doc frequencies
for entity in entity_type:
    entity_df[entity]["doc_freq"] = ""

# Populate the empty columns
for entity in entity_type:
    print("Starting {}:".format(entity))
    df = df_dict[entity]
    entity_ = entity_df[entity]
    df[entity] = df[entity].apply(lambda x: x.split(";"))
    for i in range(len(entity_)):
        term = entity_[entity][i]
        df["check"] = df[entity].apply(lambda x: term in x) 
        entity_["doc_freq"][i] = np.sum(df["check"])
        print(i)
    entity_df[entity] = entity_
    entity_df[entity].to_csv(directory + folder + "ner_{}_not_1_2.csv".format(entity), 
                             index=False) 
                             
# Define a list of entity types
entity_type = ["GPE", "NORP", "ORG"]

# Import the df with different entity types as a dictionary of df
df_dict_1 = {}
df_dict_not_1 = {}
for entity in entity_type:
    df_dict_1[entity] = pd.read_csv(directory + folder + "ner_{}_1.csv".format(entity))
entity_df = {}
for entity in entity_type:
    df_dict_not_1[entity] = pd.read_csv(directory + folder + "ner_{}_not_1_2.csv".format(entity))
    
# Concatenate the 2 df for each of the entity type
combine_dict = {}
for entity in entity_type:
    df = pd.concat([df_dict_1[entity], df_dict_not_1[entity]], axis=0)
    combine_dict[entity] = df

# Export the combined df as csv
for entity in entity_type:
    combine_dict[entity].to_csv(directory + folder + "ner_{}_2.csv".format(entity), 
                                index=False) 
                                
# Import a dictionary of csv with non-null entity type columns
df_dict = {}
for entity in entity_type:
    df_dict[entity] = pd.read_csv(directory + folder + "df_ner_{}.csv".format(entity))
 
# Import a dictionary of csv with lists of unique entity type values
unique_dict = {}
for entity in entity_type:
    unique_dict[entity] = pd.read_csv(directory + folder + "ner_{}_manual.csv".format(entity))

# Split the strings of name entities to lists of values
for entity in entity_type:
    df_dict[entity]["intersect"] = df_dict[entity][entity].apply(lambda x: x.split(";"))

# Define a function to find the intersection of two lists
def intersection(list_1, list_2):
    return list(set(list_1) & set(list_2))

# Find the intersections between the US_related values 
for entity in entity_type:
    print("Starting Entity Type: {}".format(entity))
    ner = unique_dict[entity]
    us_related = list(ner[ner["US_related"]==1][entity])
    df_dict[entity]["intersect"] = df_dict[entity]["intersect"].apply(lambda x: intersection(x, us_related))
    df_dict[entity]["intersect"] = df_dict[entity]["intersect"].apply(lambda x: ";".join(x))
    df_dict[entity].to_csv(directory + folder + "df_ner_{}_1.csv".format(entity), 
                           index=False) 

# Import a dictionary of csv with non-null entity type columns
df_dict = {}
for entity in entity_type:
    df_dict[entity] = pd.read_csv(directory + folder + "df_ner_{}_1.csv".format(entity))

# Remove the null rows in intersect
for entity in entity_type:
    intersect = df_dict[entity]["intersect"]
    df_dict[entity] = df_dict[entity][intersect.isnull().apply(lambda x: not x)]

# Export the updated df as csv
for entity in entity_type:
    df_dict[entity].to_csv(directory + folder + "df_ner_{}_2.csv".format(entity), 
                           index=False) 
                           
# Import a dictionary of csv with non-null entity type columns
df_dict = {}
for entity in entity_type:
    df_dict[entity] = pd.read_csv(directory + folder + "df_ner_{}_2.csv".format(entity))
    
# Combine the three df together
df_combine = df_dict[entity_type[0]]
for i in range(1,3):
    df_combine = pd.concat([df_combine, df_dict[entity_type[i]]], axis=0)

# Reset the index of the combined df
df_combine.reset_index(inplace=True, drop=True)

# Remove duplicate rows
url = df_combine["url"]
mask = url.duplicated().apply(lambda x: not x)
df_net = df_combine[mask]

# Export the combined df as csv
df_net.to_csv(directory + folder + "df_ner1.csv", index=False)



