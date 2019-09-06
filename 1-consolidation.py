import pandas as pd
import json
import os
import re

pd.set_option('display.max_columns', 1000)

#Sample json file  
file = "/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Sample Data/Reuter Sample/10000035592491968439.json"
directory = "/Users/oxfor/OneDrive/Documents/reuters/"
json_files = [pos_json for pos_json in os.listdir(directory) if pos_json.endswith('.json')]
print(len(json_files))

#Read a sample json file
with open(file, 'r') as f:
    datastore = json.load(f)

#Create a list of headers from the json file
header = list(datastore.keys())
print(header)

#Create a dataframe template 
jsons_data = pd.DataFrame(columns = header)

#Define a function to remove unnessary text
def clean_text(sample):
    sample = re.sub(r"[A-Z]+ \(Reuters\) - ","",sample)
    sample = re.sub(r" Reporting [\w\s,]+;( Writing [\w\s,]+;)* Editing [\w\s,]+","", sample)
    return sample

#Print the first json file
file = "/Users/oxfor/OneDrive/Documents/reuters/10000035592491968439.json"
with open(os.path.join(directory,json_files[0]), 'r') as f:
    single = json.load(f)


# Populate the dataframe
for index, js in enumerate(json_files):
    with open(os.path.join(directory,js),'r') as json_file:
        json_text = json.load(json_file)
        url = json_text['url']
        title = json_text['title']
        section = json_text['section']
        text = clean_text(json_text['text'])
        published = json_text['published']
        jsons_data.loc[index] = [url, title, section, text, published]
        if index % 1000 == 0:
            print(index)

#Export pandas dataframe to csv
jsons_data.to_csv("/Users/oxfor/OneDrive/Documents/UCL Modules/Thesis/Data/df.csv", index=False)
   
