import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

'''
This python file imports the dataset using ucimlrepo library and converts it a .csv file, to be stored in the raw data section
'''

# The following three lines of code is from the ucimlrepo library guide to import data
# Can be found in dataset documentation (Refer to README for more details)
taiwanese_bankruptcy_prediction = fetch_ucirepo(id=572) 
X = taiwanese_bankruptcy_prediction.data.features 
y = taiwanese_bankruptcy_prediction.data.targets 

raw_data_df = pd.concat([y, X], axis=1) # store dataset as its own .csv file as its imported from ucimlrepo library

# Save dataframe to the csv file
csv_file_path = "../data/raw/Taiwanese_Bankruptcy_Dataset.csv" # specifiy where to save file using relative file paths

raw_data_df.to_csv(csv_file_path, index=False) # save dataset to file path specified