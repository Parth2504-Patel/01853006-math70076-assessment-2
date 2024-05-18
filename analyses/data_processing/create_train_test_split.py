'''
This file creates a train-test split on the original data. Feature selection  and model training is done using train set only.
The test set is left unseen throughout until the very end to perform fair evaluation.
'''

# Imports
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Load in dataset and obtain inputs and target
bankruptcy_df = pd.read_csv("../../data/raw/Taiwanese_Bankruptcy_Dataset.csv") # relative file paths used, aligning with data science principles

bankruptcy_df.columns = bankruptcy_df.columns.str.strip() # As noticed in Initial_Data_Exploration notebook, column names have whitespace at the start
bankruptcy_df = bankruptcy_df.rename(columns={"Bankrupt?" : "bankrupt_status"}) # Change target variable to something more clear, Easier to handle and more interpretable
bankruptcy_df = bankruptcy_df.drop("Net Income Flag", axis=1) # Noted to drop in data exploration, since column only takes one value (no information provided from this)

# get input predictors and output variable
all_input_predictors = bankruptcy_df.drop("bankrupt_status", axis=1)
target_data = bankruptcy_df["bankrupt_status"]

# Perform stratified split (90-10 train test split), since its initial data exploration shows there is a class imbalance issue with small amount of data for minority class 
x_train, x_test, y_train, y_test = train_test_split(
    all_input_predictors, target_data,
    test_size = 0.1, random_state = 1853006, 
    shuffle=True, stratify=target_data
)

## Form the train and test dataset
train_dataset = pd.concat([
    x_train, 
    y_train
], axis=1)

test_dataset = pd.concat([
    x_test,
    y_test
], axis=1)

# Write to CSV files
train_dataset.to_csv("../../data/derived/train_dataset.csv", index=False)
test_dataset.to_csv("../../data/derived/test_dataset.csv", index=False)