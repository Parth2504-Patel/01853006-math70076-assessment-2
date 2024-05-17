'''
This file creates a train-test split on the original data, and scales the data appropriately
Note : This is done prior to any feature selection. Feature selection (or any other further steps) are done using only on the train datset to avoid data leakage.
The test set is left unseen throughout until the very end to perform fair evaluation.
'''

# Imports
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load in dataset and obtain inputs and target
bankruptcy_df = pd.read_csv("../../data/raw/Taiwanese_Bankruptcy_Dataset.csv") # relative file paths used, aligning with data science principles

bankruptcy_df.columns = bankruptcy_df.columns.str.strip() # As noticed in Initial_Data_Exploration notebook, column names have whitespace at the start
bankruptcy_df = bankruptcy_df.rename(columns={"Bankrupt?" : "bankrupt_status"}) # Change target variable to something more clear, Easier to handle and more interpretable
bankruptcy_df = bankruptcy_df.drop("Net Income Flag", axis=1) # Noted to drop in data exploration, since column only takes one value (no information provided from this)

# get input predictors and output variable
all_input_predictors = bankruptcy_df.drop("bankrupt_status", axis=1)
target_data = bankruptcy_df["bankrupt_status"]

# Perform stratified split (80-20 train test split), since its initial data exploration shows there is a class imbalance issue with small amount of data for minority class 
x_train, x_test, y_train, y_test = train_test_split(
    all_input_predictors, target_data,
    test_size = 0.2, random_state = 1853006, 
    shuffle=True, stratify=target_data
)

## Performing data scaling
# Get continuous features to scale, Liability-Assests Flag is only input binary variable in dataset
continuous_X_train = x_train.drop(["Liability-Assets Flag"], axis=1)
continous_X_test = x_test.drop(["Liability-Assets Flag"], axis=1)

# Performing scaling on (continous) features
standard_scalar = StandardScaler()
continuous_X_train_scaled = standard_scalar.fit_transform(continuous_X_train) # Fit using train data only. 
continuous_X_test_scaled = standard_scalar.transform(continous_X_test) # Use same transformation on the test

## Combine final datasets and write to csv files
# Combine the relevant parts of train/test data together 
train_dataset = pd.DataFrame(continuous_X_train_scaled, columns=continuous_X_train.columns)
test_dataset = pd.DataFrame(continuous_X_test_scaled, columns=continous_X_test.columns)

# Add in the continous features, binary variable and the target variable to form a full dataset
full_train_dataset = pd.concat([
    y_train.reset_index(drop = True), # add target v,
    train_dataset, # add the continous features,
    x_train["Liability-Assets Flag"].reset_index(drop=True), # add back the binary variable
] , axis=1)

full_test_dataset = pd.concat([
    y_test.reset_index(drop = True), # add target v,
    test_dataset, # add the continous features,
    x_test["Liability-Assets Flag"].reset_index(drop=True), # add back the binary variable
] , axis=1)

# Write to CSV files
full_train_dataset.to_csv("../../data/derived/scaled_train_dataset.csv", index=False)
full_test_dataset.to_csv("../../data/derived/scaled_test_dataset.csv", index=False)