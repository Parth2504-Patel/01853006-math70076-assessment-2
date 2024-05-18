'''
This file performs a combination of under and over sampling to address the class imbalance issue
'''

# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

train_dataset = pd.read_csv("../../data/derived/train_dataset.csv") # Read in train data
test_dataset = pd.read_csv("../../data/derived/test_dataset.csv") # Read in test data

# Get the selected inputs and the target variable
train_X = train_dataset.drop("bankrupt_status", axis=1) # get the selected input predictors
train_Y = train_dataset["bankrupt_status"] # get the y

test_X = test_dataset.drop("bankrupt_status", axis=1) # get the selected input predictors
test_Y = test_dataset["bankrupt_status"] # get the y

# sampling strategy set to 0.15, achives 791 datapoints for minority class. 
over_sampler = RandomOverSampler(sampling_strategy=0.15, random_state=1853006) 
X_oversampled, Y_oversampled = over_sampler.fit_resample(train_X, train_Y) # apply over sampling to the dataset

under_sampler = RandomUnderSampler(sampling_strategy = 1, random_state = 1853006) # sampling_strategy set to 1 so that even class balance (total dataset size is 791 * 2 = 1582)
train_X_balanced, train_Y_balanced = under_sampler.fit_resample(X_oversampled, Y_oversampled) # apply undersampling to dataset

continious_train_x = train_X_balanced.drop(["Liability-Assets Flag"], axis=1)
continious_test_x = test_X.drop(["Liability-Assets Flag"], axis=1)

## Perform scaling
standard_scaler = StandardScaler()
X_train_scaled = standard_scaler.fit_transform(continious_train_x) # fitted using train dataset
X_test_scaled = standard_scaler.transform(continious_test_x) # same transformation as train set applied to avoid data leakage

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=continious_train_x.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=continious_test_x.columns)

train_Y_balanced.reset_index(drop=True, inplace=True)
train_X_balanced["Liability-Assets Flag"].reset_index(drop=True, inplace=True)
test_X["Liability-Assets Flag"].reset_index(drop=True, inplace=True)

balanced_scaled_train_df = pd.concat([
    X_train_scaled_df,
    train_X_balanced["Liability-Assets Flag"],
    train_Y_balanced
], axis = 1)

scaled_test_df = pd.concat([
    X_test_scaled_df,
    test_X["Liability-Assets Flag"],
    test_Y
], axis=1)

# Write to csvs
balanced_scaled_train_df.to_csv("../../data/derived/scaled_balanced_train_dataset.csv", index=False)
scaled_test_df.to_csv("../../data/derived/scaled_test_dataset.csv", index=False)