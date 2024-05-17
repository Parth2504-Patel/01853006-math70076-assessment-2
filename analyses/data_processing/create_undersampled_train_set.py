'''
This file uses the final subset of features selected from feature selection and uses under and over sampling techinques to address the class imbalance issue to obtain a reasonably sized train dataset .
'''

# Imports
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

train_dataset = pd.read_csv("../../data/derived/scaled_train_dataset.csv") # Read in train data
final_selected_features = pd.read_csv("../../outputs/feature_selection/feature_lists/random_forest_filtered_features_list.txt")["feature_name"] # Read in the final selected features, and get corresponding columns from the datasets

# Get the selected inputs and the target variable
train_X = train_dataset[final_selected_features] # get the selected input predictors
train_Y = train_dataset["bankrupt_status"] # get the y

# sampling strategy set to 0.15, achives 791 datapoints for minority class. 
over_sampler = RandomOverSampler(sampling_strategy=0.15, random_state=1853006) 
X_oversamples, Y_oversamples = over_sampler.fit_resample(train_X, train_Y) # apply over sampling to the dataset

under_sampler = RandomUnderSampler(sampling_strategy = 1, random_state = 1853006) # sampling_strategy set to 1 so that even class balance (total dataset size is 791 * 2 = 1582)
train_X_resampled, train_Y_resampled = under_sampler.fit_resample(X_oversamples, Y_oversamples) # apply undersampling to dataset

# get the 20 selected features data using dataset obtained from under and oversampling
final_full_dataset = pd.concat([
    train_X_resampled, 
    train_Y_resampled
], axis =1)

final_full_dataset.to_csv("../../data/derived/class_balanced_train_dataset.csv", index=False) # write this edited dataset csv to use to pass to machine learning models