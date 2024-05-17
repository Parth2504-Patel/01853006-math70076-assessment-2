'''
This file creates the train set with the selected features obtained from the feature selection stage.
'''

# Imports
import pandas as pd

test_dataset = pd.read_csv("../../data/derived/scaled_test_dataset.csv") # Read in test data
final_selected_features = pd.read_csv("../../outputs/feature_selection/feature_lists/random_forest_filtered_features_list.txt")["feature_name"].tolist() # Read in the final selected features, and get corresponding columns from the datasets
final_selected_features_and_target = final_selected_features +  ["bankrupt_status"] # get all input and target variables

selected_features_test_set = test_dataset[final_selected_features_and_target] # get corresponding variables from initial test split

selected_features_test_set.to_csv("../../data/derived/selected_features_test_dataset.csv", index=False) # write this subsetted dataset to csv, so models can read this in directly.
