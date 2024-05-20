'''
This file creates the train and train set with the selected features obtained from the feature selection stage.
'''

# Imports
import pandas as pd
import os

def read_select_write_dataset(inital_path, output_path):
    dataset = pd.read_csv(inital_path)
    final_selected_features = pd.read_csv("../../outputs/feature_selection/feature_lists/random_forest_filtered_features_list.txt")["feature_name"].tolist() # Read in the final selected features, and get corresponding columns from the datasets
    final_selected_features_and_target = final_selected_features +  ["bankrupt_status"] # get all input and target variables
    selected_dataset = dataset[final_selected_features_and_target]
    selected_dataset.to_csv(output_path, index=False)

root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # obtain path of root folder
dervied_data_path = os.path.join(root_folder, "data", "derived") 

# Train dataset
read_select_write_dataset(
    inital_path= os.path.join(dervied_data_path, "balanced_scaled_train_dataset.csv"),
    output_path= os.path.join(dervied_data_path, "selected_features_train_dataset.csv")
)

# Train dataset
read_select_write_dataset(
    inital_path= os.path.join(dervied_data_path, "scaled_test_dataset.csv"),
    output_path= os.path.join(dervied_data_path, "selected_features_test_dataset.csv")
)