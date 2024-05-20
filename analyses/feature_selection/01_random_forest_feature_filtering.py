'''
This file is stage two of two for the feature selection process. Using the filtered variables from the VIF calculation, a further filtration is done using Random Forest.
A Random Forest classifier is first tuned using Cross-Validation, and then the tune hyperparameters are used to fit a random forest classifier.
The feature importance property of random forest classifier is used to then select the top 20 features, which forms the final subset of features considered.
'''

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle
import os

# Read in data and list of filtered features
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # obtain path of root folder
train_ds_path = os.path.join(root_folder, "data","derived", "balanced_scaled_train_dataset.csv") 
train_dataset = pd.read_csv(train_ds_path) # relative file pathing

feature_selection_output_path = os.path.join(root_folder, "outputs", "feature_selection")
vif_list_path = os.path.join(feature_selection_output_path, "feature_lists", "vif_filtered_features_list.txt") 

vif_filtered_feature_list = pd.read_csv(vif_list_path)["feature_name"] # list containing VIF filtered features

train_X = train_dataset[vif_filtered_feature_list] # get all X's using the filtered features
train_Y = train_dataset["bankrupt_status"] # get y data

# Define grid of hyperparameter to test
random_forest_hyperparams = ParameterGrid({
    "max_depth" : [5, 10, 15, 20, 23],
    "max_features" : np.arange(2,48,5)
})

# Store best model 
best_oob_score = -np.inf
best_random_forest_model = None

# try different combinations of hyperparameters possible
for hyperparam_config in random_forest_hyperparams:
    
    # Define random forest model with custom hyperparameter settings
    random_forest_model = RandomForestClassifier(
        oob_score=f1_score, # using the OOB samples and setting the OOB score metric to F1 Score
        n_estimators=200, # Fix number of trees to control computational load
        max_depth = hyperparam_config["max_depth"], 
        max_features = hyperparam_config["max_features"], 
        random_state=1853006, # seed set to CID for reproducibility
    )

    random_forest_model.fit(train_X, train_Y) # fit random forest classifier
    
    # if a better configuarion is found, update this
    if random_forest_model.oob_score_> best_oob_score:
        best_oob_score = random_forest_model.oob_score_
        best_random_forest_model = random_forest_model

## Select the top 20 features, which forms the final subset of features to use
descending_feature_importance_idx = np.argsort(best_random_forest_model.feature_importances_)[::-1] # get indices of feature importances in descending order
top_20_features_idx = descending_feature_importance_idx[:20] # get top 20 indices 
top_20_features_names = train_X.columns[top_20_features_idx] # get top 20 feature names

top_20_features_names_pd = pd.DataFrame(top_20_features_names, columns=["feature_name"]) 
top_20_features_names_pd.to_csv(os.path.join(feature_selection_output_path, "feature_lists", "random_forest_filtered_features_list.txt"), index=False) # write top feature names to txt file

# save the best model so training / tuning doesnt have to be run again for efficiency
with open(os.path.join(feature_selection_output_path, "feature_selector_random_forest_trained.pkl") , "wb") as model_file:
    pickle.dump(best_random_forest_model, model_file)