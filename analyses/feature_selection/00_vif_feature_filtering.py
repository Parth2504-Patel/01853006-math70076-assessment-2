'''
This file is stage one of two for the feature selection / filtering process to narrow down the total variables to a smaller subset of variables.
The first stage performs a Variance Inflation Factor(VIF) analysis, and filters out variables that have a VIF value greater than a threshold, set to 5.
'''

# Imports
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# Read in train dataset created
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # obtain path of root folder
train_ds_path = os.path.join(root_folder, "data", "derived", "balanced_scaled_train_dataset.csv") 
train_dataset = pd.read_csv(train_ds_path) # relative file pathing used

train_xs = train_dataset.drop("bankrupt_status", axis=1) # drop target variable, not used in vif analysis
train_xs_with_const = sm.add_constant(train_xs, prepend=True) # add constant, needed for VIF calculations

#==================
# VIF Analysis
#==================

num_of_variables = len(train_xs_with_const.columns)
vif_df = pd.DataFrame(train_xs_with_const.columns, columns=["feature_name"])
vif_df["vif_score"] = [variance_inflation_factor(train_xs_with_const.values, i) for i in range(num_of_variables)] # list comprehension to efficiently calculate VIF score for each column
vif_df = vif_df[vif_df["feature_name"] != "const"] # filter out constant as it is not a feature, only included in for complete calculations

# Filter out features that have score higher than threshold (set to 5 as suggested by literature)
vif_threshold = 5
vif_filtered_features = vif_df[vif_df["vif_score"] < vif_threshold]["feature_name"]

vif_filtered_features.to_csv(os.path.join(root_folder, "outputs", "feature_selection", "feature_lists", "vif_filtered_features_list.txt"), index=False) # write filtered features to text file. Forms a simple documentation of changes also.
