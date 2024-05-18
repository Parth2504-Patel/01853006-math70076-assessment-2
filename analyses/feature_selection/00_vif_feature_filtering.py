'''
This file is stage one of two for the feature selection / filtering process to narrow down the total variables to a smaller subset of variables.
The first stage performs a Variance Inflation Factor(VIF) analysis, and filters out variables that have a VIF value greater than a threshold, set to 5.
'''

# Imports
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Read in train dataset created
train_dataset = pd.read_csv("../../data/derived/scaled_balanced_train_dataset.csv") # relative file pathing 
all_x = train_dataset.drop("bankrupt_status", axis=1) # drop target variable, not used in vif analysis

all_x_with_const = sm.add_constant(all_x, prepend=True) # add intercept column (as first column) to perform VIF calculations
## Calculate VIF scores
all_x_with_columns = all_x_with_const.columns
num_of_variables = len(all_x_with_columns)
vif_df = pd.DataFrame(all_x_with_columns.drop("const"), columns=["feature_name"])

vif_df["vif_score"] = [variance_inflation_factor(all_x_with_const.values, i) for i in range(num_of_variables)][1:] # list comprehension to efficiently calculate VIF score for each column

## Filter out features that have score higher than threshold (set to 5 as suggested by literature)
vif_threshold = 3
vif_filtered_features = vif_df[vif_df["vif_score"] < vif_threshold]["feature_name"]

vif_filtered_features.to_csv("../../outputs/feature_selection/feature_lists/vif_filtered_features_list.txt", index=False) # write filtered features to text file. Forms a simple documentation of changes also.
