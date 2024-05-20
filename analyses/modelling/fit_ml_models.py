'''
This file fits the Logistic Regression model and the Gradient Boosting Classifer model.
For both, a grid search cross validation is done to tune its corresponding hyperparameters, and the hyperparamter configuration that attains the best mean F1 score is chosen.
The results of the tuning and the tuned models themselves are saved so the training doesnt have to be repeated as it is time-consuming
'''

# Imports
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold
import pickle
import os

# Read in the train dataset, and obtain the relevant x and y data
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # obtain path of root folder
train_ds_path = os.path.join(root_folder, "data", "derived", "selected_features_train_dataset.csv") 
train_dataset = pd.read_csv(train_ds_path) # relative file pathing used

train_xs = train_dataset.drop("bankrupt_status", axis=1)
train_ys = train_dataset["bankrupt_status"]

ten_folds = KFold(n_splits=10, shuffle=True, random_state=1853006) # create ten fold split of the dataset, can provide same split to all models to fit to

def sort_and_write_scores(desired_df, sort_by_column, save_file_path):
    '''
    The function takes in a dataframe, and sorts it by absolute value using the column name specified, and writes this sorted dataframe to csv for future reference. 
    The function sorts by the absolute value to account for models such as (logistic) regression, where the magnitude of the coefficient is an indicator of its importance to the model, and not the sign.
    This function allows it to be used for easy scalability if more models are added in the future.
    '''
    sorted_desired_df = desired_df.sort_values(by=sort_by_column, ascending=False, key=abs) # sorts by absolute value in descending order using the column name specified
    
    try:
        sorted_desired_df.to_csv(save_file_path, index=False) # write the sorted dataframe to csv to the file path that is specified

    # try catch some common exceptions to provide more informative error to user
    except PermissionError as perm_err:
        # Catches cases where there is a permission error
        print(f"Permission is denied to write to file path {save_file_path}, Please check permissions. \n The exact error is {perm_err}")   
        
    except pd.errors.EmptyDataError as empty_df_err:
        # Error if the dataframe is empty
        print(f"The dataframe is empty, please check dataframe construction \n Exact error is {empty_df_err}")
    
    except Exception as unexpected_e:
        # Catch all other cases
        print(f"Unexpected error has occured \n Exact error is {unexpected_e}")   

def save_trained_model(trained_model, save_file_path):
    '''
    This function takes the trained model and saves it at the file path specified in .pkl format so that training doesnt have to be rerun.
    '''
    try:
        with open(save_file_path, "wb") as model_file:
            pickle.dump(trained_model, model_file) # save trained model as a .pkl file    

    except Exception as unexpected_e:
        print(f"Unexpected error, exact error is {unexpected_e}")
        
#========================================================
# Logistic Regression
#========================================================

penalty_param_range = np.linspace(0.001, 100, 1000) # define range of values to try for the hyperparameter

# Perform cross-validation to tune hyperparameter
log_reg_CV = LogisticRegressionCV(
    penalty="l1", # l1 penalty used for its sparsity property,
    solver="liblinear", # default solver doesnt support l1 penalty hence liblinear used
    Cs=penalty_param_range, # pass in the range of values to try for hyperparameter
    scoring="f1", # regularisation parameter that maximises F1 score is chosen
    cv=ten_folds, # 10-Fold CV used
    random_state=1853006 # seed set for reproducibility 
)

log_reg_CV.fit(train_xs, train_ys) # perform gridsearchCV
all_logistic_reg_coeffs = log_reg_CV.coef_[0] # get all coefficients 

# Store in dataframe for visual purposes, and easy to write to csv
coeffs_df = pd.DataFrame({
    "Feature" : train_xs.columns,
    "Coefficient" : all_logistic_reg_coeffs
})

modelling_path = os.path.join(root_folder, "outputs", "modelling")
log_reg_path = os.path.join(modelling_path , "logistic_regression")

# Save the sorted coefficient values to a csv for future reference
sort_and_write_scores(
    desired_df=coeffs_df,
    sort_by_column="Coefficient",
    save_file_path = os.path.join(log_reg_path, "logistic_reg_coeffs.csv")
)

save_trained_model(
    trained_model=log_reg_CV, # Note LogisticRegressionCV automatically at the end fits a model using the best hyperparameter found, thus can directly use it
    save_file_path = os.path.join(log_reg_path, "logistic_reg_trained.pkl")
)

#========================================================
# Gradient Boosting Classifier
#========================================================

gbc_path =  os.path.join(modelling_path , "gradient_boosting_classifier")

# Define a parameter grid of the hyperparameters to try 
gradient_boosting_param_grid = {
    "learning_rate" : [0.05, 0.1, 0.15, 0.2],
    "max_depth" : range(2,4),
    "subsample" : [0.9, 1],
    "min_samples_split" : range(2,4)
}

gradient_boosting_model = GradientBoostingClassifier(random_state=1853006) # create gradient boosting classifier base model

# Perform cross-validation grid search to find best hyperparameter configuration for gradient boosting classifier
gradient_boosting_CV = GridSearchCV(
    estimator=gradient_boosting_model,
    param_grid=gradient_boosting_param_grid, # pass in custom hyperparameter grid created 
    scoring="f1", # best hyperparameter configuration is such that it maximises the F1 score
    cv=ten_folds # pass in specific fold split to maintain uniformity training the two models
)

gradient_boosting_CV.fit(train_xs, train_ys) # perform gridsearch cross-validation for gradient boosting classifier
best_gb_classifier = gradient_boosting_CV.best_estimator_ # get model with best hyperparameter configuration

gbc_feature_imp = pd.DataFrame({
    "Feature"  : train_xs.columns,
    "Importance Score" : best_gb_classifier.feature_importances_
})

# Save the sorted coefficient values to a csv for future reference
sort_and_write_scores(
    desired_df=gbc_feature_imp,
    sort_by_column="Importance Score",
    save_file_path= os.path.join(gbc_path, "gbc_feature_importance_scores.csv")
)

save_trained_model(
    trained_model=best_gb_classifier,
    save_file_path= os.path.join(gbc_path, "gbc_trained.pkl")
)