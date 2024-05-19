'''
This file fits the Logistic Regression model and the Gradient Boosting Classifer model.
For both, a grid search cross validation is done to tune the hyperparameters, (aiming to maximise the F1 score)
The results of the tuning and the tuned models themselves are saved so the training doesnt have to be repeated as it is time-consuming
'''

# Imports
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold
import pickle

# Read in the train dataset, and obtain the relevant x and y data
train_dataset = pd.read_csv("../../data/derived/selected_features_train_dataset.csv")
all_xs = train_dataset.drop("bankrupt_status", axis=1)
all_ys = train_dataset["bankrupt_status"]
ten_folds = KFold(n_splits=10, shuffle=True, random_state=1853006) # create ten folds of the dataset, can provide same split to all models to fit to

#============================
# Logistic Regression
#============================

penalty_param_range = np.linspace(0.001, 100, 1000) # define range of values to try for the hyperparameter
# Perform cross-validation to tune hyperparameter
log_reg_CV = LogisticRegressionCV(
    penalty="l1", # l1 penalty used for its sparsity property,
    solver="liblinear", # default solver doesnt support l1 penalty
    Cs=penalty_param_range,
    scoring="f1", # regularisation parameter that maximises F1 score is chosen
    cv=ten_folds, # 10-Fold CV used
    random_state=1853006
)
log_reg_CV.fit(all_xs, all_ys) # perform gridsearchCV

# Get all coefficients 
all_logistic_reg_coeffs = log_reg_CV.coef_[0] # get coefficients 
# Store in dataframe for visual purposes, and easy to write to csv
coeffs_df = pd.DataFrame({
    "Feature" : all_xs.columns,
    "Coefficient" : all_logistic_reg_coeffs
})

sorted_logistic_coeffs_df = coeffs_df.sort_values(by="Coefficient", ascending=False, key=abs) # sort by the absolute value (descending order), larger absolute implies more importance
sorted_logistic_coeffs_df.to_csv("../../outputs/modelling/logistic_regression/logistic_reg_coeffs.csv", index=False) # Save sorted feautre name and corresponding coefficient to csv

# Note LogisticRegressionCV automatically at the end fits a model using the best hyperparameter found, thus can directly use it
with open("../../outputs/modelling/logistic_regression/logistic_reg_trained.pkl", "wb") as log_reg_file:
    pickle.dump(log_reg_CV, log_reg_file) # save trained logistic regression model as a .pkl file 

#============================
# Gradient Boosting Classifier
#============================
# Define a parameter grid of the hyperparameters to try 
gradient_boosting_param_grid = {
    "learning_rate" : [0.05, 0.1, 0.15, 0.2],
    "max_depth" : range(2,4),
    "subsample" : [0.9, 1],
    "min_samples_split" : range(2,4)
}

gradient_boosting_model = GradientBoostingClassifier(random_state=1853006) # create gradient boosting classifier base model
gradient_boosting_CV = GridSearchCV(
    estimator=gradient_boosting_model, # model to use is gradient boosting model
    param_grid=gradient_boosting_param_grid, # pass in custom hyperparameter grid created 
    scoring="f1", # best hyperparameter configuration is such that it maximises the F1 score
    cv=ten_folds # pass in specific fold split to maintain uniformity training the two models
)

gradient_boosting_CV.fit(all_xs, all_ys) # perform gridsearch cross-validation for gradient boosting classifier
best_gb_classifier = gradient_boosting_CV.best_estimator_ # get model with best hyperparameter configuration

# Save the best model for later reference
with open("../../outputs/modelling/gradient_boosting_classifier/gbc_trained.pkl", "wb") as gb_cls_file:
    pickle.dump(best_gb_classifier, gb_cls_file)

gbc_feature_imp = pd.DataFrame({
    "Feature"  : all_xs.columns,
    "Importance Score" : best_gb_classifier.feature_importances_
})
gbc_feature_imp_sorted = gbc_feature_imp.sort_values(by="Importance Score", ascending=False) # sort from most to least important using score
gbc_feature_imp_sorted.to_csv("../../outputs/modelling/gradient_boosting_classifier/gbc_feature_importance_scores.csv", index=False)