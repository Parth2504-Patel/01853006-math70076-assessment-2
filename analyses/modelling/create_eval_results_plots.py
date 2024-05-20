'''
This file performs SHAP analysis and creates its corresponding plot.
This file also creates the ROC curve to show the classifiers performance.
'''
# Imports
import pandas as pd
import pickle
import shap
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def perform_SHAP_and_create_plot(fitted_model, x_train_data, x_test_data, model_name):
    '''
    This function performs SHAP analysis and creates a barplot showing the top 10 mean absolute shap values and their corresponding features
    '''
    model_explainer = shap.Explainer(fitted_model, x_train_data) # creates explainer object using the model fitted and training data
    model_shap_vals = model_explainer(x_test_data) # calculate shap values for the test dataset

    # Visual plot settings
    plt.figure(figsize=(20,6))
    shap.plots.bar(model_shap_vals, max_display=11, show=False)
    plt.yticks(fontsize=6)
    plt.xlabel("Mean absolute SHAP value", fontsize=6)
    split_underscore = model_name.split("_")
    model_title = " ".join(split_underscore)
    plt.title(f"Bar plot of mean absolute SHAP value for {model_title}")
    plt.tight_layout()
    
    plt.savefig(f"../../outputs/modelling/{model_name}/{model_name}_SHAP_plot.pdf") # save plot to corresponding output directory with relevant path
 

def create_ROC_plot(fitted_model, x_test_data, y_test_data, model_name):
    '''
    This function creates the ROC curve plot for the model provided. It also calculates the area under the curve and reported in its label in the legend.
    '''
    y_predicted_probs = fitted_model.predict_proba(x_test_data)[:, 1] # get probability for one class
    fpr, tpr, _ = roc_curve(y_test_data, y_predicted_probs)
    area_under_curve_score = auc(fpr, tpr)
    
    # Visual plot settings
    model_name_clean = model_name.split("_")
    moel_name_spaced = " ".join(model_name_clean)
    plt.figure(figsize=(9,6))
    plt.title(f"ROC curve for {moel_name_spaced}")
    plt.plot(fpr, tpr, label=f"{moel_name_spaced} (AUC = {area_under_curve_score:.3f})")
    plt.plot([0,1], [0,1], linestyle="--", label="Random Performance")
    plt.legend(loc="lower right", fontsize="small")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    
    plt.savefig(f"../../outputs/modelling/{model_name}/{model_name}_ROC.pdf") # save plot to corresponding output directory with relevant path

# Read in train dataset
train_dataset = pd.read_csv("../../data/derived/selected_features_train_dataset.csv")
x_train = train_dataset.drop("bankrupt_status", axis=1)
y_train = train_dataset["bankrupt_status"]

# Read in test dataset
test_dataset = pd.read_csv("../../data/derived/selected_features_test_dataset.csv")
x_test = test_dataset.drop("bankrupt_status", axis=1)
y_test = test_dataset["bankrupt_status"]

#============================
# Logistic Regression
#============================
# Read in logistic regression saved model
logistic_reg_model_name = "logistic_regression"

with open(f"../../outputs/modelling/{logistic_reg_model_name}/logistic_reg_trained.pkl", "rb") as pkl_file:
    log_reg_model = pickle.load(pkl_file)
    

perform_SHAP_and_create_plot(
    fitted_model=log_reg_model,
    x_train_data=x_train,
    x_test_data=x_test,
    model_name=logistic_reg_model_name
)  

create_ROC_plot(
    fitted_model = log_reg_model,
    x_test_data=x_test,
    y_test_data=y_test,
    model_name= logistic_reg_model_name
)

#============================
# Gradient Boosting Classifier
#============================
gbc_model_name = "gradient_boosting_classifier"

with open(f"../../outputs/modelling/{gbc_model_name}/gbc_trained.pkl", "rb") as pkl_file:
    gbc_model = pickle.load(pkl_file)

perform_SHAP_and_create_plot(
    fitted_model=gbc_model,
    x_train_data=x_train,
    x_test_data=x_test,
    model_name=gbc_model_name
)  

create_ROC_plot(
    fitted_model=gbc_model,
    x_test_data=x_test,
    y_test_data=y_test,
    model_name=gbc_model_name
)