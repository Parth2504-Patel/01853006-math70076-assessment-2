# 01853006-math70076-assessment-2

## Status : First Tagged Release

This repository contains the directory for coursework 2 for math70076. 
This project models a bankruptcy dataset and aims to perform feature importance analysis by fitting various machine learing models and using them as the basis of the analysis. Details of the origin of the dataset can be found in the [Acknowledgements](#acknowledgements) section


## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [FAQs](#faqs)

## Project Structure
This repository contains various folders aiming to breakdown the workflow in easy and accessible places

- ["analyses/"](analyses/) : This contains the python files that performs the majority of analysis. This includes the creating the train-test datasets, performing feature selection, training the machine learning models and conducting SHAP analysis.

 - ["data_processing/"](analyses/data_processing/) - This subdirectory of analyses contains the scripts to create the train-test split (with sampling and scaling), and to create the selected features dataset (which is subsequently used in training the machine learning models)

 - ["exploratory_data_analysis/](analyses/exploratory_data_analysis/) - This contains the script to create the exploratory data analysis plots that are required. 

 - ["feature_selection/"](analyses/feature_selection/) - This folder contains the scripts to perform the feature selection process. The files are numbered by the order which they occur in the process. The [first file](analyses/feature_selection/00_vif_feature_filtering.py) involves a VIF analysis filtering, and the [second file](analyses/feature_selection/01_random_forest_feature_filtering.py) contains the random forest filtering step.

 - ["modelling"](analyses/modelling/) - This folder contains the scripts to tune/train the machine learning, to create the evaluation plot and perform SHAP analysis.

- [data/"](data/) - This contains both the original and manipulated data that is used within the project. The original is in the ["raw/"](data/raw/) subfolder, and any manipulated are in ["derived/"](data/derived/) which includes the scaled train and test data, and the dataset with the 20 chosen features returned from feature selection. 

Within the "data/derived/" there are 4 sets of datasets available. These represent the intermediate stages, and the final two that are used for training and evaluating the models are highlighted at the end of this section. 

"balanced_scaled_train_dataset.csv" is the under and oversampled full training datset split which is scaled
"scaled_test_dataset.csv" is the full test dataset split with scaling

"selected_features{train/test}_dataset.csv" use the corresponding train or train (replace {train/test} with the dataset desired) from the first set to narrow it down to the final 20 features selected.

- ["outputs/"](outputs/) - This folder contains the outputs from the scripts within this directory, which includes outputs such as plots, saved model files (.pkl) etc. This is subcateogrised in terms of the analysis conducted to group together related outputs together. The next level of directories to "outputs/" mirrors the "analyses/" directory, in which the corresponding outputs from the folder under "analyses/" is stored in the same folder name under "outputs/". 

- ["reports/"](reports/) - This folder contains the relevant files to produce the final report. This includes the bibliography for the report, and the report itself in .tex for. The .pdf form of this can be found in the /outputs directory. 

- ["src/"](src/) - This contains a script to create the raw dataset as the dataset is imported from the UCI package as advised on the dataset documnetation. (See [Acknowledgements](#acknowledgements) section for link to dataset)

## Setup
Setting up a new virtual environment for this project is recommened to ensure that the specified dependencies for this version match to ensure no clashes occur, however this is a choice that is left to the user. 

To install the dependencies required for this project, one can call the following command "pip install -r requirements.txt". This step assumes that you have pip installed for easy installation of packages. If not, many resources can be found online to guide you through the installation of pip. 

Then this repository can be cloned. There are many options available to do this, and thus I recommend opening the repository on GitHub's website, and use the green code button to guide you in which is the best way to get setup. A common choice is git clone, that is :
git clone https://github.com/Parth2504-Patel/01853006-math70076-assessment-2.git

## Usage
Once you are setup with this repository, you can use this repository in any manner that is suitable for you. You can reproduce the results by following the section in [Project Structure](#project-structure), or you can extend on this project in whichever manner best suits you. 

## Acknowledgements
- The dataset is found from the UCI website, under the Creative Commons website, at https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
- 01853006 - Owner of this github Repo 

## License

This project uses MIT License. Please visit the [LICENSE](LICENSE) file for more details.

## FAQs
- How can I contribute to this project?
GitHub users are invited to contribute to this repo. The steps to this would be as follows : 
1) Fork and clone the repository 
2) Create a new branch from the master branch for the contribution you would like to make.
3) Once you have finished with your contributions, push the branch back to your forked repository.
4) You can now open a pull request, and I will process this as quickly as I can. 

- I want to get in contact with you, how can I do this?
Any questions are welcomed. To get in touch, the best method would be via email. The email address is 01853006@imperial.ac.uk