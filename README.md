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

- "analyses/" : This contains the python files that performs the majority of analysis. This includes the creating the train-test datasets, performing feature selection, training the machine learning models and conducting SHAP analysis.

- "data/" - This contains both the original and manipulated data that is used within the project. The original is in the "raw/" subfolder, and any manipulated are in "derived/" which includes the scaled train and test data, and the dataset with the 20 chosen features returned from feature selection. 

- "outputs/" - This folder contains the outputs from the scripts within this directory, which includes outputs such as plots, saved model files (.pkl) etc. This is subcateogrised in terms of the analysis conducted to group together related outputs together. 

- "reports/" - This folder contains the relevant files to produce the final report

- "src/"

## Setup
Setting up a new virtual environment for this project is recommened to ensure that the specified dependencies for this version match to ensure no clashes occur, however this is a choice that is left to the user. 

To install the dependencies required for this project, one can call the following command "pip install -r requirements.txt". This step assumes that you have pip installed for easy installation of packages. If not, many resources can be found online to guide you through the installation of pip. 

Then this repository can be cloned. There are many options available to do this, and thus I recommend opening the repository on GitHub's website, and use the green code button to guide you in which is the best way to get setup. 

## Usage
Once you are setup with this repository, you can use this repository in any manner that is suitable for you. You can reproduce the results by following the section in [Priject Structure](#project-structure), or you can extend on this project in whichever manner best suits you. 

## Acknowledgements
- The dataset is found from the UCI website, under the Creative Commons website, at https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
- 01853006 - Owner of this github Repo 

## License

This project uses a MIT License. Please visit the [LICENSE](LICENSE) file for more details.

## FAQs
- How can I contribute to this project?
GitHub users are invited to contribute to this repo. The steps to this would be as follows : 
1) Fork and clone the repository 
2) Create a new branch from the master branch for the contribution you would like to make.
3) Once you have finished with your contributions, push the branch back to your forked repository.
4) You can now open a pull request, and I will process as quick as I can. 

- I want to get in contact with you
Any questions are welcomed. To get in touch, the best method would be via email. The email address is 01853006@imperial.ac.uk