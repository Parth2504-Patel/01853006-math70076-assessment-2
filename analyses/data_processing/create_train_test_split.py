'''
This file creates a train-test split using the original data. (seed set for reproducibility)
Random over and undersampling is done on the train dataset to mitigate class imbalance issue.
Scaling is fitted on the train dataset, and the same scaling transformation is applied to the test datsets to avoid data leakage. Note, binary variables are handles appropraitely.
'''

# Imports
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
import os

# Load in dataset and obtain inputs and target
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # obtain path of root folder
data_path = os.path.join(root_folder, "data") # file path to data folder. Defined once, used multiple times

bankruptcy_df = pd.read_csv(os.path.join(data_path, "raw", "Taiwanese_Bankruptcy_Dataset.csv"))  # relative file paths used, aligning with data science principles

bankruptcy_df.columns = bankruptcy_df.columns.str.strip() # As noticed in Initial Data Exploration notebook, column names have whitespace at the start
bankruptcy_df = bankruptcy_df.rename(columns={"Bankrupt?" : "bankrupt_status"}) # Change target variable to something more clear, Easier to handle and more interpretable
bankruptcy_df = bankruptcy_df.drop("Net Income Flag", axis=1) # Noted to drop in data exploration, since column only takes one value (no information provided from this)

# get input predictors and output variable
all_input_predictors = bankruptcy_df.drop("bankrupt_status", axis=1)
target_data = bankruptcy_df["bankrupt_status"]

#---------------------------------------------------------------------------------
# Create (straified) train-test split
#---------------------------------------------------------------------------------
# Perform stratified split (90-10 train test split), since its initial data exploration shows there is a class imbalance issue with small amount of data for minority class 
x_train, x_test, y_train, y_test = train_test_split(
    all_input_predictors, target_data,
    test_size = 0.1, random_state = 1853006, 
    shuffle=True, stratify=target_data
)

#=================================================================================
# Perform random sampling on train dataset and scaling for both train and test
#=================================================================================

# sampling strategy set to 0.15, achives 890 datapoints for minority class. 
over_sampler = RandomOverSampler(sampling_strategy=0.15, random_state=1853006) 
X_oversampled, Y_oversampled = over_sampler.fit_resample(x_train, y_train) # apply over sampling to the dataset

# sampling_strategy set to 1 so that even class balance (total dataset size is 890 * 2 = 1780)
under_sampler = RandomUnderSampler(sampling_strategy = 1, random_state = 1853006) 
train_X_balanced, train_Y_balanced = under_sampler.fit_resample(X_oversampled, Y_oversampled) # apply undersampling to dataset

# Inplace shuffling as undersampling returns ordered y values. 
train_X_balanced, train_Y_balanced = shuffle(train_X_balanced, train_Y_balanced, random_state=1853006)

#---------------------------------------------------------------------------------
# Perform scaling
#---------------------------------------------------------------------------------

# binary variable dropped, as only continous variable scaled
continious_train_x = train_X_balanced.drop(["Liability-Assets Flag"], axis=1)
continious_test_x = x_test.drop(["Liability-Assets Flag"], axis=1)

standard_scaler = StandardScaler()
X_train_scaled = standard_scaler.fit_transform(continious_train_x) # fitted using train dataset
X_test_scaled = standard_scaler.transform(continious_test_x) # same transformation as train set applied to avoid data leakage

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=continious_train_x.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=continious_test_x.columns)

# reset index to ensure that concatenation is done properly
train_Y_balanced.reset_index(drop=True, inplace=True)
train_X_balanced["Liability-Assets Flag"].reset_index(drop=True, inplace=True)
x_test["Liability-Assets Flag"].reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

## Form the train and test dataset
train_dataset = pd.concat([
    X_train_scaled_df, 
    train_X_balanced["Liability-Assets Flag"], 
    train_Y_balanced
], axis=1)

test_dataset = pd.concat([
    X_test_scaled_df,
    x_test["Liability-Assets Flag"],
    y_test
], axis=1)

# Write to CSV files

train_dataset.to_csv(os.path.join(data_path, "derived", "balanced_scaled_train_dataset.csv"), index=False)
test_dataset.to_csv(os.path.join(data_path, "derived", "scaled_test_dataset.csv"), index=False)