'''
This file creates the EDA plots
'''
# Imports 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 

root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # obtain path of root folder
data_path = os.path.join(root_folder, "data") # file path to data folder. Defined once, used multiple times

bankruptcy_df = pd.read_csv(os.path.join(data_path, "raw", "Taiwanese_Bankruptcy_Dataset.csv"))  # relative file paths used, aligning with data science principles
all_ys = bankruptcy_df["Bankrupt?"]

## Create pie chart of target variable
target_count = all_ys.value_counts()
target_labels = ["Not Bankrupt", "Bankrupt"] 
plt.pie(target_count, 
         explode=(0,0.1),
         labels=target_labels,
         autopct="%1.2f%%",
         shadow=True)

plt.axis("equal")
plt.title("Pie chart showing class imbalance")
plt.savefig(os.path.join(root_folder, "outputs", "exploratory_data_analysis", "class_distrib_pie_chart.pdf"))

## Create correlation matrix
plt.figure(figsize=(20,20))
sns.heatmap(bankruptcy_df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(root_folder, "outputs", "exploratory_data_analysis", "correlation_marix.pdf"))
