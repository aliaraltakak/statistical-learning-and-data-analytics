# Import the libraries that will be used. 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# Function to count zeros
def count_zeros(dataframe):
    print("Count of zeros per column:")
    for column in dataframe.columns:
        zero_count = (dataframe[column] == 0).sum()
        zero_percentage = (zero_count / len(dataframe)) * 100
        print(f"Column '{column}': {zero_count} occurence of zeros with a percentage of {zero_percentage:.2f}%.")

# Read the csv files into dataframes
diabetes = pd.read_csv('diabetes.csv')
meanDiabetes = pd.read_csv('mean_valued_diabetes.csv')
zeroDiabetes = pd.read_csv('zero_removed_diabetes.csv')

# Describe the data in the original dataset
describe = diabetes.describe()
print("The data of the original dataset can be observed below: \n")
print(describe.T, "\n")

# Count zeros in the original dataset
count_zeros(diabetes)
print("\n")

# Describe the data in the mean valued dataset
meanDescribe = meanDiabetes.describe()
print("The data of the mean value replaced dataset can be observed below: \n")
print(meanDescribe.T, "\n")

# Count zeros in the mean valued dataset
count_zeros(meanDiabetes)
print("\n")

# Describe the data in the zero removed dataset
zeroDescribe = zeroDiabetes.describe()
print("The data of the zero-rows removed dataset can be observed below: \n")
print(zeroDescribe.T, "\n")

# Count zeros in the zero removed dataset
count_zeros(zeroDiabetes)
print("\n")


#Plotting function of the datasets
def plot_distribution(data, title):
    columns = data.columns[:9]
    num_columns = 3
    num_rows = int(np.ceil(len(columns) / num_columns))  # Calculate the number of rows needed

    plt.subplots(num_rows, num_columns, figsize=(18, 18))
    plt.suptitle(title, fontsize=16)

    for i, j in itertools.zip_longest(columns, range(len(columns))):
        if i is not None:  # Check if the column name is not None
            ax = plt.subplot(num_rows, num_columns, j + 1)
            sns.histplot(data[i], kde=False, bins=20, edgecolor='black', ax=ax)
            plt.ylabel("Count")
            plt.title(i)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout
    plt.show()

# Plotting for each dataset
plot_distribution(diabetes, "Visualization of the Data in the Original Dataset")
plot_distribution(meanDiabetes, "Visualization of the Data in the Mean Value Replaced Dataset")
plot_distribution(zeroDiabetes, "Visualization of the Data in the Zero-Row Removed Dataset")
