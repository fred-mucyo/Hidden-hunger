import pandas as pd

# Load the CSV
data = pd.read_csv("data/malnutrition_sample.csv")

# Show first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# Show info about dataset
print("\nDataset info:")
print(data.info())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())
