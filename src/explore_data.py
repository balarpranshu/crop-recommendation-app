import pandas as pd

data = pd.read_csv("data/Crop_recommendation.csv")

print("Dataset size:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns)

print("\nMissing values in each column:")
print(data.isnull().sum())

print("\nNumber of examples for each crop:")
print(data["label"].value_counts())

print("\nSummary Statistic :: ")
print(data.describe())