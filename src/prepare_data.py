import pandas as pd

data = pd.read_csv("data/Crop_recommendation.csv")

feature_names = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

X = data[feature_names]
y = data["label"]

print("Input data shape:", X.shape)
print("Output labels shape:", y.shape)

print("\nFirst 3 input rows:")
print(X.head(3))

print("\nFirst 3 correct crop labels:")
print(y.head(3))