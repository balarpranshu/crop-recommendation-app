import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/Crop_recommendation.csv")

feature_names = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
]

X = data[feature_names]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training input shape:", X_train.shape)
print("Test input shape:", X_test.shape)

print("\nTraining labels:", y_train.shape)
print("Test labels:", y_test.shape)