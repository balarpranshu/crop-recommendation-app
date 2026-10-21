import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("data/Crop_recommendation.csv")

feature_names = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
]

X = data[feature_names]
y = data["label"]

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

final_model.fit(X, y)

model_package = {
    "model": final_model,
    "feature_names": feature_names,
}

joblib.dump(model_package, "models/crop_recommender.joblib")

print("Model trained on all data and saved successfully.")