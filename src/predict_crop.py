import pandas as pd
import joblib

model_package = joblib.load("models/crop_recommender.joblib")

model = model_package["model"]
feature_names = model_package["feature_names"]

soil_report = {
    "N": 13,
    "P": 23,
    "K": 6,
    "temperature": 23.96,
    "humidity": 90.26,
    "ph": 7.37,
    "rainfall": 102.70,
}

input_data = pd.DataFrame([soil_report], columns=feature_names)

predicted_crop = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]
top_3_indices = probabilities.argsort()[-3:][::-1]

print("Crop recommendation:", predicted_crop)
print("\nTop 3 recommendations:")

for index in top_3_indices:
    crop = model.classes_[index]
    confidence = probabilities[index]
    print(f"{crop}: {confidence:.2%}")