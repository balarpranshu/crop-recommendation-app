import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

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

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Test accuracy: {accuracy:.2%}")

results = X_test.copy()
results["actual_crop"] = y_test
results["predicted_crop"] = predictions

mistakes = results[results["actual_crop"] != results["predicted_crop"]]

print(f"\nNumber of mistakes: {len(mistakes)}")
print("\nMistaken predictions:")
print(mistakes[["actual_crop", "predicted_crop"]])

importance_table = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": model.feature_importances_,
    }
).sort_values("importance", ascending=False)

print("\nFeature importance:")
print(importance_table)

print("\nDetailed classification report:")
print(classification_report(y_test, predictions))

fig, ax = plt.subplots(figsize=(12, 12))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    xticks_rotation=90,
    ax=ax,
)

plt.title("Crop Recommendation Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)

print("\nSaved chart to outputs/confusion_matrix.png")

sample_input = X_test.iloc[[0]]

probabilities = model.predict_proba(sample_input)[0]

top_3_indices = probabilities.argsort()[-3:][::-1]

print("\nExample soil report:")
print(sample_input)

print("\nTop 3 crop recommendations:")
for index in top_3_indices:
    crop = model.classes_[index]
    confidence = probabilities[index]
    print(f"{crop}: {confidence:.2%}")

single_tree_model = DecisionTreeClassifier(random_state=42)

single_tree_model.fit(X_train, y_train)

single_tree_predictions = single_tree_model.predict(X_test)

single_tree_accuracy = accuracy_score(y_test, single_tree_predictions)

print(f"\nSingle Decision Tree accuracy: {single_tree_accuracy:.2%}")
print(f"Random Forest accuracy: {accuracy:.2%}")

cv_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

cv_scores = cross_val_score(
    cv_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
)

print("\n5-fold cross-validation scores:")
print(cv_scores)

print(f"Average cross-validation accuracy: {cv_scores.mean():.2%}")
print(f"Score variation: {cv_scores.std():.2%}")