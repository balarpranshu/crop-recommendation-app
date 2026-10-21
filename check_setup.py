import pandas as pd
import joblib
import streamlit as st

def get_bot_response(message):
    question = message.lower()

    if any(word in question for word in ["hello", "hi", "hey"]):
        return "Hello! Ask me about N, P, K, pH, rainfall, humidity, or the crop recommendation."

    if "ph" in question or "acidic" in question or "alkaline" in question:
        return (
            "Soil pH measures acidity or alkalinity. A pH below 7 is acidic, "
            "7 is neutral, and above 7 is alkaline. Suitable pH depends on the crop."
        )

    if "nitrogen" in question or question.strip() == "n":
        return "Nitrogen (N) supports leafy growth. The model uses it with all other inputs, not by itself."

    if "phosphorus" in question or question.strip() == "p":
        return "Phosphorus (P) is important for root development and flowering. It is one of the model inputs."

    if "potassium" in question or question.strip() == "k":
        return "Potassium (K) supports overall plant health. The model uses its value along with the other measurements."

    if "rain" in question:
        return "Rainfall is one of the strongest features in this training dataset. Enter the rainfall value in millimetres."

    if "humidity" in question:
        return "Humidity is the amount of water vapour in air, expressed as a percentage. It is another important model input."

    if "recommend" in question or "model" in question or "prediction" in question:
        return (
            "Enter all seven measurements above and click 'Recommend Crop'. "
            "The app shows the top crop and two alternatives."
        )

    if "confidence" in question:
        return (
            "Model confidence reflects how strongly the Random Forest's decision trees agreed. "
            "It is useful for ranking predictions, but it does not guarantee farm success."
        )

    if "limit" in question or "warning" in question:
        return (
            "This model learned from a small curated dataset. It does not include local season, "
            "irrigation, pests, seed variety, or market conditions."
        )

    return (
        "I can help with pH, nutrients (N, P, K), rainfall, humidity, "
        "model confidence, limitations, and using the crop recommender."
    )

model_package = joblib.load("models/crop_recommender.joblib")

model = model_package["model"]
feature_names = model_package["feature_names"]

st.title("Crop Recommendation System")
st.write("Enter soil and environmental measurements to receive crop recommendations.")

nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=13.0)
phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=23.0)
potassium = st.number_input("Potassium (K)", min_value=0.0, value=6.0)
temperature = st.number_input("Temperature (°C)", value=23.96)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=90.26)
ph_value = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.37)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=102.70)

if st.button("Recommend Crop"):
    soil_report = {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph_value,
        "rainfall": rainfall,
    }

    input_data = pd.DataFrame([soil_report], columns=feature_names)

    predicted_crop = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]
    top_3_indices = probabilities.argsort()[-3:][::-1]

    recommendations = pd.DataFrame(
        {
            "Crop": model.classes_[top_3_indices],
            "Model confidence": probabilities[top_3_indices]*100,
        }
    )

    st.success(f"Best recommendation: {predicted_crop.title()}")
    st.subheader("Top 3 recommendations")
    st.dataframe(
        recommendations,
        hide_index=True,
        column_config={
            "Model confidence (%)": st.column_config.NumberColumn(format="%.2f%%")
        },
    )

    st.caption(
        "This is a dataset-based learning project, not a substitute for local agricultural advice."
    )

st.divider()
st.header("Crop Assistant")
st.write("Ask a question about the model or its inputs.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you understand the crop recommendation system?",
        }
    ]

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Ask about pH, nutrients, rainfall, or the model")

if user_question:
    st.session_state.chat_messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    bot_answer = get_bot_response(user_question)

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": bot_answer}
    )

    with st.chat_message("assistant"):
        st.write(bot_answer)