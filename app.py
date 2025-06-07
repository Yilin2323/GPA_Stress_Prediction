import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load models and encoder
gpa_model = joblib.load("gpa_model.pkl")
stress_model = joblib.load("stress_model.pkl")
stress_encoder = joblib.load("stress_encoder.pkl")

# Page config
st.set_page_config(page_title="GPA & Stress Predictor", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
        .big-font {
            font-size:22px !important;
        }
        .result-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Student Lifestyle Predictor")
st.subheader("📊 Predict your GPA & Stress Level from your daily habits")
st.write("Fill in your daily lifestyle habits and let the AI model analyze your academic and stress prediction!")

# --- Input Section ---
with st.form("prediction_form"):
    st.markdown("### ⏱️ Daily Habits")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("📘 Study Hours per Day", 0.0, 12.0, 3.0, 0.5)
        sleep_hours = st.slider("😴 Sleep Hours per Day", 0.0, 12.0, 7.0, 0.5)
        social_hours = st.slider("💬 Social Hours per Day", 0.0, 6.0, 1.5, 0.5)

    with col2:
        extracurricular_hours = st.slider("🎨 Extracurricular Hours per Day", 0.0, 5.0, 1.0, 0.5)
        physical_activity_hours = st.slider("🏃 Physical Activity Hours per Day", 0.0, 4.0, 1.0, 0.5)

    submitted = st.form_submit_button("🔍 Predict My Results")

# --- Prediction and Output ---
if submitted:
    input_data = pd.DataFrame([[
        study_hours,
        extracurricular_hours,
        sleep_hours,
        social_hours,
        physical_activity_hours
    ]], columns=[
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day"
    ])

    # Predict GPA
    predicted_gpa = gpa_model.predict(input_data)[0]

    # Predict Stress
    stress_encoded = stress_model.predict(input_data)[0]
    stress_rounded = round(stress_encoded)
    stress_label = stress_encoder.inverse_transform([stress_rounded])[0]

    # Recommendations
    def gpa_advice(gpa):
        if gpa >= 3.5:
            return "🌟 Great job! Maintain your study habits and balance."
        elif gpa >= 3.0:
            return "✅ You're doing well. Maybe increase focus time or reduce distractions."
        else:
            return "📚 Consider increasing your study hours and reducing stress to improve performance."

    def stress_advice(stress):
        if stress == "Low":
            return "😊 You're managing stress well. Keep a healthy routine!"
        elif stress == "Moderate":
            return "😐 Moderate stress detected. Consider more breaks and social time."
        else:
            return "⚠️ High stress detected! Prioritize sleep, exercise, and seek support if needed."

    # Display Results
    st.markdown(f"""
        <div class='result-box'>
            <h4>📈 Predicted GPA: <span class='big-font'>{predicted_gpa:.2f}</span></h4>
            <p>{gpa_advice(predicted_gpa)}</p>
            <h4>🧠 Predicted Stress Level: <span class='big-font'>{stress_label}</span></h4>
            <p>{stress_advice(stress_label)}</p>
        </div>
    """, unsafe_allow_html=True)
    