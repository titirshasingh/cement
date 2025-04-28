
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load the saved scaler and model
scaler = joblib.load('scaler.save')
model = tf.keras.models.load_model('final_lstm_model.keras')

# Streamlit App
st.set_page_config(page_title="Cement Strength Predictor", layout="centered")

st.title("🧱 Cement Strength Prediction App")
st.write("Enter the following features to predict the Concrete Strength:")

# Define the input fields
cement = st.number_input("Cement (kg in m³)", min_value=0.0, step=0.1)
slag = st.number_input("Blast Furnace Slag (kg in m³)", min_value=0.0, step=0.1)
fly_ash = st.number_input("Fly Ash (kg in m³)", min_value=0.0, step=0.1)
water = st.number_input("Water (kg in m³)", min_value=0.0, step=0.1)
superplasticizer = st.number_input("Superplasticizer (kg in m³)", min_value=0.0, step=0.1)
coarse_aggregate = st.number_input("Coarse Aggregate (kg in m³)", min_value=0.0, step=0.1)
fine_aggregate = st.number_input("Fine Aggregate (kg in m³)", min_value=0.0, step=0.1)
age = st.number_input("Age (days)", min_value=0, step=1)

# Predict Button
if st.button("Predict Concrete Strength"):
    # Collect input data
    input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Reshape for LSTM (samples, timesteps, features)
    input_scaled = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    st.success(f"🏗️ Predicted Concrete Strength: {prediction[0][0]:.2f} MPa")
