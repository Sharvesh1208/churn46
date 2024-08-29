import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model2 = load_model("m.h5")

# Streamlit app title
st.title("Telecom Churn Prediction")
age = st.number_input("Age", min_value=0, value=0)
subscription_length = st.number_input("Subscription Length (months)", min_value=1, max_value=100, value=1)
charge_amount = st.number_input("Charge Amount (0: lowest, 9: highest)", min_value=0, max_value=9, value=0)
seconds_of_use = st.number_input("Total Seconds of Use", min_value=0, value=0)
frequency_of_use = st.number_input("Total Number of Calls", min_value=0, value=0)
frequency_of_sms = st.number_input("Total Number of SMS", min_value=0, value=0)
distinct_called_numbers = st.number_input("Total Distinct Called Numbers", min_value=0, value=0)
age_group = st.number_input("Age Group (1: youngest, 5: oldest)", min_value=1, max_value=5, value=1)
tariff_plan = st.number_input("Tariff Plan (1: Pay as you go, 2: Contractual)", min_value=1, max_value=2, value=1)
status = st.number_input("Status (1: Active, 2: Non-active)", min_value=1, max_value=2, value=1)
call_failures = st.number_input("Number of Call Failures", min_value=0, value=0)
complains = st.number_input("Complaints (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
customer_value = st.number_input("Customer Value", value=0)

import tensorflow as tf
st.write(f"TensorFlow version: {tf.__version__}")
st.write(f"TensorFlow version: {st.__version__}")

if st.button("Predict"):
    # Ensure all 13 features, including 'age', are included
    input_data = np.array([[age, subscription_length, charge_amount, seconds_of_use, 
                            frequency_of_use, frequency_of_sms, distinct_called_numbers, 
                            age_group, tariff_plan, status, call_failures, complains, 
                            customer_value]])  # 13 features now
    
    # Make prediction
    prediction = model2.predict(input_data)

    # Display the prediction result
    if prediction >= 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")
