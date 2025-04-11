
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Macroeconomic Forecast App")

feature_list = ['lag_1_United States dollar', 'lag_1_Sterling pound',
                'lag_1_Euro', 'lag_1_South Africa Rand', 'lag_1_AverageInterbankRate',
                'lag_1_Deposit', 'lag_1_Savings', 'lag_1_Lending', 'lag_1_Overdraft',
                'lag_1_CBR Rate', 'month', 'quarter', 'year']

user_input = []
for feature in feature_list:
    value = st.number_input(f"{feature}", step=0.01)
    user_input.append(value)

input_df = pd.DataFrame([user_input], columns=feature_list)

# Scale and predict
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)

st.subheader("Predicted Average Interbank Rate:")
st.write(f"{prediction[0]:.4f}")
