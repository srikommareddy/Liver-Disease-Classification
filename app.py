#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')


# Define input features - update this list to match your model
feature_names = ['Age_of_the_patient',
  'Gender_of_the_patient',
  'Total_Bilirubin',
  'Direct_Bilirubin',
  'Alkphos_Alkaline_Phosphotase',
  'Sgpt_Alamine_Aminotransferase',
  'Sgot_Aspartate_Aminotransferase',
  'Total_Protiens',
  'ALB_Albumin',
  'A/G_Ratio_Albumin_and_Globulin_Ratio',
  'Result']

# App title
st.title("Liver Disease Classification - Random Forest Classifier")

st.sidebar.header("Enter Patient's Data")

# Input form
user_input = []
for feature in feature_names:
    val = st.sidebar.number_input(f"{feature}", value=0.0, format="%.2f")
    user_input.append(val)

if st.sidebar.button("Predict Liver Health"):
    # Convert to dataframe and scale
    input_df = pd.DataFrame([user_input], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    # Predict
    predicted_cluster = model.predict(input_scaled)[0]

    st.subheader("Liver Health Status")
    st.success(f"Patent has {predicted_cluster}")

    

