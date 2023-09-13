import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('./model/diabetes_model.pkl')

# Streamlit app title
st.title('Diabetes Prediction App')

# Input features
st.sidebar.header('Input Features')

# Define input fields for user
pregnancies = st.sidebar.text_input('Pregnancies', '0')
glucose = st.sidebar.text_input('Glucose (mg/dL)', '100')
blood_pressure = st.sidebar.text_input('Blood Pressure (mm Hg)', '72')
skin_thickness = st.sidebar.text_input('Skin Thickness (mm)', '23')
insulin = st.sidebar.text_input('Insulin (μU/ml)', '30')
bmi = st.sidebar.text_input('BMI', '32.0')
diabetes_pedigree_function = st.sidebar.text_input('Diabetes Pedigree Function', '0.3725')

# Load the data from diabetes.csv
df = pd.read_csv('diabetes.csv')

# Select the "Age" column
age = st.sidebar.selectbox('Select Age', df['Age'])

# Create a DataFrame with the user's input
user_input = pd.DataFrame({
    'Pregnancies': [float(pregnancies)],
    'Glucose': [float(glucose)],
    'BloodPressure': [float(blood_pressure)],
    'SkinThickness': [float(skin_thickness)],
    'Insulin': [float(insulin)],
    'BMI': [float(bmi)],
    'DiabetesPedigreeFunction': [float(diabetes_pedigree_function)],
    'Age': [float(age)]
})

# Make a prediction
if st.button('Predict'):
    prediction = model.predict(user_input)
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.write('The patient may have diabetes.')
    else:
        st.write('The patient is likely diabetes-free.')

