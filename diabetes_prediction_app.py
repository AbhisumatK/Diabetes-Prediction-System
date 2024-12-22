import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load and preprocess the data
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Balance the dataset
df_no_diabetes = data[data['diabetes'] == 0]
df_diabetes = data[data['diabetes'] == 1]
df_no_diabetes_sampled = df_no_diabetes.sample(n=8500, random_state=42)
df_balanced = pd.concat([df_no_diabetes_sampled, df_diabetes])
data = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Map categorical data
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data['smoking_history'] = data['smoking_history'].map({
    'former': 1,
    'not current': 1,
    'current': 2,
    'never': 0,
    'ever': 0,
    'No Info': -1
})

data = data.dropna()

x = data.drop(columns='diabetes', axis=1)
y = data['diabetes']

scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
x = standardized_data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Streamlit App
st.title('Diabetes Prediction Webapp')

gender = st.selectbox('Select your gender', ('Male', 'Female'))
age = st.number_input('Enter your age', min_value=0, max_value=120)
ht = st.selectbox('Do you have hypertension?', ('Yes', 'No'))
hd = st.selectbox('Do you suffer from any form of heart disease?', ('Yes', 'No'))
smoking_history = st.selectbox('What is your smoking history?', ('Never smoked', 'Former smoker', 'Current smoker', 'Do not wish to disclose'))
bmi = st.number_input('Enter your BMI')
hg_level = st.number_input('Enter your hemoglobin level')
bgl = st.number_input('Enter your blood glucose level')

# Map inputs to numerical values
gender = 1 if gender == 'Male' else 0
ht = 1 if ht == 'Yes' else 0
hd = 1 if hd == 'Yes' else 0

smoking_history_map = {
    'Never smoked': 0,
    'Former smoker': 1,
    'Current smoker': 2,
    'Do not wish to disclose': -1
}
smoking_history = smoking_history_map[smoking_history]

# Predict button
if st.button('Predict'):
    input_data = np.array([[gender, age, ht, hd, smoking_history, bmi, hg_level, bgl]])
    input_data_reshaped = scaler.transform(input_data)
    prediction = classifier.predict(input_data_reshaped)

    if prediction[0] == 0:
        st.success('The person is NOT diabetic')
    else:
        st.error('The person is diabetic')

#streamlit run diabetes_prediction_app.py
# in command prompt
