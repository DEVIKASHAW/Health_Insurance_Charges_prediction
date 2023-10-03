import streamlit as st
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

# Load the pre-trained Gradient Boosting model
model = joblib.load("model_joblib_gb1.pkl")

# Function to predict insurance charges
def predict_charges(age, sex, bmi, children, smoker, region_northeast, region_northwest, region_southeast, region_southwest):
    # Mapping categorical variables to numerical values
    sex_mapping = {"Male": 1, "Female": 0}
    smoker_mapping = {"Yes": 1, "No": 0}

    # Encoding categorical variables
    sex_encoded = sex_mapping.get(sex, 0)
    smoker_encoded = smoker_mapping.get(smoker, 0)
    
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_encoded],
        'region_northeast': [region_northeast],
        'region_northwest': [region_northwest],
        'region_southeast': [region_southeast],
        'region_southwest': [region_southwest]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return prediction

# Streamlit UI elements
st.title("Insurance Charges Prediction")
st.sidebar.header("User Input")

# User input fields
age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.radio("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 1)
smoker = st.sidebar.radio("Smoker", ["Yes", "No"])
region_northeast = st.sidebar.checkbox("Northeast")
region_northwest = st.sidebar.checkbox("Northwest")
region_southeast = st.sidebar.checkbox("Southeast")
region_southwest = st.sidebar.checkbox("Southwest")

# Make prediction when the "Predict" button is clicked
if st.sidebar.button("Predict"):
    prediction = predict_charges(age, sex, bmi, children, smoker, region_northeast, region_northwest, region_southeast, region_southwest)
    st.subheader(f"Predicted Insurance Charges: ${prediction:.2f}")
