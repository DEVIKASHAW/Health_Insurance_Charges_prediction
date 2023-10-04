import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained Gradient Boosting model
model = joblib.load("df.pkl")

# Function to predict insurance charges
def predict_charges(age, sex, bmi, children, smoker, region):
    # Mapping categorical variables to numerical values
    sex_mapping = {"Male": 1, "Female": 0}
    smoker_mapping = {"Yes": 1, "No": 0}
    region_mapping = {
        "Northeast": 0,
        "Northwest": 1,
        "Southeast": 2,
        "Southwest": 3
    }

    # Encoding categorical variables
    sex_encoded = sex_mapping.get(sex, 0)
    smoker_encoded = smoker_mapping.get(smoker, 0)
    region_encoded = region_mapping.get(region, 0)
    
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_encoded],
        'region_northeast': [1 if region_encoded == 0 else 0],
        'region_northwest': [1 if region_encoded == 1 else 0],
        'region_southeast': [1 if region_encoded == 2 else 0],
        'region_southwest': [1 if region_encoded == 3 else 0]
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
region = st.sidebar.radio("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Make prediction when the "Predict" button is clicked
if st.sidebar.button("Predict"):
    prediction = predict_charges(age, sex, bmi, children, smoker, region)
    st.subheader(f"Predicted Insurance Charges: ${prediction:.2f}")

