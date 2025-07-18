import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('../models/house_model.pkl')

st.title("🏠 House Price Prediction App")
st.write("Enter the features below to predict house price:")

# Input fields
area = st.number_input("Area (in sq ft)", value=1000)
bedrooms = st.slider("Number of Bedrooms", 1, 10, value=2)
bathrooms = st.slider("Number of Bathrooms", 1, 10, value=1)
stories = st.slider("Number of Stories", 1, 4, value=1)

# Binary fields
mainroad = st.selectbox("Main Road Access?", ['yes', 'no'])
guestroom = st.selectbox("Guest Room?", ['yes', 'no'])
basement = st.selectbox("Basement?", ['yes', 'no'])
hotwaterheating = st.selectbox("Hot Water Heating?", ['yes', 'no'])
airconditioning = st.selectbox("Air Conditioning?", ['yes', 'no'])
prefarea = st.selectbox("Preferred Area?", ['yes', 'no'])

# Categorical: Furnishing status
furnishingstatus = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# When button is clicked
if st.button("Predict Price 💰"):
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [1 if mainroad == 'yes' else 0],
        'guestroom': [1 if guestroom == 'yes' else 0],
        'basement': [1 if basement == 'yes' else 0],
        'hotwaterheating': [1 if hotwaterheating == 'yes' else 0],
        'airconditioning': [1 if airconditioning == 'yes' else 0],
        'prefarea': [1 if prefarea == 'yes' else 0],
        'furnishingstatus': [furnishingstatus]
    })

    # Convert categorical to dummy (one-hot)
    input_data = pd.get_dummies(input_data)

    # Align with training data columns
    model_input_columns = model.feature_names_in_
    for col in model_input_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_input_columns]

    # Predict
    price = model.predict(input_data)[0]
    st.success(f"🏡 Estimated Price: ₹{price:,.2f}")
