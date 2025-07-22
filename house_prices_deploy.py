import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import kagglehub
@st.cache_data
def load_model():
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
    file_path = os.path.join(path, 'Housing.csv')
    data = pd.read_csv(file_path)
    data = data.drop(columns=['mainroad', 'hotwaterheating', 'prefarea'])

    # Encode
    data = pd.get_dummies(data, columns=['guestroom', 'basement'], drop_first=True)
    data = pd.get_dummies(data, columns=['airconditioning'], drop_first=True)
    encoder = OrdinalEncoder(categories=[['furnished', 'unfurnished', 'semi-furnished']])
    data[['furnishingstatus']] = encoder.fit_transform(data[['furnishingstatus']])

    X = data.drop(columns=['price'])
    y = data['price']  # 1D → model.predict returns 1D

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=10)
    model.fit(X_scaled, y)

    return model, scaler, encoder
model, scaler, encoder = load_model()

st.title("🏠 House Price Predictor")

st.write("Enter the details of the house below:")

with st.form("prediction_form"):
    area = st.number_input("Area (sqft)", min_value=300, max_value=10000, step=50)
    bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10)
    bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=10)
    stories = st.number_input("Number of stories", min_value=1, max_value=5)
    parking = st.number_input("Number of parking spots", min_value=0, max_value=5)
    guestroom = st.radio("Guestroom", options=["yes", "no"])
    basement = st.radio("Basement", options=["yes", "no"])
    airconditioning = st.radio("Airconditioning", options=["yes", "no"])
    furnishingstatus = st.selectbox("Furnishing status", options=["furnished", "unfurnished", "semi-furnished"])

    submit = st.form_submit_button("Predict Price")

    if submit:
        guestroom_yes = 1 if guestroom == "yes" else 0
        basement_yes = 1 if basement == "yes" else 0
        airconditioning_yes = 1 if airconditioning == "yes" else 0
        furnishing_encoded = encoder.transform([[furnishingstatus]])[0][0]
        input_array = np.array([[area, bedrooms, bathrooms, stories, parking,
                                 furnishing_encoded, guestroom_yes, basement_yes, airconditioning_yes]])
        input_scaled = scaler.transform(input_array)

        predicted_price = model.predict(input_scaled).flatten()[0]
        st.success(f" Predicted House Price: ₹ {predicted_price:,.2f}")