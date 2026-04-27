import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

@st.cache_resource
def load_model():
    with open("car_price_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🚗 Car Price Prediction App")
st.markdown("Predict the resale price of a used car using Machine Learning.")

current_year = datetime.now().year

present_price = st.number_input("Present Price (in Lakhs)", min_value=0.5, max_value=100.0, value=8.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
year = st.slider("Manufacturing Year", 2000, current_year, 2018)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

if st.button("Predict Price"):
    car_age = current_year - year

    input_data = pd.DataFrame({
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Owner": [owner],
        "Car_Age": [car_age],
        "Fuel_Type_Diesel": [1 if fuel_type == "Diesel" else 0],
        "Fuel_Type_Petrol": [1 if fuel_type == "Petrol" else 0],
        "Seller_Type_Individual": [1 if seller_type == "Individual" else 0],
        "Transmission_Manual": [1 if transmission == "Manual" else 0]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
