# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib


# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

# Load neighborhoods from training data
@st.cache_data
def load_neighborhoods():
    df = pd.read_csv("cleaned_buenos_aires_data.csv")  # Or "clean_data.csv" if renamed
    return sorted(df["neighborhood"].dropna().unique())


# Page layout
st.title("🏙️ Buenos Aires Apartment Price Predictor")

# Load model and data
model = load_model()
neighborhoods = load_neighborhoods()

# User input
st.header("Property Details")
neighborhood = st.selectbox("Neighborhood", neighborhoods)
surface = st.number_input("Surface Area (m²)", min_value=10, max_value=500, value=50)
rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=2)
lat = st.number_input("Latitude", format="%.6f", value=-34.603722)
lon = st.number_input("Longitude", format="%.6f", value=-58.381592)

# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "neighborhood": neighborhood,
        "surface_covered_in_m2": surface,
        "rooms": rooms,
        "lat": lat,
        "lon": lon,
    }])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ${prediction:,.2f} USD")
