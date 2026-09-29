import os

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load trained model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "best_model.pkl"
)

model = joblib.load(MODEL_PATH)

st.title("🏠 Sydney Housing Price Prediction")
st.write(
    "Enter the basic characteristics of a Sydney property "
    "to obtain an estimated sale price."
)

st.sidebar.header("Property Information")

suburb = st.sidebar.selectbox(
    "Suburb",
    ["Blacktown", "Parramatta", "Mosman"]
)

property_type = st.sidebar.selectbox(
    "Property Type",
    ["House", "Duplex/semi-detached"]
)

bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

bathrooms = st.sidebar.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

car_spaces = st.sidebar.number_input(
    "Car Spaces",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

land_size = st.sidebar.number_input(
    "Land Size (m²)",
    min_value=50.0,
    max_value=5000.0,
    value=500.0,
    step=10.0
)

sale_year = st.sidebar.number_input(
    "Sale Year",
    min_value=2020,
    max_value=2030,
    value=2026,
    step=1
)

sale_month_num = st.sidebar.number_input(
    "Sale Month",
    min_value=1,
    max_value=12,
    value=9,
    step=1
)

# Engineered features
log_land_size = np.log1p(land_size)

bathrooms_per_bedroom = bathrooms / bedrooms

total_basic_features = bedrooms + bathrooms + car_spaces

# Create input dataframe
input_data = pd.DataFrame([{
    "suburb": suburb,
    "property_type": property_type,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "car_spaces": car_spaces,
    "land_size": land_size,
    "sale_year": sale_year,
    "sale_month_num": sale_month_num,
    "log_land_size": log_land_size,
    "bathrooms_per_bedroom": bathrooms_per_bedroom,
    "total_basic_features": total_basic_features
}])

st.subheader("Property Summary")
st.dataframe(input_data, use_container_width=True)

if st.button("Predict Sale Price", type="primary"):

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Sale Price: ${prediction:,.0f}"
    )

    st.info(
        "This prediction is an estimate based on the features "
        "provided and the training data. It should not be treated "
        "as a professional property valuation."
    )

st.markdown("---")
st.caption(
    "SIT720 8.1D — Sydney Housing Price Prediction and Decision Support System"
)
