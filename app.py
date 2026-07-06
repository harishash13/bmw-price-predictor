import streamlit as st
import joblib
import numpy as np

# ============================================
# Load the saved model
# ============================================
rf_model = joblib.load('bmw_price_model.pkl')

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="BMW Price Predictor",
    page_icon="car",
    layout="centered"
)

# ============================================
# Header
# ============================================
st.title("BMW Used Car Price Predictor")
st.markdown("Enter the details of a used BMW to get an estimated resale price.")
st.markdown("---")

# ============================================
# Input Form
# ============================================
col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("BMW Model", [
        '1 Series', '2 Series', '3 Series', '4 Series', '5 Series',
        '6 Series', '7 Series', '8 Series', 'M2', 'M3', 'M4', 'M5',
        'M6', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'Z3', 'Z4', 'i3', 'i8'
    ])
    year = st.slider("Year of Registration",
                     min_value=1996, max_value=2020, value=2018)
    transmission_type = st.selectbox("Transmission",
                                     ['Automatic', 'Manual', 'Semi-Auto'])
    fuel_type = st.selectbox("Fuel Type",
                             ['Diesel', 'Electric', 'Hybrid', 'Other', 'Petrol'])

with col2:
    mileage = st.number_input("Mileage (miles)",
                              min_value=0, max_value=250000, value=20000, step=1000)
    engine_size = st.selectbox("Engine Size (L)",
                               [1.5, 2.0, 3.0, 4.4])
    mpg = st.number_input("MPG",
                          min_value=10.0, max_value=150.0, value=50.0, step=0.1)
    tax = st.number_input("Annual Tax (£)",
                          min_value=0, max_value=600, value=145, step=5)

st.markdown("---")

# ============================================
# Predict Button
# ============================================
if st.button("Predict Price", use_container_width=True):

    all_models = ['1 Series', '2 Series', '3 Series', '4 Series', '5 Series',
                  '6 Series', '7 Series', '8 Series', 'M2', 'M3', 'M4', 'M5',
                  'M6', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'Z3', 'Z4', 'i3', 'i8']
    all_transmissions = ['Automatic', 'Manual', 'Semi-Auto']
    all_fueltypes = ['Diesel', 'Electric', 'Hybrid', 'Other', 'Petrol']

    model_encoded = all_models.index(model_name)
    transmission_encoded = all_transmissions.index(transmission_type)
    fueltype_encoded = all_fueltypes.index(fuel_type)

    input_data = np.array([[model_encoded, year, transmission_encoded,
                            mileage, fueltype_encoded, tax, mpg, engine_size]])

    predicted_price = rf_model.predict(input_data)[0]

    # ============================================
    # Display Result
    # ============================================
    st.success(f"Estimated Price: £{predicted_price:,.0f}")
    st.markdown("---")

    # Show input summary
    st.markdown("### Configuration Summary")
    col3, col4 = st.columns(2)
    with col3:
        st.write(f"**Model:** BMW {model_name}")
        st.write(f"**Year:** {year}")
        st.write(f"**Transmission:** {transmission_type}")
        st.write(f"**Fuel Type:** {fuel_type}")
    with col4:
        st.write(f"**Mileage:** {mileage:,} miles")
        st.write(f"**Engine Size:** {engine_size}L")
        st.write(f"**MPG:** {mpg}")
        st.write(f"**Tax:** £{tax}")

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("*Built with Random Forest — 95.93% accuracy on 10,664 BMW listings*")
