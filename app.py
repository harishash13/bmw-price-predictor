# ================================================================
# BMW Used Car Price Predictor - Streamlit Web Application
# ================================================================
# This app loads a pre-trained Random Forest model and predicts
# the resale price of a used BMW based on 8 input features.
# It also includes:
#   1. A "Good Deal Checker" - compares a listed price to the
#      model's predicted fair price
#   2. SHAP Explainability - shows why the model predicted that
#      specific price for that specific car
# ================================================================

import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# ================================================================
# STEP 1: Load the pre-trained model
# ================================================================
# The model was trained separately and saved as a .pkl file.
# Loading it here means we do not need to retrain every time
# the app runs.
rf_model = joblib.load('bmw_price_model.pkl')

# ================================================================
# STEP 2: Page Configuration
# ================================================================
st.set_page_config(
    page_title="BMW Price Predictor",
    page_icon="car",
    layout="centered"
)

# ================================================================
# STEP 3: Header Section
# ================================================================
st.title("BMW Used Car Price Predictor")
st.markdown("Enter the details of a used BMW to get an estimated resale price.")
st.markdown("---")

# ================================================================
# STEP 4: Input Form
# ================================================================
# Two columns are used to keep the form compact and organised.
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

# ================================================================
# STEP 5: Optional Input - Check a Specific Listing
# ================================================================
# This lets the user enter what a car is ACTUALLY listed for,
# so the app can tell them if it's a good deal or overpriced.
st.markdown("### Optional: Check a Specific Listing")
listed_price = st.number_input(
    "Enter the price this car is currently listed for (£) - leave as 0 to skip",
    min_value=0, max_value=200000, value=0, step=500
)

st.markdown("---")

# ================================================================
# STEP 6: Predict Button and Logic
# ================================================================
if st.button("Predict Price", use_container_width=True):

    # ------------------------------------------------------------
    # 6.1: Encode categorical inputs
    # ------------------------------------------------------------
    # The model was trained on numbers, not text, so we convert
    # each categorical selection into the same numeric code used
    # during training (Label Encoding).
    all_models = ['1 Series', '2 Series', '3 Series', '4 Series', '5 Series',
                  '6 Series', '7 Series', '8 Series', 'M2', 'M3', 'M4', 'M5',
                  'M6', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'Z3', 'Z4', 'i3', 'i8']
    all_transmissions = ['Automatic', 'Manual', 'Semi-Auto']
    all_fueltypes = ['Diesel', 'Electric', 'Hybrid', 'Other', 'Petrol']

    model_encoded = all_models.index(model_name)
    transmission_encoded = all_transmissions.index(transmission_type)
    fueltype_encoded = all_fueltypes.index(fuel_type)

    # ------------------------------------------------------------
    # 6.2: Build the input array in the same column order as training
    # ------------------------------------------------------------
    input_data = np.array([[model_encoded, year, transmission_encoded,
                            mileage, fueltype_encoded, tax, mpg, engine_size]])

    # ------------------------------------------------------------
    # 6.3: Make the prediction
    # ------------------------------------------------------------
    predicted_price = rf_model.predict(input_data)[0]

    st.success(f"Estimated Fair Price: £{predicted_price:,.0f}")

    # ------------------------------------------------------------
    # 6.4: Good Deal Checker
    # ------------------------------------------------------------
    # Only runs if the user entered a listed price (i.e. did not
    # leave it at 0).
    if listed_price > 0:
        difference = listed_price - predicted_price
        percent_diff = (difference / predicted_price) * 100

        st.markdown("---")
        st.markdown("### Deal Analysis")

        if percent_diff > 10:
            st.error(
                f"This listing is priced £{difference:,.0f} ({percent_diff:.1f}%) "
                f"ABOVE market value. This may be OVERPRICED."
            )
        elif percent_diff < -10:
            st.success(
                f"This listing is priced £{abs(difference):,.0f} ({abs(percent_diff):.1f}%) "
                f"BELOW market value. This may be a GOOD DEAL."
            )
        else:
            st.info(
                f"This listing is priced close to market value "
                f"(difference: £{difference:,.0f}, {percent_diff:.1f}%). This is a FAIR price."
            )

    # ------------------------------------------------------------
    # 6.5: SHAP Explainability
    # ------------------------------------------------------------
    # Explains WHY the model predicted this specific price by
    # showing how much each feature pushed the price up or down
    # for this exact car (not just in general).
    st.markdown("---")
    st.markdown("### Why This Price? (Feature Contribution)")

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(input_data)

    feature_names = ['model', 'year', 'transmission', 'mileage',
                     'fuelType', 'tax', 'mpg', 'engineSize']

    # Pair each feature with its contribution and sort by impact size
    contributions = list(zip(feature_names, shap_values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    names = [c[0] for c in contributions]
    values = [c[1] for c in contributions]
    colors = ['#2a9d4f' if v > 0 else '#e63946' for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Contribution to Price (£)')
    ax.set_title('How Each Feature Affected This Prediction')
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown(
        "**Reading this chart:** Green bars pushed the price UP. "
        "Red bars pushed the price DOWN. Longer bars mean bigger "
        "impact on this specific prediction."
    )

    # ------------------------------------------------------------
    # 6.6: Configuration Summary
    # ------------------------------------------------------------
    st.markdown("---")
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

# ================================================================
# STEP 7: Footer
# ================================================================
st.markdown("---")
st.markdown("*Built with Random Forest — 95.93% accuracy on 10,664 BMW listings*")
