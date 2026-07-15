# ================================================================
# BMW Used Car Price Predictor - Streamlit Web Application
# ================================================================
# Predicts the fair resale price of a used BMW, scores a specific
# listing against that prediction, and explains the prediction
# using SHAP values.
# ================================================================

import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Constants - single source of truth for encoding, used everywhere
# ----------------------------------------------------------------
MODEL_PATH = "bmw_price_model.pkl"

ALL_MODELS = [
    "1 Series", "2 Series", "3 Series", "4 Series", "5 Series",
    "6 Series", "7 Series", "8 Series", "M2", "M3", "M4", "M5",
    "M6", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z3", "Z4", "i3", "i8"
]
ALL_TRANSMISSIONS = ["Automatic", "Manual", "Semi-Auto"]
ALL_FUELTYPES = ["Diesel", "Electric", "Hybrid", "Other", "Petrol"]
FEATURE_NAMES = ["model", "year", "transmission", "mileage",
                 "fuelType", "tax", "mpg", "engineSize"]

DEAL_OVERPRICED_THRESHOLD = 10   # percent
DEAL_GOOD_THRESHOLD = -10        # percent


# ================================================================
# CACHED RESOURCES
# ================================================================
# @st.cache_resource ensures the model and SHAP explainer are
# loaded/built ONCE per server session, not on every button click.
# Streamlit reruns the whole script on every interaction, so without
# this the model would be re-read from disk every single time.
# ================================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


# ================================================================
# CORE LOGIC (kept separate from UI so it is testable/reusable)
# ================================================================

def encode_inputs(model_name, transmission_type, fuel_type):
    """Convert categorical selections to the numeric codes the model
    was trained on. Raises a clear error for any value the model has
    never seen, instead of failing with a cryptic ValueError."""
    try:
        return (
            ALL_MODELS.index(model_name),
            ALL_TRANSMISSIONS.index(transmission_type),
            ALL_FUELTYPES.index(fuel_type),
        )
    except ValueError as e:
        st.error(
            f"'{e}' is not a category this model was trained on. "
            "Please choose a value from the dropdown."
        )
        st.stop()


def predict_with_confidence(model, input_data):
    """Return a point prediction plus a rough confidence range,
    computed from the spread of predictions across the Random
    Forest's individual trees (no retraining required)."""
    point_estimate = model.predict(input_data)[0]
    tree_predictions = np.array(
        [tree.predict(input_data)[0] for tree in model.estimators_]
    )
    std_dev = tree_predictions.std()
    lower = max(0, point_estimate - std_dev)
    upper = point_estimate + std_dev
    return point_estimate, lower, upper


def compute_deal_score(listed_price, predicted_price):
    """Score a listing from 0-100, where 50 = priced exactly at the
    model's estimate, 100 = a very good deal, 0 = significantly
    overpriced. Also returns a plain-language verdict and a
    suggested opening offer when the car looks overpriced."""
    percent_diff = ((listed_price - predicted_price) / predicted_price) * 100
    raw_score = 50 - (percent_diff * 2)
    deal_score = int(np.clip(raw_score, 0, 100))

    if percent_diff > DEAL_OVERPRICED_THRESHOLD:
        verdict = "Overpriced"
        suggested_offer = predicted_price * 1.03  # small margin above fair value
    elif percent_diff < DEAL_GOOD_THRESHOLD:
        verdict = "Good Deal"
        suggested_offer = None
    else:
        verdict = "Fair Price"
        suggested_offer = None

    return deal_score, verdict, percent_diff, suggested_offer


def plot_shap_contributions(shap_values, feature_names):
    """Bar chart of how each feature pushed this specific prediction
    up or down. Sorted by impact so the most influential feature is
    always at the top."""
    contributions = sorted(
        zip(feature_names, shap_values[0]), key=lambda x: abs(x[1]), reverse=True
    )
    names = [c[0] for c in contributions]
    values = [c[1] for c in contributions]
    colors = ["#2a9d4f" if v > 0 else "#e63946" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to Price (£)")
    ax.set_title("How Each Feature Affected This Prediction")
    plt.tight_layout()
    return fig, contributions


def describe_top_driver(contributions):
    """Turn the single biggest SHAP contribution into one plain
    English sentence, so a non-technical reader gets the headline
    finding without needing to interpret a chart."""
    top_feature, top_value = contributions[0]
    direction = "increased" if top_value > 0 else "decreased"
    return (
        f"The biggest factor in this prediction was **{top_feature}**, "
        f"which {direction} the estimated price by roughly £{abs(top_value):,.0f}."
    )


# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(page_title="BMW Price Predictor",
                   page_icon="car", layout="centered")

st.title("BMW Used Car Price Predictor")
st.caption("Estimate a fair resale price, check a listing, and see why.")
st.divider()

# ================================================================
# INPUT FORM
# ================================================================
col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("BMW Model", ALL_MODELS)
    year = st.slider("Year of Registration", min_value=1996,
                     max_value=2020, value=2018)
    transmission_type = st.selectbox("Transmission", ALL_TRANSMISSIONS)
    fuel_type = st.selectbox("Fuel Type", ALL_FUELTYPES)

with col2:
    mileage = st.number_input(
        "Mileage (miles)", min_value=0, max_value=250000, value=20000, step=1000)
    engine_size = st.selectbox("Engine Size (L)", [1.5, 2.0, 3.0, 4.4])
    mpg = st.number_input("MPG", min_value=10.0,
                          max_value=150.0, value=50.0, step=0.1)
    tax = st.number_input("Annual Tax (£)", min_value=0,
                          max_value=600, value=145, step=5)

st.divider()
st.subheader("Optional: Check a Specific Listing")
listed_price = st.number_input(
    "Price this car is currently listed for (£) - leave at 0 to skip",
    min_value=0, max_value=200000, value=0, step=500
)

st.divider()

# ================================================================
# PREDICTION
# ================================================================
if st.button("Predict Price", use_container_width=True, type="primary"):

    with st.spinner("Running the model..."):
        rf_model = load_model()

        model_encoded, transmission_encoded, fueltype_encoded = encode_inputs(
            model_name, transmission_type, fuel_type
        )
        input_data = np.array([[
            model_encoded, year, transmission_encoded,
            mileage, fueltype_encoded, tax, mpg, engine_size
        ]])

        predicted_price, lower_bound, upper_bound = predict_with_confidence(
            rf_model, input_data)

    # ------------------------------------------------------------
    # Result: headline metric + confidence range
    # ------------------------------------------------------------
    st.metric("Estimated Fair Price", f"£{predicted_price:,.0f}")
    st.caption(f"Likely range: £{lower_bound:,.0f} - £{upper_bound:,.0f}")

    # ------------------------------------------------------------
    # Deal Analysis
    # ------------------------------------------------------------
    if listed_price > 0:
        st.divider()
        st.subheader("Deal Analysis")

        deal_score, verdict, percent_diff, suggested_offer = compute_deal_score(
            listed_price, predicted_price
        )

        score_col, verdict_col = st.columns([1, 2])
        with score_col:
            st.metric("Deal Score", f"{deal_score}/100")
        with verdict_col:
            if verdict == "Overpriced":
                st.error(
                    f"{verdict} by roughly {abs(percent_diff):.1f}% "
                    f"(£{listed_price - predicted_price:,.0f} above the estimate)."
                )
                st.write(
                    f"Suggested opening offer: **£{suggested_offer:,.0f}**")
            elif verdict == "Good Deal":
                st.success(
                    f"{verdict} - priced roughly {abs(percent_diff):.1f}% "
                    f"below the estimate."
                )
            else:
                st.info(f"{verdict} - within a normal range of the estimate.")

    # ------------------------------------------------------------
    # SHAP Explainability
    # ------------------------------------------------------------
    st.divider()
    st.subheader("Why This Price?")

    with st.spinner("Explaining the prediction..."):
        explainer = get_shap_explainer(rf_model)
        shap_values = explainer.shap_values(input_data)
        fig, contributions = plot_shap_contributions(
            shap_values, FEATURE_NAMES)

    st.write(describe_top_driver(contributions))
    st.pyplot(fig)
    st.caption(
        "Green bars increased the price, red bars decreased it. "
        "Longer bars had a bigger effect on this specific car."
    )

    # ------------------------------------------------------------
    # Configuration Summary
    # ------------------------------------------------------------
    st.divider()
    st.subheader("Configuration Summary")
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

st.divider()
st.caption("Built with Random Forest - 95.93% R² on 10,664 BMW listings.")
