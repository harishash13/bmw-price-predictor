# BMW Used Car Price Predictor

A machine learning web application that predicts the fair resale price of a used BMW based on its specifications, checks whether a specific listing is a good deal, and explains exactly why a price was predicted.

**Live App:** https://bmw-price-predictor-harish.streamlit.app

---

## Overview

The used car market suffers from information asymmetry — sellers may overprice cars out of emotional attachment, while buyers may undervalue cars by not accounting for key specifications like engine size, mileage, or fuel type. This project addresses that problem using a data-driven price prediction model trained on 10,664 UK BMW listings.

---

## Features

- **Price Prediction** — enter a BMW's specifications and get an estimated fair market price
- **Good Deal Checker** — enter a real listing price to see if it is overpriced, fairly priced, or a good deal
- **Explainability (SHAP)** — see exactly which features pushed the predicted price up or down for that specific car
- **95.93% model accuracy** (Random Forest Regressor)

---

## Dataset

- **Primary dataset:** BMW Used Car Listings (10,781 rows, 9 columns, UK market, 1996–2020)
- **Supplementary dataset:** [BMW Used Car Listing by MySarahMadBhat](https://www.kaggle.com/datasets/mysarahmadbhat/bmw-used-car-listing) — merged to expand available features
- **Final cleaned dataset:** 10,664 rows, 9 columns, zero nulls, zero duplicates

---

## Features Used for Prediction

| Feature | Description |
|---|---|
| model | BMW model series (e.g. 3 Series, X5, M4) |
| year | Year of registration |
| transmission | Manual, Automatic, or Semi-Auto |
| mileage | Total distance driven (miles) |
| fuelType | Diesel, Petrol, Hybrid, Electric, Other |
| tax | Annual road tax (£) |
| mpg | Fuel efficiency (miles per gallon) |
| engineSize | Engine displacement (litres) |

---

## Model Performance

| Metric | Random Forest | XGBoost |
|---|---|---|
| MAE | £1,459 | £1,559 |
| RMSE | £2,301 | £2,315 |
| R² Score | 95.93% | 95.88% |

Random Forest was selected as the production model based on its slightly better performance across all three metrics.

---

## Top Price Drivers (Feature Importance)

1. **year** — 39.38%
2. **engineSize** — 25.44%
3. **model** — 15.62%
4. **mileage** — 12.62%
5. mpg — 4.40%
6. tax — 1.32%
7. transmission — 0.67%
8. fuelType — 0.54%

---

## Tech Stack

- **Python** — core language
- **Pandas** — data cleaning and manipulation
- **Scikit-learn** — Random Forest model
- **XGBoost** — comparison model
- **SHAP** — model explainability
- **Matplotlib** — visualizations
- **Streamlit** — web application and deployment
- **Joblib** — model serialization

---

## Project Structure

```
bmw-price-predictor/
├── app.py                    # Streamlit web application
├── bmw_price_model.pkl       # Trained and compressed Random Forest model
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Running Locally

1. Clone this repository
```bash
git clone https://github.com/harishash13/bmw-price-predictor.git
cd bmw-price-predictor
```

2. Install dependencies
```bash
pip3 install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

The app will open automatically in your browser at `localhost:8501`.

---

## How It Works

1. User enters a BMW's specifications through the web form
2. Categorical inputs (model, transmission, fuel type) are encoded to match the training format
3. The trained Random Forest model predicts a fair market price
4. If a listing price is provided, it is compared against the prediction to flag overpriced, fair, or good-deal listings
5. SHAP values are calculated to show how each feature contributed to that specific prediction

---

## Limitations

- Trained only on UK market BMW data — not applicable to other brands or markets
- Supports only the 24 BMW models present in the training data
- Covers registration years 1996–2020
- Engine size options limited to 1.5L, 2.0L, 3.0L, and 4.4L

---

## Future Improvements

- Price range prediction (confidence intervals) instead of a single number
- Depreciation forecasting over future years
- Comparison against similar cars in the dataset
- Negotiation suggestion based on predicted vs listed price

---

## Author

Built by Harish as part of a data science and machine learning project covering the full pipeline: data merging, cleaning, exploratory data analysis, statistical testing, model building, explainability, and deployment.
