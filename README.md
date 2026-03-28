# Thermophysical Property Predictor

A Streamlit web app that trains 9 regression models to predict thermophysical properties (thermal conductivity & specific heat, or density & viscosity) from temperature, particle loading, and concentration.

## Quick Start

### Requirements
- Python 3.10 or newer

### Steps

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Run the app**
   ```
   streamlit run app.py
   ```

3. **Open your browser** at `http://localhost:8501`

## How to Use

1. Upload your CSV file in the sidebar.  
   The file must contain columns: `temp`, `loading`, `conc`, and either:
   - `thcond` + `spheat`  (thermal conductivity & specific heat), or
   - `density` + `visc`  (density & viscosity)

2. Enter values for a new data point (temperature, loading, concentration).

3. Click **Train & Predict**.

## Tabs

| Tab | Contents |
|-----|----------|
| 📊 Feature Analysis | Correlation heatmap, feature importances, SHAP values |
| Per-model tabs | Metrics (R², MAE, RMSE), formula (where applicable), prediction for new point, parity plots |
