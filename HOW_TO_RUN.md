# How to Run — Thermophysical Property Predictor

## Online Access

The app is publicly available online — no installation required:

**https://thermophysical-predictor-millie20260329.streamlit.app/**

---

## Prerequisites

- **Python 3.10 or newer** must be installed on your machine.  
  Download from: https://www.python.org/downloads/  
  During installation, check **"Add Python to PATH"**.

---

## Step-by-Step Instructions

### 1. Unzip the File

Extract the zip file to any folder on your computer, for example:
```
C:\Users\YourName\Desktop\thermophysical-predictor\
```

---

### 2. Open a Terminal in That Folder

- On **Windows**: Open the folder in File Explorer, then right-click an empty area and choose **"Open in Terminal"** (or **"Open PowerShell window here"**).
- Alternatively, open Command Prompt or PowerShell and run:
  ```
  cd "C:\Users\YourName\Desktop\thermophysical-predictor"
  ```

---

### 3. Create a Virtual Environment

```
python -m venv .venv
```

This creates an isolated Python environment inside a `.venv` folder.

---

### 4. Activate the Virtual Environment

**Windows (PowerShell):**
```
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```
.venv\Scripts\activate.bat
```

**Mac / Linux:**
```
source .venv/bin/activate
```

> After activation, your terminal prompt will show `(.venv)` at the beginning.

---

### 5. Install Dependencies

```
pip install -r requirements.txt
```

This installs all required packages (Streamlit, pandas, scikit-learn, XGBoost, CatBoost, SHAP, etc.).  
This step may take a few minutes on first run.

---

### 6. Run the App

```
streamlit run app.py
```

The app will start and automatically open in your default web browser at:
```
http://localhost:8501
```

---

## Using the App

1. **Upload your CSV file** using the sidebar.  
   The file must contain these columns:

   | Column | Description |
   |--------|-------------|
   | `temp` | Temperature |
   | `loading` | Particle loading |
   | `conc` | Concentration |
   | `thcond` + `spheat` | Thermal conductivity & specific heat *(Option A)* |
   | `density` + `visc` | Density & viscosity *(Option B)* |

2. **Enter values** for a new data point (temperature, loading, concentration) in the sidebar.

3. Click **Train & Predict** to train 9 regression models and see predictions.

4. Explore results in the tabs:
   - **Feature Analysis** — correlation heatmap, feature importances, SHAP values
   - **Per-model tabs** — R², MAE, RMSE metrics, formula (where applicable), parity plots, and prediction for your new point

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` not recognized | Reinstall Python and check "Add Python to PATH" |
| `Activate.ps1 cannot be loaded` (PowerShell) | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Port 8501 already in use | Run: `streamlit run app.py --server.port 8502` |
| Package install errors | Make sure the virtual environment is activated (see step 4) |

---

## Stopping the App

Press **Ctrl + C** in the terminal to stop the Streamlit server.
