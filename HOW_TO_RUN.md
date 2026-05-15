# How to Run — Thermophysical Property Predictor

For model descriptions, data format, and repository layout, see [README.md](README.md).

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

### 6. Run the Streamlit App

```
streamlit run app.py
```

The app will start and usually open in your default web browser at:

```
http://localhost:8501
```

---

### 7. (Optional) Run the Command-Line Training Script

If you prefer a script that prints metrics and shows Matplotlib figures instead of the browser UI:

```
python 5d.py
```

- **Locally:** a file dialog asks for your CSV. A small window then asks which of `temp`, `loading`, and `conc` to use as inputs (or use the console prompt if no GUI is available).
- **Google Colab:** the script uses Colab’s file upload when `google.colab` is available.

Use the same CSV rules as the app (see [README.md](README.md)). CatBoost may write a `catboost_info` folder; you can delete it or leave it untracked (it is listed in `.gitignore` in this repo).

---

## Using the App

1. **Upload your CSV** in the sidebar (**CSV file**).  
   Your file must include whichever inputs you plan to use from `temp`, `loading`, and `conc`, plus **either** the pair `thcond` and `spheat` **or** the pair `density` and `visc` (see [README.md](README.md) for the full column list).

2. Under **Independent variables**, choose one or more of `temp`, `loading`, and `conc`. Training and plots use only the columns you select.

3. Click **Train Models** (🔧 Train Models). Wait until training finishes; a success message appears when models are ready.

4. Open the **Predict** tab (🔮 Predict) in the main area. Enter values for each selected input, then click **Predict** (🔮 Predict) to see all nine models’ outputs in one table.

5. Explore other tabs:
   - **Feature Analysis** — correlations, tree-based feature importances, SHAP summaries
   - **One tab per model** — test-set R², MAE, RMSE; fitted formulas for linear / polynomial / ridge models; parity plots

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` not recognized | Reinstall Python and check "Add Python to PATH" |
| `Activate.ps1 cannot be loaded` (PowerShell) | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Port 8501 already in use | Run: `streamlit run app.py --server.port 8502` |
| Package install errors | Make sure the virtual environment is activated (see step 4) |
| `5d.py` exits or skips the file / feature GUI | On a headless server, Tk may be unavailable; run from a desktop session or see the script’s non-interactive fallback (console prompt or all available inputs). |
| Matplotlib windows do not appear (`5d.py`) | Run from an environment with a display (local desktop). On remote Linux, configure a display or save figures by adapting the script. |

---

## Stopping the App

Press **Ctrl + C** in the terminal to stop the Streamlit server.
