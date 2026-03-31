# Thermophysical Property Predictor

A Streamlit-based machine learning web application for predicting thermophysical properties of nanofluids or similar fluid systems. Given experimental or simulation data with temperature, particle loading, and concentration as inputs, the app automatically trains **nine different regression models**, evaluates them against each other using standard metrics, and lets you instantly predict properties for new conditions—all through an interactive browser interface with no coding required.

---

## Table of Contents

1. [What This Application Does](#what-this-application-does)
2. [Input Data Format](#input-data-format)
3. [Machine Learning Models](#machine-learning-models)
4. [User Interface Guide](#user-interface-guide)
5. [Quick Start](#quick-start)

---

## What This Application Does

### Purpose

This tool is designed for researchers and engineers working with thermophysical properties of fluids—particularly nanofluids, where properties like thermal conductivity, specific heat, density, and viscosity are measured as functions of temperature, nanoparticle loading (volume or weight fraction), and concentration. Collecting enough experimental data is expensive and time-consuming; this app allows you to:

- **Train predictive models** on your existing experimental dataset with a single click.
- **Compare nine regression algorithms** side-by-side to identify which one best fits your data.
- **Evaluate model performance** using rigorous metrics (R², MAE, RMSE) on a held-out test set.
- **Predict properties** for new temperature/loading/concentration conditions not in your dataset.
- **Understand your data** through correlation analysis, feature importance charts, and SHAP interpretability plots.

### Workflow

1. You prepare a CSV file with your experimental data.
2. You upload the CSV and click **Train Models**.
3. The app splits the data (80% training / 20% testing), trains all nine models, and caches the results so navigation remains instant.
4. You explore the results in the tabbed interface: feature analysis, a unified prediction table, and one dedicated tab per model.
5. You enter new conditions and click **Predict** to get predictions from all nine models simultaneously.

### Supported Property Pairs

The application detects which property pair is present in your CSV and trains two sets of models accordingly — one set per target variable:

| Mode | Target 1 | Target 2 |
|------|----------|----------|
| Thermal mode | `thcond` — thermal conductivity | `spheat` — specific heat |
| Transport mode | `density` — density | `visc` — dynamic viscosity |

All nine models are trained independently for each target, giving you 18 trained models in total per run.

### Data Splitting and Caching

The data is split into **80% training / 20% testing** using a fixed random seed (`random_state=42`) to ensure reproducibility. Once trained, all models are cached by Streamlit's resource cache (`@st.cache_resource`), so switching between tabs does not retrain anything. The cache is only invalidated if you upload a different CSV file.

---

## Input Data Format

Your CSV file must contain exactly these columns (column names are case-sensitive):

| Column | Description | Unit (typical) |
|--------|-------------|----------------|
| `temp` | Temperature of the fluid | °C or K |
| `loading` | Nanoparticle loading / volume fraction | dimensionless or % |
| `conc` | Concentration / weight fraction | dimensionless or % |
| `thcond` | Thermal conductivity | W/(m·K) |
| `spheat` | Specific heat capacity | J/(kg·K) |
| `density` | Density | kg/m³ |
| `visc` | Dynamic viscosity | Pa·s or mPa·s |

- Include either the `thcond`+`spheat` pair **or** the `density`+`visc` pair — not required to have all four.
- Extra columns beyond those listed are ignored.
- There is no hard minimum on row count, but at least 30–50 rows are recommended for meaningful model training.

Example CSV structure:
```
temp,loading,conc,thcond,spheat
25,0.01,0.50,0.612,4182
40,0.01,0.50,0.631,4175
55,0.02,0.50,0.649,4168
...
```

---

## Machine Learning Models

All nine models listed below are trained for **each target variable** every time you click Train Models. They span a broad range of complexity — from simple interpretable linear models to powerful ensemble and deep learning approaches — so you can always find the right balance between accuracy and interpretability for your data.

---

### 1. Linear Regression

**Type:** Parametric, interpretable  
**Library:** scikit-learn `LinearRegression`

The most fundamental regression model. It assumes a strictly linear relationship between the three inputs and the target:

$$\hat{y} = \beta_0 + \beta_1 \cdot \text{temp} + \beta_2 \cdot \text{loading} + \beta_3 \cdot \text{conc}$$

Coefficients are found analytically using Ordinary Least Squares (OLS). The app displays the exact fitted formula (with all coefficient values) in the model's tab. This model serves as a baseline; if more complex models do not substantially outperform it, the relationship in your data is essentially linear.

**Settings:** Default OLS — no regularisation, no hyperparameters.

---

### 2. Polynomial Regression (degree 2)

**Type:** Parametric, interpretable  
**Library:** scikit-learn `Pipeline` → `PolynomialFeatures(degree=2)` + `LinearRegression`

Extends linear regression by adding all degree-2 terms: squares and cross-products of the three inputs. For inputs `temp`, `loading`, `conc`, this creates 9 additional terms (e.g., `temp²`, `temp·loading`, `loading·conc`, etc.). The model remains linear in the *parameters*, so OLS still solves it exactly, and the full formula is displayed in the app.

This model is well-suited for properties that curve with temperature or have interaction effects between loading and concentration.

**Settings:**
| Parameter | Value |
|-----------|-------|
| `degree` | 2 |
| `include_bias` | False (intercept handled by LinearRegression) |

---

### 3. Ridge Regression

**Type:** Parametric, interpretable, regularised  
**Library:** scikit-learn `Ridge`

Identical to Linear Regression but adds an L2 penalty on the magnitude of the coefficients:

$$\text{Loss} = \sum (y_i - \hat{y}_i)^2 + \alpha \sum \beta_j^2$$

The regularisation term $\alpha \sum \beta_j^2$ shrinks large coefficients toward zero, which reduces overfitting compared to plain OLS, especially when predictors are correlated. The fitted formula is displayed in the app.

**Settings:**
| Parameter | Value | Effect |
|-----------|-------|--------|
| `alpha` | 1.0 | Moderate regularisation strength |

---

### 4. Random Forest

**Type:** Ensemble, tree-based, non-parametric  
**Library:** scikit-learn `RandomForestRegressor`

Trains a large number of decision trees on random subsets of the training data (bootstrap sampling) and random subsets of features at each split, then averages their predictions. This bagging strategy reduces variance significantly compared to a single tree. Random Forest is robust to outliers and handles non-linear interactions between variables well without any feature engineering.

**Settings:**
| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_estimators` | 200 | Number of trees in the forest |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Uses all CPU cores in parallel |

All other parameters use scikit-learn defaults (`max_depth=None` — trees grown until pure, `min_samples_split=2`, `max_features="sqrt"`).

---

### 5. Gradient Boosting

**Type:** Ensemble, tree-based, non-parametric  
**Library:** scikit-learn `GradientBoostingRegressor`

Builds trees sequentially, where each new tree corrects the residual errors of all previous trees. This boosting strategy reduces both bias and variance progressively. The learning rate controls how much each tree contributes, and shallow trees (`max_depth=3`) act as "weak learners" that prevent overfitting. Gradient Boosting often achieves high accuracy on tabular data.

**Settings:**
| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_estimators` | 300 | Number of sequential trees |
| `learning_rate` | 0.05 | Step size / shrinkage per tree |
| `max_depth` | 3 | Depth of each tree (shallow = regularisation) |
| `random_state` | 42 | Reproducibility |

---

### 6. XGBoost

**Type:** Ensemble, tree-based, gradient boosted  
**Library:** `xgboost.XGBRegressor`

An optimised and highly performant implementation of gradient boosting with several additional regularisation techniques. XGBoost adds column and row subsampling per tree (similar to Random Forest), second-order gradient statistics for faster convergence, and built-in L1/L2 regularisation. It is one of the most widely used and competition-winning algorithms for structured tabular data.

**Settings:**
| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_estimators` | 300 | Number of boosting rounds |
| `learning_rate` | 0.05 | Shrinkage per round |
| `max_depth` | 4 | Slightly deeper than GBM for more expressiveness |
| `subsample` | 0.9 | 90% of rows sampled per tree (reduces overfitting) |
| `colsample_bytree` | 0.9 | 90% of features sampled per tree |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Parallel training |

---

### 7. K-Nearest Neighbours (KNN)

**Type:** Instance-based, non-parametric, lazy learner  
**Library:** scikit-learn `KNeighborsRegressor`

Makes predictions by finding the K most similar training samples to the new input point and averaging their target values. Similarity is based on Euclidean distance in the 3D feature space. Distance-weighting means closer neighbours have more influence on the prediction. KNN makes no assumptions about the functional form of the relationship and can capture highly localised patterns.

**Settings:**
| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_neighbors` | 5 | Number of nearest neighbours |
| `weights` | `"distance"` | Weight neighbours by inverse of their distance |

Note: KNN performance is sensitive to feature scale differences; the three input features (`temp`, `loading`, `conc`) should ideally be on similar scales.

---

### 8. Neural Network (MLP)

**Type:** Deep learning, non-parametric  
**Library:** scikit-learn `MLPRegressor` wrapped in a `Pipeline` + `TransformedTargetRegressor`

A Multi-Layer Perceptron with two hidden layers. The inputs are standardised before being passed to the network, and the target variable is also standardised before training — both actions improve convergence and numerical stability. The ReLU activation function allows the network to learn complex non-linear relationships.

**Architecture and Settings:**
| Parameter | Value | Effect |
|-----------|-------|--------|
| Hidden layers | `(64, 32)` | Two layers: 64 neurons → 32 neurons |
| `activation` | `"relu"` | Rectified Linear Unit — fast and effective |
| `alpha` | 0.05 | L2 regularisation on weights (prevents overfitting) |
| `learning_rate_init` | 0.001 | Initial step size for Adam optimiser |
| `max_iter` | 2000 | Maximum training epochs |
| `random_state` | 42 | Reproducibility |
| Input scaling | `StandardScaler` | Zero mean, unit variance normalisation |
| Target scaling | `StandardScaler` (via `TransformedTargetRegressor`) | Normalises output distribution |

---

### 9. CatBoost

**Type:** Ensemble, tree-based, gradient boosted  
**Library:** `catboost.CatBoostRegressor`

A gradient boosting algorithm developed by Yandex that uses ordered boosting and oblivious (symmetric) decision trees. CatBoost is known for its strong out-of-the-box performance, built-in handling of overfitting via ordered statistics, and fast inference. `verbose=0` suppresses training logs.

**Settings:**
| Parameter | Value | Effect |
|-----------|-------|--------|
| `iterations` | 300 | Number of boosting rounds |
| `learning_rate` | 0.05 | Shrinkage per round |
| `depth` | 6 | Tree depth (symmetric/oblivious trees) |
| `random_seed` | 42 | Reproducibility |
| `verbose` | 0 | Suppresses training output |

---

### Model Comparison Summary

| Model | Type | Interpretable | Formula Shown | Strengths |
|-------|------|:---:|:---:|-----------|
| Linear Regression | Parametric | ✓ | ✓ | Baseline, fast, fully transparent |
| Polynomial Reg. deg2 | Parametric | ✓ | ✓ | Captures curvature and interactions |
| Ridge Regression | Parametric regularised | ✓ | ✓ | Better generalisation than OLS |
| Random Forest | Ensemble / bagging | — | — | Robust, handles non-linearity |
| Gradient Boosting | Ensemble / boosting | — | — | High accuracy on tabular data |
| XGBoost | Optimised boosting | — | — | State-of-the-art for tabular data |
| KNN | Instance-based | — | — | Captures local patterns |
| Neural Network | Deep learning | — | — | Flexible function approximation |
| CatBoost | Ordered boosting | — | — | Strong out-of-the-box performance |

---

## User Interface Guide

The application is divided into a **sidebar** (controls) and a **main area** (results tabs). Here is a detailed walkthrough of every section.

---

### Sidebar — Upload & Train

The sidebar is visible at all times on the left side of the screen.

**CSV File Uploader**  
A file picker that accepts `.csv` files. Drag-and-drop or browse to select your data file. The file is read into memory but never written to disk.

**Train Models button**  
Appears after a file is uploaded. Clicking it triggers the 80/20 data split and trains all 18 models (9 per target). A spinner is shown while training. Once complete, a green success banner appears in the main area confirming which targets were detected.

> If you upload a different CSV and click Train again, the cache is cleared and all models are retrained from scratch.

---

### Tab 1 — 📊 Feature Analysis

This tab provides a statistical understanding of your data before examining individual model results. It contains three sections:

#### Section 1: Correlation (inputs → outputs)

Two heatmaps are shown side by side:

- **Pearson Correlation** — measures the strength and direction of the *linear* relationship between each input feature and each target variable. Values range from -1 (perfect negative linear correlation) to +1 (perfect positive linear correlation). Values near 0 mean no linear relationship.
- **Spearman Correlation** — measures the strength of the *monotonic* (not necessarily linear) relationship, making it more robust to outliers and non-linear but monotonic trends.

Each cell displays the numerical correlation value. Cells with |value| > 0.5 are shown with white text on a coloured background for emphasis.

#### Section 2: Feature Importances (tree-based models)

A grouped bar chart showing the **normalised feature importance** of each input (`temp`, `loading`, `conc`) as assigned by the four tree-based models: Random Forest, Gradient Boosting, XGBoost, and CatBoost. Importances are normalised to sum to 1 within each model.

This answers the question: *"Which input variable has the greatest influence on each target property?"* The chart is produced separately for each target (two side-by-side plots).

#### Section 3: SHAP Values (best tree model)

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain individual model predictions. Unlike feature importances (which give global averages), SHAP shows both the **direction** and **magnitude** of each feature's contribution to each prediction.

The app auto-selects the **best-performing tree model** (by R² on the test set) for each target and computes SHAP values on the test set. Two plots appear per target:

- **Bar summary plot** — shows the mean absolute SHAP value per feature, ranking them by overall importance.
- **Beeswarm summary plot** — shows the distribution of SHAP values across all test samples, colour-coded by feature value (red = high, blue = low). This reveals whether high temperature, for example, pushes predictions up or down.

---

### Tab 2 — 🔮 Predict

A unified prediction panel for quickly querying all nine models at once.

**Input fields** (three columns):
- `Temperature` — enter the temperature value
- `Loading` — enter the particle loading value
- `Concentration` — enter the concentration value

All fields accept decimal values with 4-digit precision (e.g., `55.0000`).

**Predict button**  
Runs all nine trained models on the entered values and displays a single results table with one row per model and columns for each target property. This makes it easy to compare model predictions side by side and identify any outliers or disagreements between models.

---

### Tabs 3–11 — Individual Model Tabs

Each of the nine models has its own dedicated tab. The tabs are labelled with the model name and appear in the order shown above. Each tab contains the following sections:

#### Metrics — test set (80/20 split)

A metrics table is shown for **each target variable** side by side, containing three evaluation metrics computed on the 20% held-out test set:

| Metric | Full Name | Meaning |
|--------|-----------|---------|
| **R²** | Coefficient of Determination | Proportion of variance explained. 1.0 = perfect fit. Values < 0 mean the model is worse than predicting the mean. |
| **MAE** | Mean Absolute Error | Average absolute difference between predicted and true values. Expressed in the same units as the target. |
| **RMSE** | Root Mean Squared Error | Similar to MAE but penalises large errors more heavily due to squaring. Also in target units. |

#### Formula (Linear, Polynomial, and Ridge models only)

For the three interpretable parametric models, the exact fitted equation is displayed as a code block with all numerical coefficients. This allows you to reproduce predictions without the software or copy the formula into a spreadsheet. For example:

```
thcond = 0.512341 + 0.003214·temp + 0.812500·loading + 0.045100·conc
```

For the Polynomial model, all 9 degree-2 terms and their coefficients are shown.

#### Parity Plots

Two scatter plots are shown per target variable (four plots total per model tab):

- **Test data parity plot** — true values (x-axis) vs. predicted values (y-axis) on the 20% test set. A perfect model would have all points falling exactly on the red dashed diagonal line (the 1:1 line). Scatter around this line indicates prediction error; systematic deviation indicates bias.
- **Full dataset parity plot** — the same plot but using all data (training + test). Comparing the two plots reveals if a model is overfitting (much better on training data than test data).

Both plots include axis labels using the actual target variable name, a grid, and alpha-blended markers to show point density.

---

## Quick Start

### Online Access

The app is publicly available — no installation required:

**https://thermophysical-predictor-millie20260329.streamlit.app/**

### Local Setup

#### Requirements
- Python 3.10 or newer  
  See [HOW_TO_RUN.md](HOW_TO_RUN.md) for complete step-by-step installation instructions.

#### Steps

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Run the app**
   ```
   streamlit run app.py
   ```

3. **Open your browser** at `http://localhost:8501`
