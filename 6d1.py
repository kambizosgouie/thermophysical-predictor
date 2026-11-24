# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import io
import math
import zipfile
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from openpyxl.utils.exceptions import InvalidFileException
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from data_utils import excel_sheet_names, numeric_data_frame, read_tabular_data
from metrics_utils import compute_regression_metrics
from plot_utils import correlation_figure, parity_figure


st.set_page_config(page_title="General Property Predictor 6d1", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #091620;
            --bg-2: #102534;
            --panel: #111f2d;
            --panel-strong: #153047;
            --primary: #6ad3ff;
            --secondary: #5de4c3;
            --text: #eaf8ff;
            --muted: #bfd7e8;
            --border: rgba(106, 211, 255, 0.45);
            --shadow: rgba(93, 228, 195, 0.32);
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
            color: var(--text);
            font-family: "Verdana", "Segoe UI", sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            max-width: 1500px;
        }

        h1, h2, h3, h4, h5, h6,
        .stTabs [role="tab"] {
            font-family: "Times New Roman", Georgia, serif;
            letter-spacing: 0.02em;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(17, 31, 45, 0.98), rgba(11, 22, 32, 0.98));
            border-right: 1px solid var(--border);
        }

        .stButton > button {
            border: 0;
            border-radius: 0.75rem;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            color: #072231;
            font-weight: 700;
            box-shadow: 0 8px 22px var(--shadow);
        }

        .stButton > button:hover {
            filter: brightness(1.05);
        }

        .stButton > button:focus {
            box-shadow: 0 0 0 2px rgba(106, 211, 255, 0.7), 0 8px 22px var(--shadow);
        }

        .stDataFrame, .stSelectbox, .stMultiSelect, .stNumberInput, [data-testid="stFileUploader"] section {
            background: rgba(17, 31, 45, 0.9);
            border: 1px solid var(--border);
            border-radius: 0.8rem;
        }

        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }

        .stTabs [role="tab"] {
            background: rgba(17, 31, 45, 0.7);
            border: 1px solid var(--border);
            color: var(--muted);
            border-radius: 0.7rem 0.7rem 0 0;
            padding: 0.5rem 0.9rem;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, rgba(106, 211, 255, 0.15), rgba(93, 228, 195, 0.12));
            border-color: rgba(106, 211, 255, 0.7);
            color: var(--text);
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("General Property Predictor — 6d1")

TARGET_UNITS: dict[str, str] = {
    "thcond": "W/(m·K)",
    "spheat": "J/(kg·K)",
    "density": "kg/m³",
    "visc": "mPa·s",
}
TREE_MODELS = {"Random Forest", "Gradient Boosting", "XGBoost", "CatBoost"}
MODEL_NAMES = [
    "Linear Regression",
    "Polynomial Regression deg2",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "Ridge Regression",
    "KNN",
    "Neural Network",
    "CatBoost",
]


def make_models():
    return [
        ("Linear Regression", LinearRegression()),
        (
            "Polynomial Regression deg2",
            Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lin", LinearRegression()),
            ]),
        ),
        ("Random Forest", RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)),
        ("XGBoost", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)),
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("KNN", KNeighborsRegressor(n_neighbors=5, weights="distance")),
        ("Neural Network", TransformedTargetRegressor(
            regressor=Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(64, 32), activation="relu",
                    alpha=0.05, learning_rate_init=0.001,
                    max_iter=2000, random_state=42)),
            ]),
            transformer=StandardScaler(),
        )),
        ("CatBoost", CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=6,
            random_seed=42, verbose=0)),
    ]


def metric_label(metric_name: str, target_name: str) -> str:
    unit = TARGET_UNITS.get(target_name)
    return f"{metric_name} [{unit}]" if unit else metric_name


def formula_markdown(model, target_name: str, model_name: str, feature_names):
    if model_name == "Polynomial Regression deg2":
        poly = model.named_steps["poly"]
        estimator = model.named_steps["lin"]
        names = poly.get_feature_names_out(feature_names)
    else:
        estimator = model
        names = feature_names

    terms = " ".join(
        f"{coef:+.6f}·{feature}"
        for coef, feature in zip(estimator.coef_, names)
    )
    return (
        f"**{model_name} formula for {target_name}**\n\n"
        f"```\n{target_name} = {estimator.intercept_:.6f} {terms}\n```"
    )


THUMBNAIL_WIDTH = 560


def render_zoomable_figure(figure, label, key_suffix, preview_size=(9, 5), zoom_size=(16, 9)):
    def to_png(size, dpi):
        clone = deepcopy(figure)
        clone.set_size_inches(*size)
        buffer = io.BytesIO()
        clone.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(clone)
        return buffer.getvalue()

    # Fixed-width image: st.pyplot would stretch any figure to the container width.
    st.image(to_png(preview_size, 100), width=THUMBNAIL_WIDTH)

    @st.dialog(label, width="large")
    def show_zoom():
        st.image(to_png(zoom_size, 170), width="stretch")

    if st.button(f"⛶ Full screen — {label}", key=f"zoom_open_{key_suffix}"):
        show_zoom()


def preferred_targets(columns):
    for pair in (("thcond", "spheat"), ("density", "visc")):
        matches = [column for column in pair if column in columns]
        if matches:
            return matches
    return [columns[-1]] if columns else []


@st.cache_resource(show_spinner=False)
def train_all(file_bytes, file_name, sheet_name, feature_names, target_names):
    raw = read_tabular_data(file_bytes, file_name, sheet_name)
    numeric = numeric_data_frame(raw)
    selected = list(feature_names) + list(target_names)
    model_data = numeric[selected].dropna()
    train_frame, test_frame = train_test_split(
        model_data, test_size=0.2, random_state=42
    )
    x_train = train_frame[list(feature_names)]
    x_test = test_frame[list(feature_names)]

    results = {}
    for target_name in target_names:
        y_train = train_frame[target_name]
        y_test = test_frame[target_name]
        models = make_models()
        for _, model in models:
            model.fit(x_train, y_train)
        results[target_name] = {
            "models": models,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }
    return results


for key, value in [("training_config", None), ("trained", False)]:
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("Upload and Train")
    uploaded_file = st.file_uploader(
        "CSV or Excel file", type=["csv", "xlsx", "xlsm"]
    )
    if uploaded_file is None:
        st.info("Upload a CSV, XLSX, or XLSM file to choose variables.")
        st.stop()

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    sheet_name = None

    try:
        if file_name.lower().endswith((".xlsx", ".xlsm")):
            sheet_name = st.selectbox(
                "Worksheet", excel_sheet_names(file_bytes)
            )
        raw_frame = read_tabular_data(file_bytes, file_name, sheet_name)
    except (
        InvalidFileException,
        ImportError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        zipfile.BadZipFile,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        st.error(f"Could not read the uploaded file: {error}")
        st.stop()

    numeric_frame = numeric_data_frame(raw_frame)
    numeric_columns = list(numeric_frame.columns)
    st.caption("Preview (first 10 rows)")
    st.dataframe(raw_frame.head(10), hide_index=True, use_container_width=True)

    if len(numeric_columns) < 2:
        st.error("The selected sheet needs at least two numeric columns.")
        st.stop()

    default_targets = preferred_targets(numeric_columns)
    default_features = [
        column for column in numeric_columns if column not in default_targets
    ]
    st.caption(
        "Each row should be one observation and the header row should contain "
        "variable names. Non-numeric columns are excluded."
    )
    feature_selection = st.multiselect(
        "Input variables",
        options=numeric_columns,
        default=default_features,
        help="Choose one or more values the model will use to make predictions.",
    )
    target_selection = st.multiselect(
        "Target variables to predict",
        options=numeric_columns,
        default=default_targets,
        help="The app trains a separate set of nine models for each target.",
    )

    overlap = sorted(set(feature_selection) & set(target_selection))
    selected_columns = feature_selection + target_selection
    if overlap:
        st.error(
            "A column cannot be both an input and a target: "
            + ", ".join(overlap)
        )
    elif not feature_selection or not target_selection:
        st.warning("Select at least one input and one target variable.")
    else:
        usable_rows = numeric_frame[selected_columns].dropna().shape[0]
        st.caption(
            f"{usable_rows} rows have numeric values in all selected columns."
        )
        if usable_rows < 10:
            st.warning("At least 10 complete rows are required to train models.")

    train_button = st.button(
        "Train Models",
        type="primary",
        disabled=(
            not feature_selection
            or not target_selection
            or bool(overlap)
            or numeric_frame[selected_columns].dropna().shape[0] < 10
        ),
        use_container_width=True,
    )

    config = (
        hashlib.sha256(file_bytes).hexdigest(),
        file_name,
        sheet_name,
        tuple(feature_selection),
        tuple(target_selection),
    )
    if train_button:
        st.session_state.training_config = config
        st.session_state.trained = True

if (
    not st.session_state.trained
    or st.session_state.training_config != config
):
    st.info(
        "Choose input and target variables in the sidebar, then click "
        "**Train Models**."
    )
    st.stop()

feature_cols = list(feature_selection)
target_names = list(target_selection)
training_data = numeric_frame[feature_cols + target_names].dropna()

with st.spinner("Training models for the selected targets…"):
    target_results = train_all(
        file_bytes,
        file_name,
        sheet_name,
        tuple(feature_cols),
        tuple(target_names),
    )

st.success(
    f"Models trained · Targets: **{', '.join(target_names)}** · "
    f"Inputs: **{', '.join(feature_cols)}**"
)

tab_names = ["Feature Analysis", "Predict"] + MODEL_NAMES
tabs = st.tabs(tab_names)

with tabs[0]:
    st.header("Feature Analysis")
    st.subheader("1. Correlation (inputs → targets)")
    st.caption(
        "Pearson measures linear association; Spearman measures "
        "monotonic association."
    )
    figure = correlation_figure(training_data, feature_cols, target_names)
    correlation_height = max(4.5, 0.6 * len(target_names) + 2)
    render_zoomable_figure(
        figure,
        "All targets correlation",
        "corr_all_targets",
        preview_size=(9, correlation_height),
        zoom_size=(15, correlation_height * 1.67),
    )
    plt.close(figure)

    st.subheader("2. Feature Importances")
    for target_name in target_names:
        result = target_results[target_name]
        models = result["models"]
        importance_rows = []
        for model_name, model in models:
            if model_name in TREE_MODELS:
                importance_rows.append(
                    pd.Series(
                        model.feature_importances_,
                        index=feature_cols,
                        name=model_name,
                    )
                )
        importance_frame = pd.DataFrame(importance_rows)
        figure, axis = plt.subplots(
            figsize=(max(8, 2.5 * len(feature_cols)), 4)
        )
        importance_frame.T.plot(kind="bar", ax=axis)
        axis.set_ylabel("Feature Importance")
        axis.set_title(f"Tree-model feature importances — {target_name}")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        render_zoomable_figure(
            figure,
            f"{target_name} importances",
            f"importance_{target_name}",
            preview_size=(9, 4.5),
            zoom_size=(15, 7.5),
        )
        plt.close(figure)

    st.subheader("3. SHAP Values (best tree model)")
    for target_name in target_names:
        result = target_results[target_name]
        models = result["models"]
        x_test = result["x_test"]
        y_test = result["y_test"]
        best_name, best_model, best_r2 = None, None, -math.inf
        for model_name, model in models:
            if model_name not in TREE_MODELS:
                continue
            score = compute_regression_metrics(
                y_test, model.predict(x_test)
            )["r2"]
            if not np.isfinite(score):
                score = -math.inf
            if best_model is None or score > best_r2:
                best_name, best_model, best_r2 = model_name, model, score
        st.caption(f"{target_name} — best tree model by test R²: {best_name}")
        shap_values = shap.TreeExplainer(best_model).shap_values(x_test)
        for plot_type in ("bar", "dot"):
            shap.summary_plot(
                shap_values,
                x_test,
                feature_names=feature_cols,
                plot_type=plot_type,
                show=False,
            )
            figure = plt.gcf()
            render_zoomable_figure(
                figure,
                f"{target_name} SHAP {plot_type}",
                f"shap_{target_name}_{plot_type}",
                preview_size=(9, 4.8),
                zoom_size=(15, 8),
            )
            plt.close("all")

with tabs[1]:
    st.subheader("Predict target values for a new observation")
    prediction_columns = st.columns(min(len(feature_cols), 4))
    prediction_values = {}
    for index, feature_name in enumerate(feature_cols):
        with prediction_columns[index % len(prediction_columns)]:
            prediction_values[feature_name] = st.number_input(
                feature_name,
                value=float(training_data[feature_name].median()),
                format="%.6g",
                key=f"prediction_{feature_name}",
            )
    if st.button("Predict", type="primary"):
        new_data = pd.DataFrame([prediction_values], columns=feature_cols)
        prediction_rows = []
        for model_index, model_name in enumerate(MODEL_NAMES):
            row = {"Model": model_name}
            for target_name in target_names:
                model = target_results[target_name]["models"][model_index][1]
                row[target_name] = float(model.predict(new_data)[0])
            prediction_rows.append(row)
        st.dataframe(
            pd.DataFrame(prediction_rows).round(6),
            hide_index=True,
            use_container_width=True,
        )

for tab, model_name in zip(tabs[2:], MODEL_NAMES):
    with tab:
        st.subheader("Metrics — held-out test set (80/20 split)")
        metric_rows = []
        for target_name in target_names:
            result = target_results[target_name]
            model = dict(result["models"])[model_name]
            y_test = result["y_test"]
            y_test_pred = model.predict(result["x_test"])
            y_full = pd.concat([result["y_train"], y_test])
            x_full = pd.concat([result["x_train"], result["x_test"]])
            full_metrics = compute_regression_metrics(
                y_full, model.predict(x_full)
            )
            test_metrics = compute_regression_metrics(y_test, y_test_pred)
            metric_rows.append({
                "Target": target_name,
                "R² (test)": f"{test_metrics['r2']:.4f}",
                "R² (all)": f"{full_metrics['r2']:.4f}",
                metric_label("MAE", target_name): f"{test_metrics['mae']:.6f}",
                metric_label("RMSE", target_name): f"{test_metrics['rmse']:.6f}",
                "MAPE (%) test": f"{test_metrics['mape']:.6f}",
                "MAPE (%) all": f"{full_metrics['mape']:.6f}",
                "AAD (%) test": f"{test_metrics['aad']:.6f}",
                "AAD (%) all": f"{full_metrics['aad']:.6f}",
                "MAE% of Mean test": f"{test_metrics['mae_pct_of_mean']:.6f}",
                "MAE% of Mean all": f"{full_metrics['mae_pct_of_mean']:.6f}",
            })
        st.dataframe(
            pd.DataFrame(metric_rows), hide_index=True, use_container_width=True
        )

        if model_name in {
            "Linear Regression", "Polynomial Regression deg2", "Ridge Regression"
        }:
            st.subheader("Fitted formulas")
            for target_name in target_names:
                model = dict(target_results[target_name]["models"])[model_name]
                st.markdown(
                    formula_markdown(
                        model, target_name, model_name, feature_cols
                    )
                )

        st.subheader("Parity Plots")
        for target_name in target_names:
            result = target_results[target_name]
            model = dict(result["models"])[model_name]
            figure = parity_figure(
                result["x_train"],
                result["x_test"],
                result["y_train"],
                result["y_test"],
                model,
                target_name,
            )
            render_zoomable_figure(
                figure,
                f"{model_name} {target_name}",
                f"parity_{model_name}_{target_name}",
                preview_size=(9, 4.8),
                zoom_size=(15, 8),
            )
            plt.close(figure)
