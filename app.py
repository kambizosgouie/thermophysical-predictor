# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import shap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Thermophysical Predictor", layout="wide")
st.title("Thermophysical Property Predictor")

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [("trained", False), ("csv_bytes", None), ("t1", None), ("t2", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload & Train")
    uploaded_file = st.file_uploader("CSV file", type="csv")
    train_btn = st.button(
        "🔧 Train Models",
        type="primary",
        disabled=(uploaded_file is None),
        use_container_width=True,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_metrics(ytrue, ypred):
    return (
        r2_score(ytrue, ypred),
        mean_absolute_error(ytrue, ypred),
        np.sqrt(mean_squared_error(ytrue, ypred)),
    )


def make_models():
    return [
        ("Linear Regression", LinearRegression()),
        ("Polynomial Regression deg2", Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin",  LinearRegression()),
        ])),
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


def linear_formula_md(model, target_name, label):
    c = model.coef_
    b = model.intercept_
    return (
        f"**{label} formula for {target_name}**\n\n"
        f"```\n"
        f"{target_name} = {b:.6f}"
        f" + {c[0]:.6f}·temp"
        f" + {c[1]:.6f}·loading"
        f" + {c[2]:.6f}·conc\n"
        f"```"
    )


def poly_formula_md(pipe, target_name, degree):
    poly_step = pipe.named_steps["poly"]
    lin_step  = pipe.named_steps["lin"]
    feat_names = poly_step.get_feature_names_out(["temp", "loading", "conc"])
    coefs = lin_step.coef_
    inter = lin_step.intercept_

    expr = f"{target_name} = {inter:.6f}"
    for nm, c in zip(feat_names[1:], coefs[1:]):
        expr += f" + {c:.6f}·{nm}"
    return f"**Polynomial deg{degree} formula for {target_name}**\n\n```\n{expr}\n```"


def parity_fig(X_train, X_test, y_train, y_test, model, target_name):
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (X_used, y_used, subtitle) in zip(axes, [
        (X_test,  y_test,  "Test data"),
        (X_full,  y_full,  "Full dataset (train + test)"),
    ]):
        y_pred = model.predict(X_used)
        ax.scatter(y_used, y_pred, alpha=0.7, edgecolor="k", s=30)
        mn = min(y_used.min(), y_pred.min())
        mx = max(y_used.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=2)
        ax.set_xlabel(f"True {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{target_name} — Parity Plots", fontsize=13)
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

# Handle Train button press
if train_btn and uploaded_file is not None:
    import io as _io
    uploaded_file.seek(0)
    _bytes = uploaded_file.read()
    _df = pd.read_csv(_io.BytesIO(_bytes))
    if {"thcond", "spheat"}.issubset(_df.columns):
        _t1, _t2 = "thcond", "spheat"
    elif {"density", "visc"}.issubset(_df.columns):
        _t1, _t2 = "density", "visc"
    else:
        st.error("CSV must contain either **(thcond, spheat)** or **(density, visc)** columns.")
        st.stop()
    st.session_state.csv_bytes = _bytes
    st.session_state.t1 = _t1
    st.session_state.t2 = _t2
    st.session_state.trained = True

if not st.session_state.trained:
    st.info(
        "Upload a CSV file and click **🔧 Train Models** to begin."
    )
    st.stop()

target1_name = st.session_state.t1
target2_name = st.session_state.t2

# Train models (cached — only re-runs when CSV changes)
@st.cache_resource(show_spinner=False)
def train_all(csv_bytes, t1, t2, version="v3"):
    import io
    _df = pd.read_csv(io.BytesIO(csv_bytes))
    _X  = _df[["temp", "loading", "conc"]]
    _y1 = _df[t1]
    _y2 = _df[t2]
    _Xtr1, _Xte1, _y1tr, _y1te = train_test_split(_X, _y1, test_size=0.2, random_state=42)
    _Xtr2, _Xte2, _y2tr, _y2te = train_test_split(_X, _y2, test_size=0.2, random_state=42)

    _m1 = make_models()
    _m2 = make_models()
    for _, m in _m1:
        m.fit(_Xtr1, _y1tr)
    for _, m in _m2:
        m.fit(_Xtr2, _y2tr)
    return _m1, _m2, _Xtr1, _Xte1, _y1tr, _y1te, _Xtr2, _Xte2, _y2tr, _y2te


with st.spinner("Loading models…"):
    models1, models2, Xtrain1, Xtest1, y1_train, y1_test, \
        Xtrain2, Xtest2, y2_train, y2_test = train_all(
            st.session_state.csv_bytes, target1_name, target2_name, version="v3"
        )

st.success(f"Models trained · Targets: **{target1_name}**, **{target2_name}**")

# ── Tabs ──────────────────────────────────────────────────────────────────────
all_tab_names = ["📊 Feature Analysis", "🔮 Predict"] + [name for name, _ in models1]
tabs = st.tabs(all_tab_names)
feature_tab = tabs[0]
predict_tab = tabs[1]
model_tabs  = tabs[2:]

# ── Feature Analysis tab ──────────────────────────────────────────────────────
with feature_tab:
    X_full = pd.concat([Xtrain1, Xtest1])
    y1_full = pd.concat([y1_train, y1_test])
    y2_full = pd.concat([y2_train, y2_test])
    full_df = X_full.copy()
    full_df[target1_name] = y1_full.values
    full_df[target2_name] = y2_full.values

    FEATURES = ["temp", "loading", "conc"]

    # ── 1. Correlation ────────────────────────────────────────────────────────
    st.subheader("1. Correlation (inputs → outputs)")
    st.caption(
        "Pearson: linear relationship strength & direction.  "
        "Spearman: monotonic (non-linear) relationship."
    )
    pearson  = full_df.corr(method="pearson" )[FEATURES].loc[[target1_name, target2_name]]
    spearman = full_df.corr(method="spearman")[FEATURES].loc[[target1_name, target2_name]]

    fig_corr, axes_corr = plt.subplots(1, 2, figsize=(10, 2.5))
    for ax, data, title in zip(
        axes_corr,
        [pearson, spearman],
        ["Pearson", "Spearman"],
    ):
        im = ax.imshow(data.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES)
        ax.set_yticks(range(2)); ax.set_yticklabels([target1_name, target2_name])
        for r in range(2):
            for c in range(len(FEATURES)):
                ax.text(c, r, f"{data.values[r, c]:.2f}", ha="center", va="center",
                        fontsize=11, color="white" if abs(data.values[r, c]) > 0.5 else "black")
        ax.set_title(f"{title} Correlation")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig_corr.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    # ── 2. Feature Importances (tree models) ─────────────────────────────────
    st.subheader("2. Feature Importances (tree-based models)")
    st.caption("Magnitude of each input's contribution — averaged across RF, GBM, XGBoost, CatBoost.")

    tree_names = ["Random Forest", "Gradient Boosting", "XGBoost", "CatBoost"]

    def get_importances(models_list):
        rows = []
        for nm, mdl in models_list:
            if nm in tree_names:
                imp = mdl.feature_importances_
                rows.append(pd.Series(imp / imp.sum(), index=FEATURES, name=nm))
        return pd.DataFrame(rows)

    imp1 = get_importances(models1)
    imp2 = get_importances(models2)

    fig_imp, axes_imp = plt.subplots(1, 2, figsize=(12, 4))
    for ax, imp_df, tname in zip(axes_imp, [imp1, imp2], [target1_name, target2_name]):
        x = np.arange(len(FEATURES))
        width = 0.18
        for i, (idx, row) in enumerate(imp_df.iterrows()):
            ax.bar(x + i * width, row.values, width, label=idx)
        ax.set_xticks(x + width * (len(imp_df) - 1) / 2)
        ax.set_xticklabels(FEATURES)
        ax.set_ylabel("Normalised Importance")
        ax.set_title(f"Feature Importances → {tname}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig_imp.tight_layout()
    st.pyplot(fig_imp)
    plt.close(fig_imp)

    # ── 3. SHAP (best tree model by R²) ──────────────────────────────────────
    st.subheader("3. SHAP Values (best tree model)")
    st.caption(
        "SHAP shows each input's contribution to each prediction — "
        "**direction** (positive/negative) and **magnitude**."
    )

    def best_tree_model(models_list, X_te, y_te):
        best_name, best_model, best_r2 = None, None, -np.inf
        for nm, mdl in models_list:
            if nm in tree_names:
                r2 = r2_score(y_te, mdl.predict(X_te))
                if r2 > best_r2:
                    best_r2, best_name, best_model = r2, nm, mdl
        return best_name, best_model

    bname1, bmodel1 = best_tree_model(models1, Xtest1, y1_test)
    bname2, bmodel2 = best_tree_model(models2, Xtest2, y2_test)

    for bname, bmodel, X_tr, X_te, tname in [
        (bname1, bmodel1, Xtrain1, Xtest1, target1_name),
        (bname2, bmodel2, Xtrain2, Xtest2, target2_name),
    ]:
        st.markdown(f"**{tname}** — best tree model: *{bname}*")
        explainer = shap.TreeExplainer(bmodel)
        shap_vals = explainer.shap_values(X_te)
        fig_shap, ax_shap = plt.subplots(figsize=(7, 3))
        shap.summary_plot(
            shap_vals, X_te,
            feature_names=FEATURES,
            plot_type="bar",
            show=False,
        )
        st.pyplot(plt.gcf())
        plt.close("all")

        fig_bee, _ = plt.subplots(figsize=(7, 3))
        shap.summary_plot(shap_vals, X_te, feature_names=FEATURES, show=False)
        st.pyplot(plt.gcf())
        plt.close("all")

# ── Predict tab ─────────────────────────────────────────────────────────────────
with predict_tab:
    st.subheader("New Data Point Prediction")
    pc1, pc2, pc3 = st.columns(3)
    p_temp    = pc1.number_input("Temperature",   value=55.0, format="%.4f", key="p_temp")
    p_loading = pc2.number_input("Loading",       value=0.3,  format="%.4f", key="p_loading")
    p_conc    = pc3.number_input("Concentration", value=0.8,  format="%.4f", key="p_conc")
    do_predict = st.button("🔮 Predict", type="primary", use_container_width=False)

    if do_predict:
        newdata = pd.DataFrame({"temp": [p_temp], "loading": [p_loading], "conc": [p_conc]})
        rows = []
        for (name, m1), (_, m2) in zip(models1, models2):
            rows.append({
                "Model":       name,
                target1_name: round(float(m1.predict(newdata)[0]), 6),
                target2_name: round(float(m2.predict(newdata)[0]), 6),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ── Per-model tabs ────────────────────────────────────────────────────────────
for tab, (name, m1), (_, m2) in zip(model_tabs, models1, models2):
    with tab:

        # ── Metrics ──────────────────────────────────────────────────────────
        st.subheader("Metrics — test set (80/20 split)")
        r2_1, mae_1, rmse_1 = get_metrics(y1_test, m1.predict(Xtest1))
        r2_2, mae_2, rmse_2 = get_metrics(y2_test, m2.predict(Xtest2))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{target1_name}**")
            st.dataframe(
                pd.DataFrame({
                    "R²":   [f"{r2_1:.4f}"],
                    "MAE":  [f"{mae_1:.6f}"],
                    "RMSE": [f"{rmse_1:.6f}"],
                }),
                hide_index=True, use_container_width=True,
            )
        with col2:
            st.markdown(f"**{target2_name}**")
            st.dataframe(
                pd.DataFrame({
                    "R²":   [f"{r2_2:.4f}"],
                    "MAE":  [f"{mae_2:.6f}"],
                    "RMSE": [f"{rmse_2:.6f}"],
                }),
                hide_index=True, use_container_width=True,
            )

        # ── Formula (interpretable models only) ──────────────────────────────
        if name == "Linear Regression":
            st.subheader("Formula")
            st.markdown(linear_formula_md(m1, target1_name, "Linear"))
            st.markdown(linear_formula_md(m2, target2_name, "Linear"))
        elif name == "Polynomial Regression deg2":
            st.subheader("Formula")
            st.markdown(poly_formula_md(m1, target1_name, 2))
            st.markdown(poly_formula_md(m2, target2_name, 2))
        elif name == "Ridge Regression":
            st.subheader("Formula")
            st.markdown(linear_formula_md(m1, target1_name, "Ridge"))
            st.markdown(linear_formula_md(m2, target2_name, "Ridge"))

        # ── Parity plots ──────────────────────────────────────────────────────
        st.subheader("Parity Plots")
        fig1 = parity_fig(Xtrain1, Xtest1, y1_train, y1_test, m1, target1_name)
        fig2 = parity_fig(Xtrain2, Xtest2, y2_train, y2_test, m2, target2_name)
        st.pyplot(fig1)
        st.pyplot(fig2)
        plt.close("all")
