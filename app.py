import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path("HRDataset_v14.csv")
EMISSIONS_PATH = Path("emissions.csv")
TARGET = "Termd"


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("HRDataset_v14.csv not found in the current folder.")
    return pd.read_csv(DATA_PATH)


def pseudonymize(value: object) -> str:
    return "EMP_" + hashlib.sha256(str(value).encode()).hexdigest()[:8].upper()


def prepare_data(df_raw: pd.DataFrame):
    df = df_raw.copy()

    pii_columns = [c for c in ["Employee_Name", "EmpID", "ManagerName", "ManagerID"] if c in df.columns]
    quasi_identifiers = [
        c
        for c in ["DOB", "Zip", "DateofHire", "DateofTermination", "LastPerformanceReview_Date"]
        if c in df.columns
    ]
    leakage_columns = [c for c in ["TermReason", "EmploymentStatus", "EmpStatusID", "DateofTermination"] if c in df.columns]
    redundant_codes = [c for c in ["DeptID", "PositionID", "PerfScoreID", "MaritalStatusID", "GenderID"] if c in df.columns]
    sensitive_attrs = [c for c in ["Sex", "RaceDesc", "HispanicLatino"] if c in df.columns]

    if "Employee_Name" in df.columns:
        df["Employee_ID_Anon"] = df["Employee_Name"].apply(pseudonymize)
        df.drop(columns=["Employee_Name"], inplace=True)

    if "ManagerName" in df.columns:
        df["Manager_Anon"] = df["ManagerName"].apply(pseudonymize)
        df.drop(columns=["ManagerName"], inplace=True)

    if "DOB" in df.columns:
        df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
        df["Age"] = 2024 - df["DOB"].dt.year
        df["AgeBracket"] = pd.cut(
            df["Age"],
            bins=[0, 30, 40, 50, 60, 100],
            labels=["<30", "30-39", "40-49", "50-59", "60+"],
        )
        df.drop(columns=["DOB", "Age"], inplace=True)

    cols_to_drop = [c for c in pii_columns + quasi_identifiers + redundant_codes if c in df.columns and c not in ["DOB"]]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    sensitive_for_audit = [c for c in ["Sex", "RaceDesc"] if c in df.columns]
    exclude_from_features = set(
        [TARGET]
        + sensitive_attrs
        + leakage_columns
        + ["Employee_ID_Anon", "Manager_Anon", "AgeBracket", "MarriedID", "FromDiversityJobFairID"]
    )

    if "DateofHire" in df.columns:
        df["DateofHire"] = pd.to_datetime(df["DateofHire"], errors="coerce")
        snapshot_date = pd.Timestamp("2019-01-01")
        df["TenureYears"] = ((snapshot_date - df["DateofHire"]).dt.days / 365.25).clip(lower=0)
        df.drop(columns=["DateofHire"], inplace=True)

    feature_columns = [c for c in df.columns if c not in exclude_from_features]
    X_full = df[feature_columns].copy()
    y = df[TARGET].copy()
    sensitive_df = df[sensitive_for_audit].copy() if sensitive_for_audit else pd.DataFrame(index=df.index)

    numeric_features = X_full.select_dtypes(include="number").columns.tolist()
    categorical_features = X_full.select_dtypes(exclude="number").columns.tolist()

    return {
        "df": df,
        "X_full": X_full,
        "y": y,
        "sensitive_df": sensitive_df,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "sensitive_for_audit": sensitive_for_audit,
    }


def build_preprocessor(numeric_features, categorical_features):
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


@st.cache_resource
def train_artifacts():
    df_raw = load_raw_data()
    prepared = prepare_data(df_raw)

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        prepared["X_full"],
        prepared["y"],
        prepared["sensitive_df"],
        test_size=0.2,
        stratify=prepared["y"],
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(prepared["numeric_features"], prepared["categorical_features"])

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fitted_prep = model.named_steps["preprocessor"]
    feature_names = np.asarray(fitted_prep.get_feature_names_out())
    X_test_transformed = fitted_prep.transform(X_test)

    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    shap_values_raw = explainer.shap_values(X_test_transformed)
    if isinstance(shap_values_raw, list):
        shap_vals = np.asarray(shap_values_raw[1 if len(shap_values_raw) > 1 else 0])
    else:
        arr = np.asarray(shap_values_raw)
        if arr.ndim == 3:
            if arr.shape[1] == X_test_transformed.shape[1]:
                shap_vals = arr[:, :, 1 if arr.shape[2] > 1 else 0]
            else:
                shap_vals = arr[:, 1 if arr.shape[1] > 1 else 0, :]
        else:
            shap_vals = arr

    if shap_vals.shape[1] != feature_names.shape[0]:
        raise ValueError("SHAP feature mismatch between transformed matrix and feature names.")

    return {
        "df_raw": df_raw,
        "prepared": prepared,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_test": sensitive_test,
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_names": feature_names,
        "shap_vals": shap_vals,
        "explainer": explainer,
    }


def compute_group_metrics(y_true_arr, y_pred_arr, group_series, attribute):
    local = pd.DataFrame({attribute: group_series.values, "y_true": y_true_arr, "y_pred": y_pred_arr}).dropna()
    local[attribute] = local[attribute].astype(str)
    rows = []
    for group_name, gdf in local.groupby(attribute):
        sel_rate = gdf["y_pred"].mean()
        rows.append(
            {
                "group": group_name,
                "support": len(gdf),
                "selection_rate": sel_rate,
                "observed_attrition_rate": gdf["y_true"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def render_shap_contributions(row_shap, feature_names, top_n=12):
    order = np.argsort(np.abs(row_shap))[::-1][:top_n]
    vals = row_shap[order]
    names = feature_names[order]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(vals)), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names[::-1])
    ax.set_title("Top SHAP Contributions (Local)")
    ax.set_xlabel("Contribution to attrition risk")
    ax.axvline(0, color="black", linewidth=1)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="HR Attrition Trusted AI Dashboard", layout="wide")
    st.title("HR Attrition Trusted AI Dashboard")
    st.caption("End-to-end UI for prediction, explainability, fairness, and frugal AI monitoring.")

    artifacts = train_artifacts()

    y_test = artifacts["y_test"]
    y_pred = artifacts["y_pred"]
    y_proba = artifacts["y_proba"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Test ROC-AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
    col2.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col3.metric("Rows", f"{len(artifacts['df_raw'])}")

    tab_overview, tab_predict, tab_xai, tab_fairness, tab_frugal = st.tabs(
        ["Overview", "Predict", "Explainability", "Fairness", "Frugal AI"]
    )

    with tab_overview:
        st.subheader("Dataset Overview")
        st.dataframe(artifacts["df_raw"].head(20), use_container_width=True)
        st.write("Target distribution")
        st.bar_chart(artifacts["df_raw"][TARGET].value_counts().rename(index={0: "Active", 1: "Terminated"}))

    with tab_predict:
        st.subheader("Single Employee Risk Prediction")
        X_full = artifacts["prepared"]["X_full"]
        sample_idx = st.selectbox("Select baseline employee row", X_full.index.tolist(), index=0)
        base_row = X_full.loc[sample_idx].copy()

        edited = {}
        st.markdown("Adjust inputs")
        for col in X_full.columns:
            val = base_row[col]
            if pd.api.types.is_numeric_dtype(X_full[col]):
                edited[col] = st.number_input(col, value=float(val) if pd.notna(val) else 0.0)
            else:
                options = sorted([str(x) for x in X_full[col].dropna().unique().tolist()])
                default = str(val) if pd.notna(val) else (options[0] if options else "")
                edited[col] = st.selectbox(col, options=options, index=options.index(default) if default in options else 0)

        input_df = pd.DataFrame([edited])
        risk = float(artifacts["model"].predict_proba(input_df)[:, 1][0])
        label = "High" if risk >= 0.6 else "Medium" if risk >= 0.3 else "Low"
        st.metric("Predicted Attrition Risk", f"{risk:.1%}", label)
        st.progress(min(max(risk, 0.0), 1.0))

    with tab_xai:
        st.subheader("Explainability (SHAP)")
        row_idx = st.slider("Select test employee index", 0, len(artifacts["X_test"]) - 1, 0)
        row_shap = np.asarray(artifacts["shap_vals"][row_idx]).ravel()
        fig = render_shap_contributions(row_shap, artifacts["feature_names"], top_n=12)
        st.pyplot(fig)
        st.write("Positive bars increase predicted attrition risk; negative bars decrease it.")

    with tab_fairness:
        st.subheader("Fairness Snapshot")
        sensitive_test = artifacts["sensitive_test"]
        shown = False
        for attr in ["Sex", "RaceDesc"]:
            if attr in sensitive_test.columns:
                shown = True
                st.markdown(f"**{attr}**")
                metrics = compute_group_metrics(y_test.values, y_pred, sensitive_test[attr], attr)
                st.dataframe(metrics, use_container_width=True)
        if not shown:
            st.info("No sensitive columns available for fairness display.")

    with tab_frugal:
        st.subheader("Frugal AI Monitor")
        if EMISSIONS_PATH.exists():
            emissions = pd.read_csv(EMISSIONS_PATH)
            st.dataframe(emissions.tail(10), use_container_width=True)
            if "emissions" in emissions.columns and len(emissions) > 0:
                last = emissions["emissions"].dropna().iloc[-1]
                st.metric("Latest run emissions", f"{last * 1000:.6f} g CO2eq")
        else:
            st.info("No emissions.csv file found yet. Run your notebook frugal cell or tracking script first.")


if __name__ == "__main__":
    main()
