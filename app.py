"""
Medical Insurance Cost Prediction - Streamlit App
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="💊", layout="wide")


# ----------------------------- Cached loaders -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("outputs/best_model.joblib")
    with open("outputs/best_model_info.json") as f:
        info = json.load(f)
    return model, info


@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")


@st.cache_data
def load_model_comparison():
    return pd.read_csv("outputs/model_comparison.csv")


model, info = load_model()
df = load_data()
comparison_df = load_model_comparison()

st.title("💊 Medical Insurance Cost Predictor")
st.caption(
    f"Powered by **{info['best_model']}**  |  Test R² = {info['metrics']['R2']}  |  "
    f"RMSE = ₹{info['metrics']['RMSE']:,.0f}"
)

tab_predict, tab_dashboard = st.tabs(["🔮 Predict", "📊 Dashboard"])

# ============================================================
# TAB 1: PREDICTION
# ============================================================
with tab_predict:
    st.write("Enter the details below to estimate annual medical insurance charges.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 64, 30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=55.0, value=25.0, step=0.1)
        children = st.slider("Number of Children", 0, 5, 0)

    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    if st.button("Predict Cost", type="primary"):
        input_df = pd.DataFrame([{
            "age": age, "sex": sex, "bmi": bmi,
            "children": children, "smoker": smoker, "region": region
        }])
        prediction = model.predict(input_df)[0]
        st.success(f"### Estimated Annual Insurance Charges: **₹{prediction:,.2f}**")

        if smoker == "yes":
            st.warning("Smoking status significantly increases predicted charges.")
        if bmi >= 30:
            st.info("BMI is in the obese range, which tends to raise predicted charges.")

    st.divider()
    st.caption("Medical Insurance Cost Prediction")

# ============================================================
# TAB 2: DASHBOARD
# ============================================================
with tab_dashboard:
    st.subheader("Dataset & Model Overview")

    # ---- Filters (in an expander so dashboard stays clean) ----
    with st.expander("🔍 Filter data", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            region_filter = st.multiselect(
                "Region", options=sorted(df["region"].unique()),
                default=sorted(df["region"].unique())
            )
        with f2:
            smoker_filter = st.multiselect(
                "Smoker", options=sorted(df["smoker"].unique()),
                default=sorted(df["smoker"].unique())
            )
        with f3:
            age_range = st.slider(
                "Age range", int(df["age"].min()), int(df["age"].max()),
                (int(df["age"].min()), int(df["age"].max()))
            )

    fdf = df[
        df["region"].isin(region_filter)
        & df["smoker"].isin(smoker_filter)
        & df["age"].between(age_range[0], age_range[1])
    ]

    # ---- KPI row ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records", f"{len(fdf):,}")
    k2.metric("Avg. Charges", f"₹{fdf['charges'].mean():,.0f}" if len(fdf) else "₹0")
    k3.metric("Median Charges", f"₹{fdf['charges'].median():,.0f}" if len(fdf) else "₹0")
    smoker_pct = (fdf["smoker"] == "yes").mean() * 100 if len(fdf) else 0
    k4.metric("% Smokers", f"{smoker_pct:.1f}%")

    st.divider()

    # ---- Row: Model comparison ----
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model Comparison (R²)**")
        comp_sorted = comparison_df.sort_values("R2", ascending=True)
        fig = px.bar(
            comp_sorted, x="R2", y="Model", orientation="h",
            text=comp_sorted["R2"].map(lambda x: f"{x:.3f}"),
            color="R2", color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Model Comparison (RMSE, lower is better)**")
        comp_sorted_rmse = comparison_df.sort_values("RMSE", ascending=False)
        fig2 = px.bar(
            comp_sorted_rmse, x="RMSE", y="Model", orientation="h",
            text=comp_sorted_rmse["RMSE"].map(lambda x: f"₹{x:,.0f}"),
            color="RMSE", color_continuous_scale="Reds",
        )
        fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ---- Row: Charges distribution + Smoker impact ----
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Charges Distribution**")
        fig3 = px.histogram(fdf, x="charges", nbins=40, color="smoker",
                             color_discrete_map={"yes": "#EF553B", "no": "#636EFA"})
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("**Charges by Smoking Status**")
        fig4 = px.box(fdf, x="smoker", y="charges", color="smoker",
                      color_discrete_map={"yes": "#EF553B", "no": "#636EFA"})
        fig4.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ---- Row: Age vs Charges (BMI colored) + Region breakdown ----
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**Age vs Charges (colored by BMI)**")
        fig5 = px.scatter(
            fdf, x="age", y="charges", color="bmi",
            color_continuous_scale="Viridis",
            symbol="smoker",
            hover_data=["sex", "children", "region"],
        )
        fig5.update_layout(height=380)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        st.markdown("**Average Charges by Region**")
        region_avg = fdf.groupby("region", as_index=False)["charges"].mean().sort_values("charges")
        fig6 = px.bar(region_avg, x="region", y="charges", color="charges",
                      color_continuous_scale="Blues", text_auto=".2s")
        fig6.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)

    st.divider()

    # ---- Feature importance (from saved plot) ----
    st.markdown("**Feature Importance (Best Model)**")
    st.image("outputs/feature_importance.png", use_container_width=True)

    # ---- Key insights ----
    st.markdown("**Key Insights**")
    st.markdown(
        """
- **Smoking status** is the single biggest driver of insurance cost.
- **Age** and **BMI** are the next most important numeric predictors.
- **Sex, children, and region** have comparatively minor effects.
- **Gradient Boosting** is the best-performing model, explaining ~88% of the variance in charges.
        """
    )

    with st.expander("📄 View raw filtered data"):
        st.dataframe(fdf, use_container_width=True)
