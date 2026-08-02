import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

# -----------------------------------------------------------------------------
# Streamlit Page Config & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Composite Shear Capacity AI Design Tool",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark modern theme & glowing metric cards
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
    }
    .stAppHeader {
        background: rgba(11, 15, 25, 0.8);
    }
    .metric-card-box {
        background: rgba(22, 31, 49, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
    }
    .metric-val-blue {
        font-size: 2.3rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 0.2rem 0;
    }
    .metric-val-green {
        font-size: 2.3rem;
        font-weight: 800;
        color: #34d399;
        margin: 0.2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helper: Create High-Contrast Dynamic Dark Mode SHAP Summary Plot
# -----------------------------------------------------------------------------
def make_high_contrast_shap_fig(shap_values, X_data, feature_names, title_str):
    plt.close('all')
    fig = plt.figure(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor('#0b0f19')
    
    shap.summary_plot(
        shap_values,
        X_data,
        feature_names=feature_names,
        show=False,
        plot_size=(9, 6)
    )
    
    ax = plt.gca()
    ax.set_facecolor('#0b0f19')
    
    plt.setp(ax.get_yticklabels(), color='#ffffff', fontsize=10, fontweight='bold')
    plt.setp(ax.get_xticklabels(), color='#ffffff', fontsize=10)
    ax.xaxis.label.set_color('#ffffff')
    ax.yaxis.label.set_color('#ffffff')
    ax.set_title(title_str, color='#38bdf8', fontsize=12, pad=12, fontweight='bold')
    
    for child in plt.gcf().get_children():
        if hasattr(child, 'yaxis') and child != ax:
            child.yaxis.label.set_color('#ffffff')
            plt.setp(child.get_yticklabels(), color='#ffffff', fontsize=9)
            
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# Load Models & Data (with Auto-Train if missing)
# -----------------------------------------------------------------------------
def check_and_train_if_needed():
    if not os.path.exists('models/best_model.joblib') or not os.path.exists('results/metrics_summary.json'):
        with st.spinner("Initial model setup in progress... Training AI models on dataset..."):
            from src.train_models import train_and_evaluate
            from src.explainability_and_plots import generate_plots_and_explainability
            train_and_evaluate()
            generate_plots_and_explainability()

check_and_train_if_needed()

@st.cache_resource
def load_model_assets():
    best_model = joblib.load('models/best_model.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
    rf_model = joblib.load('models/randomforest_model.joblib')
    explainer_rf = shap.TreeExplainer(rf_model)
    with open('models/metadata.json', 'r') as f:
        meta_info = json.load(f)
    with open('results/metrics_summary.json', 'r') as f:
        metrics_summary = json.load(f)
    return best_model, preprocessor, rf_model, explainer_rf, meta_info, metrics_summary

best_model, preprocessor, rf_model, explainer_rf, meta_info, metrics_summary = load_model_assets()

DEFAULT_PARAMS = {
    "Temperature (ºC)": 20.0,
    "Connector type": "Stud",
    "Diameter (mm)": 19.0,
    "Height (mm)": 100.0,
    "Concrete Grade (Mpa)": 30.0,
    "ASTM Fire Exposure Time (minute)": 0.0,
    "ISO Fire Exposure Time (minute)": 0.0,
    "Steel fy,θ": 380.0,
    "Steel fp,θ": 380.0,
    "Steel Ea,θ": 200000.0,
    "Steel ɛp,θ": 0.002,
    "Concrete Thermal Expansion (m-1C-1)": 0.000012,
    "Concrete Conductivity (W/mK)": 1.5,
    "Concrete Specific Heat (J/kgK)": 900.0,
    "Concrete Poisson's ratio, μ": 0.18,
    "Concrete Elastic Modulus (N/mm2)": 24000.0,
    "Concrete Compressive Strength (N/mm2)": 30.0,
    "Steel Conductivity (W/mK)": 45.0,
    "Steel Specific Heat (J/kgK)": 500.0,
    "Steel Poisson's ratio, μ": 0.3,
    "Steel Thermal Expansion (m-1C-1)": 0.000014,
    "Reduction factor (relative to fy) for effective yield strength ky,θ =fy,θ/fy": 1.0,
    "Reduction factor (relative to fy) for effective elastic modulos kE,θ =Ea,θ/Ea": 1.0
}

# -----------------------------------------------------------------------------
# Header & Top Controls
# -----------------------------------------------------------------------------
st.title("🏗️ Composite Deck Shear Capacity AI Design Tool")
st.caption("Predictive Thermo-Structural Design & Non-Linear Load-Slip Modeling at Elevated Temperatures")

# -----------------------------------------------------------------------------
# Sidebar: Input Parameters & One-Click Pipeline Trigger
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Connector & Thermal Inputs")

connector_type = st.sidebar.selectbox(
    "Connector Geometry Type",
    options=["Stud", "Bar", "Channel", "Tee", "Helical"],
    index=0
)

temperature_C = st.sidebar.slider(
    "Fire Temperature (ºC)",
    min_value=20, max_value=1200, value=20, step=10
)

diameter_mm = st.sidebar.slider(
    "Connector Diameter (mm)",
    min_value=10, max_value=65, value=19, step=1
)

height_mm = st.sidebar.slider(
    "Connector Height (mm)",
    min_value=40, max_value=160, value=100, step=5
)

concrete_grade_MPa = st.sidebar.slider(
    "Concrete Grade (MPa)",
    min_value=20, max_value=40, value=30, step=5
)

iso_exposure_min = st.sidebar.slider(
    "ISO Fire Exposure Time (min)",
    min_value=0, max_value=375, value=0, step=5
)

steel_fy_MPa = st.sidebar.slider(
    "Steel Yield Strength fy,θ (MPa)",
    min_value=100, max_value=500, value=380, step=10
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Automated AI Management")
if st.sidebar.button("Re-Run AI Training & SHAP Plots", use_container_width=True):
    with st.spinner("Re-training XGBoost, Random Forest, Decision Tree, & Neural Net..."):
        from src.train_models import train_and_evaluate
        from src.explainability_and_plots import generate_plots_and_explainability
        train_and_evaluate()
        generate_plots_and_explainability()
        st.cache_resource.clear()
        st.sidebar.success("Model pipeline updated & re-loaded successfully!")

# -----------------------------------------------------------------------------
# Prediction Function with Cohesive Parameter Mapping
# -----------------------------------------------------------------------------
input_dict = {}
for col in meta_info['feature_cols']:
    input_dict[col] = DEFAULT_PARAMS.get(col, 0.0)

T_val = float(temperature_C)
if iso_exposure_min > 0:
    T_iso = 20.0 + 345.0 * np.log10(8.0 * float(iso_exposure_min) + 1.0)
    T_val = max(T_val, T_iso)

if T_val <= 400:
    ky_factor = 1.0
    kE_factor = 1.0
elif T_val <= 500:
    ky_factor = 1.0 - 0.22 * (T_val - 400.0) / 100.0
    kE_factor = 0.70 - 0.10 * (T_val - 400.0) / 100.0
elif T_val <= 600:
    ky_factor = 0.78 - 0.31 * (T_val - 500.0) / 100.0
    kE_factor = 0.60 - 0.29 * (T_val - 500.0) / 100.0
elif T_val <= 700:
    ky_factor = 0.47 - 0.24 * (T_val - 600.0) / 100.0
    kE_factor = 0.31 - 0.18 * (T_val - 600.0) / 100.0
elif T_val <= 800:
    ky_factor = 0.23 - 0.12 * (T_val - 700.0) / 100.0
    kE_factor = 0.13 - 0.04 * (T_val - 700.0) / 100.0
else:
    ky_factor = max(0.02, 0.11 - 0.09 * (T_val - 800.0) / 400.0)
    kE_factor = max(0.02, 0.09 - 0.07 * (T_val - 800.0) / 400.0)

eff_fy = float(steel_fy_MPa) * ky_factor
eff_fp = eff_fy * 0.8
eff_Ea = 200000.0 * kE_factor

for col in meta_info['feature_cols']:
    if 'Connector' in col:
        input_dict[col] = connector_type
    elif 'Temperature' in col:
        input_dict[col] = T_val
    elif 'Diameter' in col:
        input_dict[col] = float(diameter_mm)
    elif 'Height' in col:
        input_dict[col] = float(height_mm)
    elif 'Concrete' in col and 'Grade' in col:
        input_dict[col] = float(concrete_grade_MPa)
    elif 'Concrete' in col and 'Compressive' in col:
        input_dict[col] = float(concrete_grade_MPa)
    elif 'ISO' in col:
        input_dict[col] = float(iso_exposure_min)
    elif 'ASTM' in col:
        input_dict[col] = float(iso_exposure_min) * 1.12
    elif 'fy' in col and 'ky' in col:
        input_dict[col] = ky_factor
    elif 'Ea' in col and 'kE' in col:
        input_dict[col] = kE_factor
    elif 'Steel' in col and 'fy' in col:
        input_dict[col] = eff_fy
    elif 'Steel' in col and 'fp' in col:
        input_dict[col] = eff_fp
    elif 'Steel' in col and 'Ea' in col:
        input_dict[col] = eff_Ea

input_df = pd.DataFrame([input_dict])
for col in input_df.select_dtypes(include=['object']).columns:
    input_df[col] = input_df[col].astype(str).str.strip()

X_trans = preprocessor.transform(input_df)
pred = best_model.predict(X_trans)

pred_capacity_kN = max(0.0, float(pred[0][0]))
pred_slip_mm = max(0.1, float(pred[0][1]))

# -----------------------------------------------------------------------------
# Dynamic Local Neighborhood Matrix for Live SHAP Summary Plots
# -----------------------------------------------------------------------------
dyn_rows = []
for temp_v in np.linspace(max(20.0, T_val - 250), min(1200.0, T_val + 250), 12):
    for dia_v in np.linspace(max(10.0, float(diameter_mm) - 15), min(65.0, float(diameter_mm) + 15), 4):
        r_temp = input_dict.copy()
        r_temp['Temperature (ºC)'] = float(temp_v)
        r_temp['Diameter (mm)'] = float(dia_v)
        r_temp['Concrete Grade (Mpa)'] = float(concrete_grade_MPa)
        dyn_rows.append(r_temp)

df_dyn = pd.DataFrame(dyn_rows)
for col in df_dyn.select_dtypes(include=['object']).columns:
    df_dyn[col] = df_dyn[col].astype(str).str.strip()

X_dyn_trans = preprocessor.transform(df_dyn)
shap_dyn = explainer_rf(X_dyn_trans)

ohe_names = preprocessor.named_transformers_['cat'].get_feature_names_out(meta_info['cat_cols']).tolist()
clean_feature_names = [c.replace('Connector type_', '').replace('\n', ' ') for c in (ohe_names + meta_info['num_cols'])]

# -----------------------------------------------------------------------------
# Physically Accurate Load-Slip Curve
# -----------------------------------------------------------------------------
s_max = pred_slip_mm * 1.5
slip_steps = np.linspace(0, s_max, 100)

load_steps = []
norm_denom = 1.0 - np.exp(-3.0)
for s in slip_steps:
    if s <= pred_slip_mm:
        val = pred_capacity_kN * ((1.0 - np.exp(-3.0 * s / pred_slip_mm)) / norm_denom)**0.65
    else:
        over_ratio = (s - pred_slip_mm) / pred_slip_mm
        val = pred_capacity_kN * max(0.15, 1.0 - 0.45 * (over_ratio**1.1))
    load_steps.append(float(val))

# -----------------------------------------------------------------------------
# App Layout Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Load-Slip Prediction",
    "🔍 Explainable AI (SHAP Plots)",
    "📊 Model Benchmarks",
    "📘 Executive Report & Guide"
])

# -----------------------------------------------------------------------------
# Tab 1: Prediction & Interactive Plotly Curve
# -----------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card-box">
                <div class="metric-title">Predicted Ultimate Shear Capacity</div>
                <div class="metric-val-blue">{pred_capacity_kN:.2f} kN</div>
                <div style="font-size:0.8rem; color:#9ca3af;">Residual Load Peak</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card-box">
                <div class="metric-title">Predicted Slip at Failure</div>
                <div class="metric-val-green">{pred_slip_mm:.2f} mm</div>
                <div style="font-size:0.8rem; color:#9ca3af;">Deformation at Peak Load</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("📈 Predicted Non-Linear Thermo-Structural Load-Slip Curve")

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=slip_steps,
        y=load_steps,
        mode='lines',
        name='Predicted Thermo-Structural Curve',
        line=dict(color='#38bdf8', width=3.5),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.08)'
    ))
    
    fig.add_trace(go.Scatter(
        x=[pred_slip_mm],
        y=[pred_capacity_kN],
        mode='markers+text',
        name=f'Peak Capacity ({pred_capacity_kN:.2f} kN, {pred_slip_mm:.2f} mm)',
        marker=dict(color='#ef4444', size=13, line=dict(color='#ffffff', width=2)),
        text=[f"  Peak ({pred_capacity_kN:.2f} kN, {pred_slip_mm:.2f} mm)"],
        textposition="top center"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="Slip s (mm)", gridcolor="rgba(255,255,255,0.1)", range=[0, s_max * 1.05]),
        yaxis=dict(title="Shear Force P (kN)", gridcolor="rgba(255,255,255,0.1)", range=[0, pred_capacity_kN * 1.15]),
        legend=dict(x=0.02, y=0.98)
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 2: Clean Dynamic SHAP Beeswarm Summary Plots ONLY
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🐝 Live Dynamic SHAP Beeswarm Summary Plots")
    st.caption(f"Every dot, color, and feature ranking re-renders LIVE in high contrast centered around your active inputs ({temperature_C}ºC, {connector_type}, {diameter_mm}mm diameter, {concrete_grade_MPa}MPa concrete).")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_shear = make_high_contrast_shap_fig(shap_dyn[:, :, 0], X_dyn_trans, clean_feature_names, "Live Shear Capacity SHAP Beeswarm")
        st.pyplot(fig_shear, clear_figure=True)

    with col_b:
        fig_slip = make_high_contrast_shap_fig(shap_dyn[:, :, 1], X_dyn_trans, clean_feature_names, "Live Slip SHAP Beeswarm")
        st.pyplot(fig_slip, clear_figure=True)

# -----------------------------------------------------------------------------
# Tab 3: Model Benchmarks
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("📊 Trained AI Models Scoreboard")
    
    benchmark_data = []
    for model_name, info in metrics_summary.items():
        benchmark_data.append({
            "Model": model_name + (" 🏆 (Best)" if model_name == meta_info['best_model_name'] else ""),
            "Shear Capacity R²": round(info['test']['capacity']['r2'], 4),
            "Shear RMSE (kN)": round(info['test']['capacity']['rmse'], 2),
            "Shear MAE (kN)": round(info['test']['capacity']['mae'], 2),
            "Slip R²": round(info['test']['slip']['r2'], 4),
            "Slip RMSE (mm)": round(info['test']['slip']['rmse'], 2),
            "Slip MAE (mm)": round(info['test']['slip']['mae'], 2),
        })

    benchmark_df = pd.DataFrame(benchmark_data)
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Tab 4: Executive Report & Guide
# -----------------------------------------------------------------------------
with tab4:
    if os.path.exists('PROJECT_GUIDE_AND_EXECUTIVE_REPORT.md'):
        with open('PROJECT_GUIDE_AND_EXECUTIVE_REPORT.md', 'r', encoding='utf-8') as f:
            report_md = f.read()
        st.markdown(report_md)
