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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Load-Slip Prediction",
    "🔍 Explainable AI (SHAP Plots)",
    "📊 Model Benchmarks",
    "🖼️ Paper-Aligned Graph Gallery",
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
# Tab 4: Paper-Aligned Graph Gallery (Fully Dynamic & Interactive)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🖼️ Dynamic Paper-Aligned Sensitivity & Prediction Gallery")
    st.caption(f"All parametric curves and active design markers below dynamically update live centered around your active inputs: Temperature = {T_val:.1f}ºC, {connector_type}, Diameter = {diameter_mm}mm, Height = {height_mm}mm, Concrete Grade = {concrete_grade_MPa}MPa, Steel Yield = {steel_fy_MPa}MPa.")

    # ---------------------------------------------------------
    # Dynamic Figure 4: Thermal Degradation Curves
    # ---------------------------------------------------------
    st.markdown("#### Figure 4: Dynamic Thermal Degradation Curves (20°C - 800°C)")
    st.caption("Shows residual shear capacity degradation for all 5 connector types under your active geometry & material inputs, with your active design point highlighted.")
    
    temp_sweep = np.linspace(20, 800, 40)
    fig4 = go.Figure()
    colors = ['#38bdf8', '#fbbf24', '#34d399', '#f87171', '#c084fc']
    
    all_connectors = ['Stud', 'Bar', 'Channel', 'Tee', 'Helical']
    for idx, conn_t in enumerate(all_connectors):
        synth_list = []
        for t_i in temp_sweep:
            r = input_dict.copy()
            r['Temperature (ºC)'] = float(t_i)
            for c_col in meta_info['feature_cols']:
                if 'Connector' in c_col:
                    r[c_col] = conn_t
            synth_list.append(r)
        synth_df = pd.DataFrame(synth_list)
        for col in synth_df.select_dtypes(include=['object']).columns:
            synth_df[col] = synth_df[col].astype(str).str.strip()
        preds_synth = best_model.predict(preprocessor.transform(synth_df))
        fig4.add_trace(go.Scatter(
            x=temp_sweep,
            y=[max(0.0, float(p[0])) for p in preds_synth],
            mode='lines',
            name=f"Connector: {conn_t}",
            line=dict(color=colors[idx % len(colors)], width=2.5)
        ))
        
    # Highlight Active Design Point
    fig4.add_trace(go.Scatter(
        x=[T_val],
        y=[pred_capacity_kN],
        mode='markers+text',
        name=f"Active Input ({T_val:.0f}°C, {pred_capacity_kN:.2f} kN)",
        marker=dict(color='#ef4444', size=15, symbol='star', line=dict(color='#ffffff', width=2)),
        text=[f"  Active Design ({pred_capacity_kN:.1f} kN)"],
        textposition="top center"
    ))
    
    fig4.update_layout(
        template="plotly_dark", height=450,
        xaxis=dict(title="Temperature (°C)", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Predicted Residual Shear Capacity (kN)", gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")

    # ---------------------------------------------------------
    # Dynamic Figure 5: Connector Geometry (Height & Diameter)
    # ---------------------------------------------------------
    st.markdown("#### Figure 5: Dynamic Connector Geometry Sensitivity (Height & Diameter)")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        height_sweep = np.linspace(40, 160, 30)
        fig5a = go.Figure()
        for t_mark in [20, 400, 600, 800]:
            h_list = []
            for h_i in height_sweep:
                r = input_dict.copy()
                r['Temperature (ºC)'] = float(t_mark)
                r['Height (mm)'] = float(h_i)
                h_list.append(r)
            h_df = pd.DataFrame(h_list)
            for col in h_df.select_dtypes(include=['object']).columns:
                h_df[col] = h_df[col].astype(str).str.strip()
            p_h = best_model.predict(preprocessor.transform(h_df))
            fig5a.add_trace(go.Scatter(
                x=height_sweep, y=[max(0.0, float(p[0])) for p in p_h],
                mode='lines', name=f"Temp = {t_mark}°C"
            ))
        fig5a.add_trace(go.Scatter(
            x=[float(height_mm)], y=[pred_capacity_kN],
            mode='markers+text', name=f"Active Height ({height_mm}mm)",
            marker=dict(color='#ef4444', size=13, symbol='diamond', line=dict(color='#ffffff', width=2)),
            text=[f" Active ({pred_capacity_kN:.1f} kN)"], textposition="top right"
        ))
        fig5a.update_layout(
            template="plotly_dark", height=400,
            xaxis=dict(title="Connector Height (mm)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Shear Capacity (kN)", gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig5a, use_container_width=True)

    with col_g2:
        diam_sweep = np.linspace(10, 30, 30)
        fig5b = go.Figure()
        for t_mark in [20, 400, 600, 800]:
            d_list = []
            for d_i in diam_sweep:
                r = input_dict.copy()
                r['Temperature (ºC)'] = float(t_mark)
                r['Diameter (mm)'] = float(d_i)
                d_list.append(r)
            d_df = pd.DataFrame(d_list)
            for col in d_df.select_dtypes(include=['object']).columns:
                d_df[col] = d_df[col].astype(str).str.strip()
            p_d = best_model.predict(preprocessor.transform(d_df))
            fig5b.add_trace(go.Scatter(
                x=diam_sweep, y=[max(0.0, float(p[0])) for p in p_d],
                mode='lines', name=f"Temp = {t_mark}°C"
            ))
        fig5b.add_trace(go.Scatter(
            x=[float(diameter_mm)], y=[pred_capacity_kN],
            mode='markers+text', name=f"Active Diam ({diameter_mm}mm)",
            marker=dict(color='#ef4444', size=13, symbol='diamond', line=dict(color='#ffffff', width=2)),
            text=[f" Active ({pred_capacity_kN:.1f} kN)"], textposition="top right"
        ))
        fig5b.update_layout(
            template="plotly_dark", height=400,
            xaxis=dict(title="Connector Diameter (mm)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Shear Capacity (kN)", gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig5b, use_container_width=True)
        
    st.markdown("---")

    # ---------------------------------------------------------
    # Dynamic Figure 6: Material Strengths (Concrete & Steel)
    # ---------------------------------------------------------
    st.markdown("#### Figure 6: Dynamic Material Strengths Sensitivity (Concrete Grade & Steel Yield)")
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        c_grade_sweep = np.linspace(20, 80, 30)
        fig6a = go.Figure()
        for t_mark in [20, 400, 600, 800]:
            cg_list = []
            for cg_i in c_grade_sweep:
                r = input_dict.copy()
                r['Temperature (ºC)'] = float(t_mark)
                r['Concrete Grade (Mpa)'] = float(cg_i)
                cg_list.append(r)
            cg_df = pd.DataFrame(cg_list)
            for col in cg_df.select_dtypes(include=['object']).columns:
                cg_df[col] = cg_df[col].astype(str).str.strip()
            p_cg = best_model.predict(preprocessor.transform(cg_df))
            fig6a.add_trace(go.Scatter(
                x=c_grade_sweep, y=[max(0.0, float(p[0])) for p in p_cg],
                mode='lines', name=f"Temp = {t_mark}°C"
            ))
        fig6a.add_trace(go.Scatter(
            x=[float(concrete_grade_MPa)], y=[pred_capacity_kN],
            mode='markers+text', name=f"Active Concrete ({concrete_grade_MPa}MPa)",
            marker=dict(color='#34d399', size=13, symbol='star', line=dict(color='#ffffff', width=2)),
            text=[f" Active ({pred_capacity_kN:.1f} kN)"], textposition="top right"
        ))
        fig6a.update_layout(
            template="plotly_dark", height=400,
            xaxis=dict(title="Concrete Grade (MPa)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Shear Capacity (kN)", gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig6a, use_container_width=True)

    with col_m2:
        fy_sweep = np.linspace(100, 500, 30)
        fig6b = go.Figure()
        for t_mark in [20, 400, 600, 800]:
            fy_list = []
            for fy_i in fy_sweep:
                r = input_dict.copy()
                r['Temperature (ºC)'] = float(t_mark)
                r['Steel fy,θ'] = float(fy_i)
                fy_list.append(r)
            fy_df = pd.DataFrame(fy_list)
            for col in fy_df.select_dtypes(include=['object']).columns:
                fy_df[col] = fy_df[col].astype(str).str.strip()
            p_fy = best_model.predict(preprocessor.transform(fy_df))
            fig6b.add_trace(go.Scatter(
                x=fy_sweep, y=[max(0.0, float(p[0])) for p in p_fy],
                mode='lines', name=f"Temp = {t_mark}°C"
            ))
        fig6b.add_trace(go.Scatter(
            x=[float(steel_fy_MPa)], y=[pred_capacity_kN],
            mode='markers+text', name=f"Active Steel fy ({steel_fy_MPa}MPa)",
            marker=dict(color='#34d399', size=13, symbol='star', line=dict(color='#ffffff', width=2)),
            text=[f" Active ({pred_capacity_kN:.1f} kN)"], textposition="top right"
        ))
        fig6b.update_layout(
            template="plotly_dark", height=400,
            xaxis=dict(title="Steel Yield Strength fy (MPa)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Shear Capacity (kN)", gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig6b, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # Dynamic Figure 10: Multi-Temperature Load-Slip Curves
    # ---------------------------------------------------------
    st.markdown("#### Figure 10: Dynamic Multi-Temperature Thermo-Structural Load-Slip Curves")
    fig10 = go.Figure()
    s_mesh = np.linspace(0.1, max(12.0, pred_slip_mm * 1.5), 100)
    
    for t_step in [20, 300, 500, 700]:
        r_step = input_dict.copy()
        r_step['Temperature (ºC)'] = float(t_step)
        p_step = best_model.predict(preprocessor.transform(pd.DataFrame([r_step])))
        cap_step = max(0.0, float(p_step[0][0]))
        slip_step = max(0.1, float(p_step[0][1]))
        
        y_curve = []
        for s_val in s_mesh:
            if s_val <= slip_step:
                val = cap_step * ((1.0 - np.exp(-3.0 * s_val / slip_step)) / (1.0 - np.exp(-3.0)))**0.65
            else:
                over = (s_val - slip_step) / slip_step
                val = cap_step * max(0.15, 1.0 - 0.45 * (over**1.1))
            y_curve.append(float(val))
            
        fig10.add_trace(go.Scatter(
            x=s_mesh, y=y_curve, mode='lines',
            name=f"Curve @ {t_step}°C (P_u={cap_step:.1f} kN)",
            line=dict(dash='dash', width=2)
        ))
        
    # Active curve
    fig10.add_trace(go.Scatter(
        x=slip_steps, y=load_steps, mode='lines',
        name=f"Active Design Curve @ {T_val:.0f}°C",
        line=dict(color='#38bdf8', width=4)
    ))
    fig10.add_trace(go.Scatter(
        x=[pred_slip_mm], y=[pred_capacity_kN], mode='markers+text',
        name=f"Active Peak ({pred_capacity_kN:.2f} kN, {pred_slip_mm:.2f} mm)",
        marker=dict(color='#ef4444', size=14, line=dict(color='#ffffff', width=2)),
        text=[f"  Peak ({pred_capacity_kN:.1f} kN)"], textposition="top center"
    ))
    
    fig10.update_layout(
        template="plotly_dark", height=480,
        xaxis=dict(title="Slip s (mm)", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Shear Force P (kN)", gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig10, use_container_width=True)
    st.markdown("---")

    # ---------------------------------------------------------
    # Dataset Benchmark Static Figures (Figures 1, 2, 3, 7, 8, 9)
    # ---------------------------------------------------------
    st.markdown("#### Dataset Benchmark Figures (Model Training & Feature Evaluation)")
    bench_figs = [
        ("Figure 1: Pearson Correlation Coefficient Matrix", "results/fig1_pearson_correlation_matrix.png"),
        ("Figure 2: Actual vs. Predicted Performance Across AI Algorithms", "results/fig2_actual_vs_predicted_all_models.png"),
        ("Figure 3: Test Specimen Prediction Tracking Comparison", "results/fig3_sample_testing_predictions_tracking.png"),
        ("Figure 7: Residual Error Distributions across AI Algorithms", "results/fig7_residual_error_distributions.png"),
        ("Figure 8: SHAP Feature Importance Summary (Ultimate Shear Capacity)", "results/fig8_shap_summary_and_feature_importance.png"),
        ("Figure 9: SHAP Dependence Plots for Dominant Parameters", "results/fig9_shap_dependence_plots.png"),
    ]
    for b_title, b_path in bench_figs:
        st.markdown(f"##### {b_title}")
        if os.path.exists(b_path):
            st.image(b_path, use_column_width=True)
        else:
            st.warning(f"Image not found at {b_path}.")
        st.markdown("---")

# -----------------------------------------------------------------------------
# Tab 5: Executive Report & Guide
# -----------------------------------------------------------------------------
with tab5:
    if os.path.exists('PROJECT_GUIDE_AND_EXECUTIVE_REPORT.md'):
        with open('PROJECT_GUIDE_AND_EXECUTIVE_REPORT.md', 'r', encoding='utf-8') as f:
            report_md = f.read()
        st.markdown(report_md)


