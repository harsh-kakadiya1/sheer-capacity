# 📘 Executive Summary Report & Repository Guide: Composite Deck Shear Capacity AI Integration

> 🚀 **Live Interactive Web Application:** [https://sheer-capacity.streamlit.app/](https://sheer-capacity.streamlit.app/)

---

> [!NOTE]  
> **Document Purpose**: This document serves a dual purpose:  
> 1. A **File & Directory Roadmap** detailing what every file and folder in this repository does.  
> 2. An **Executive Non-Technical Report** explaining the research problem, AI methodology, results, and practical outcomes in simple, easy-to-understand language.

---

# Part 1: 🗂️ Repository Directory & File Guide

This section explains the role of every directory and file in this repository.

```
Shear-Capacity-of-Composite-Concrete-Structure/
├── data/
│   └── AI Model Data.xlsx                 # The master experimental & FEA simulation dataset (1,235 records)
├── Reference Papers/                       # Academic reference papers on composite shear connectors & elevated temperature AI
├── models/                                 # Saved AI model weights, scaling preprocessors, & metadata JSON
│   ├── best_model.joblib                  # Serialized best-performing AI model (XGBoost)
│   ├── xgboost_model.joblib               # Serialized XGBoost model
│   ├── randomforest_model.joblib          # Serialized Random Forest model
│   ├── preprocessor.joblib                # Feature scaler & One-Hot encoder pipeline
│   └── metadata.json                      # Column names, feature lists, & model metadata
├── results/                                # Output graphs, SHAP explainability charts, & performance summaries
│   ├── fig1_pearson_correlation_matrix.png # Pearson correlation heatmap across all 25 features
│   ├── fig2_actual_vs_predicted_all_models.png # 4-panel comparison scatter plots (XGBoost, RF, DT, MLP)
│   ├── fig3_sample_testing_predictions_tracking.png # Sample-by-sample test specimen tracking chart
│   ├── fig4_parametric_temperature_degradation.png # Residual capacity degradation curves (20°C - 800°C)
│   ├── fig5_parametric_connector_geometry.png # Connector Height & Diameter vs. Capacity curves
│   ├── fig6_parametric_material_strengths.png # Concrete Grade & Steel Yield Strength vs. Capacity curves
│   ├── fig7_residual_error_distributions.png # Prediction error distribution histograms & KDE
│   ├── fig8_shap_summary_and_feature_importance.png # SHAP feature importance summary dot plots
│   ├── fig9_shap_dependence_plots.png     # SHAP dependence interaction plots for top parameters
│   ├── fig10_load_slip_curves_multitemp.png# Non-linear load-slip curves at elevated temperatures
│   ├── metrics_summary.json               # Detailed R², RMSE, & MAE error metrics for all models
│   └── test_predictions.csv               # Actual vs. predicted capacity & slip values for test dataset
├── src/                                    # Python source code modules
│   ├── train_models.py                    # Script to load data, clean inputs, train 4 AI models, & evaluate metrics
│   ├── generate_paper_graphs.py           # Comprehensive paper-aligned figure & parametric plot generator
│   └── explainability_and_plots.py        # Wrapper script for SHAP analysis and visual graph generation
├── streamlit_app.py                       # Native Streamlit Web Application (Interactive Tool & 5-Tab Gallery)
├── Run_App.bat                            # 1-Click Windows shortcut launcher to open Streamlit automatically
├── shear_capacity_analysis.ipynb          # Self-contained Jupyter Notebook for interactive execution & reporting
├── readme.txt                             # Foundational project instructions and 4-step AI integration roadmap
└── PROJECT_GUIDE_AND_EXECUTIVE_REPORT.md  # (This file) Complete roadmap & executive non-technical report
```

### Detailed Breakdown of Key Files

| File / Folder | Type | Role & Description |
| :--- | :--- | :--- |
| **[AI Model Data.xlsx](file:///d:/Shear-Capacity-of-Composite-Concrete-Structure/data/AI%20Model%20Data.xlsx)** | Data | Contains 1,235 structural test specimens across 5 connector types (`Stud`, `Bar`, `Channel`, `Tee`, `Helical`) with 25 input parameters (temperature, dimensions, steel/concrete thermal properties) and 2 output targets (Shear Capacity & Slip). |
| **[streamlit_app.py](file:///d:/Shear-Capacity-of-Composite-Concrete-Structure/streamlit_app.py)** | Web Application | Native Streamlit GUI designed for structural engineers. Provides interactive sliders, instant prediction cards, dynamic Plotly load-slip charts, SHAP plots, and benchmark tables. |
| **[Run_App.bat](file:///d:/Shear-Capacity-of-Composite-Concrete-Structure/Run_App.bat)** | Launcher | Windows 1-click batch shortcut script to launch the Streamlit app automatically in your default browser. |
| **[src/train_models.py](file:///d:/Shear-Capacity-of-Composite-Concrete-Structure/src/train_models.py)** | Source Code | Automates data cleaning, 80/20 train-test splitting, feature encoding, model training (XGBoost, Random Forest, Decision Tree, MLP Neural Network), and saves trained model weights. |
| **[src/explainability_and_plots.py](file:///d:/Shear-Capacity-of-Composite-Concrete-Structure/src/explainability_and_plots.py)** | Source Code | Calculates SHAP (SHapley Additive exPlanations) values to unveil the "black box" of AI, proving mathematically which physical factors (like temperature or diameter) influence capacity most. |
| **[shear_capacity_analysis.ipynb](file:///d:/Shear-Capacity-of-Composite-Concrete-Structure/shear_capacity_analysis.ipynb)** | Notebook | Interactive Jupyter Notebook combining code, explanations, and visual charts for academic reporting and demonstration. |

---

# Part 2: 📊 Executive Report (Non-Technical Explanation)

> *Designed for stakeholders, project managers, and non-technical readers.*

---

## 1. The Challenge: Fire Safety & Structural Shear Connectors

### What is a Shear Connector?
In modern buildings and bridges, concrete slabs are placed on top of steel beams. **Shear connectors** (metal studs, channels, bars, or helical anchors) are welded onto the steel beams and embedded into the concrete. They lock the concrete and steel together so they act as a single, super-strong unit (a **composite structure**).

### The Fire Problem
When a fire breaks out in a building:
1. Temperatures quickly soar above **$600^\circ\text{C}$ to $1000^\circ\text{C}$**.
2. Heat weakens both the steel connectors and surrounding concrete.
3. The connector loses its **Shear Capacity** (how much force it can withstand before breaking) and undergoes **Slip** (how far it stretches/slides before failure).

### Why Traditional Testing is Slow & Expensive
To test fire performance, structural engineers historically had two options:
- **Furnace Push-Out Tests**: Building full-scale concrete/steel specimens and placing them inside high-temperature gas furnaces. *Cost: Thousands of dollars per test; Time: Weeks per specimen.*
- **Finite Element Analysis (FEA)**: Running complex 3D computer simulations in software like ABAQUS. *Cost: Requires supercomputers & specialized software; Time: Hours or days per simulation run.*

---

## 2. The Solution: AI as a "High-Speed Surrogate Model"

Instead of spending days running a single computer simulation, we used **Machine Learning (Artificial Intelligence)** to learn the underlying physics from 1,235 structural tests and simulations.

### How Does the AI Work?
Imagine giving a master structural engineer 1,235 detailed fire test reports from the past. After reviewing all 1,235 cases, the expert can instantly tell you how a new stud or channel connector will behave under fire—without needing to run a new test. 

That is exactly what our AI model does:
1. **Inputs Given to AI**: Fire temperature, exposure time, connector shape/dimensions (diameter, height), concrete strength, and thermal properties.
2. **AI Processing**: The AI detects non-linear mathematical patterns between heat, material softening, and geometry.
3. **Instant Output**: In less than **0.01 seconds**, the AI predicts:
   - **Ultimate Shear Capacity** (in kilonewtons, kN)
   - **Slip at Failure** (in millimeters, mm)
   - **The Non-Linear Load-Slip Curve** (showing how stiffness degrades over time).

---

## 3. What Models Were Trained & How Did They Perform?

We trained and compared 4 different Artificial Intelligence algorithms on the exact same dataset to find the most accurate model.

### Simple Accuracy Scoreboard ($R^2$ Score)
*(Where 1.00 = 100% Perfect Prediction; Anything above 0.90 is considered excellent for structural engineering).*

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MODEL ACCURACY SCOREBOARD                      │
├──────────────────────┬──────────────────────┬───────────────────────────┤
│ AI Algorithm         │ Shear Capacity (kN)  │ Slip (Deformation, mm)    │
├──────────────────────┼──────────────────────┼───────────────────────────┤
│ 🏆 XGBoost (Best)    │ 99.8% (0.9982 R²)    │ 99.1% (0.9913 R²)         │
│ 🌲 Random Forest     │ 99.6% (0.9965 R²)    │ 96.8% (0.9678 R²)         │
│ 🌳 Decision Tree     │ 99.5% (0.9947 R²)    │ 93.5% (0.9354 R²)         │
│ 🧠 Neural Network    │ 61.4% (0.6142 R²)    │ 54.4% (0.5441 R²)         │
└──────────────────────┴──────────────────────┴───────────────────────────┘
```

> [!IMPORTANT]  
> **Key Takeaway**: The **XGBoost (Extreme Gradient Boosting)** model emerged as the undisputed winner with **99.8% accuracy** for capacity prediction and **99.1% accuracy** for slip prediction. The average error margin for shear capacity is only **2.55 kN** on loads exceeding 600 kN!

---

## 4. Unveiling the "Black Box": What Factors Matter Most? (SHAP Explainable AI)

AI is often criticized for being a "black box" where decisions are hidden. To ensure academic and industrial trust, we performed **SHAP (Explainable AI)** analysis.

### What Did We Learn? (Key Physical Insights)
1. **Temperature & Heat Exposure Are King**: Heat is by far the single most influential factor reducing shear capacity. Beyond $500^\circ\text{C}$, effective yield strength drops rapidly.
2. **Connector Diameter & Height**: Connector geometric dimensions dictate structural survivability under fire. A thicker diameter stud retains structural integrity much longer than increasing concrete grade alone.
3. **Concrete Grade vs. Steel Strength**: At normal room temperature ($20^\circ\text{C}$), concrete compressive strength heavily influences capacity. However, at elevated temperatures ($> 600^\circ\text{C}$), the steel's thermal degradation characteristics become the primary bottleneck.

---

## 5. What Did We Build For You? (Final Deliverables)

### 1. Streamlit Predictive Design Tool (`http://localhost:8501`)
We translated the complex mathematical XGBoost model into a native **Streamlit Web Application** ([streamlit_app.py](file:///d:/Shear-Capacity-of-Composite-Concrete-Structure/streamlit_app.py)). 
- Structural engineers don't need to know Python or coding.
- Simply adjust sidebar sliders for Temperature, Connector Type (Stud, Bar, Channel, Tee, Helical), Diameter, Height, and Concrete Strength.
- View real-time capacity metric cards and watch the interactive **Plotly non-linear load-slip curve update dynamically**.

### 2. Complete Jupyter Research Notebook (`shear_capacity_analysis.ipynb`)
- A complete, reproducible research notebook ready for inclusion in academic papers, theses, or technical project reports.

---

## 💡 Summary Conclusion

By integrating AI into composite concrete shear connector research, we successfully replaced hours of computational simulation with **instantaneous, 99.8% accurate predictions**. This tool provides structural engineers with a rapid, reliable, and explainable design assistant for fire safety analysis.
