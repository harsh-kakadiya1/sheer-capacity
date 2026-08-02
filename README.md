# 🏗️ AI/ML Surrogate Model for Composite Concrete Shear Capacity at Elevated Temperatures

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sheer-capacity.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

> 🚀 **Live Interactive Web Application:** [https://sheer-capacity.streamlit.app/](https://sheer-capacity.streamlit.app/)

---

## 📌 Project Overview
This repository implements an Artificial Intelligence (AI) and Machine Learning (ML) surrogate modeling framework to predict the non-linear thermo-structural behavior and residual ultimate shear capacity ($V_u$) of composite concrete shear connectors under fire and elevated temperature conditions (up to 800°C).

The AI surrogate model bypasses computationally heavy Non-Linear Finite Element Analysis (FEA) to provide instantaneous ($< 1$ second) shear capacity predictions and full non-linear load-slip curve generation.

---

## 🌐 Live Web Application & Interactive Design Tool

You can access the deployed interactive AI prediction tool directly in your browser:

👉 **[https://sheer-capacity.streamlit.app/](https://sheer-capacity.streamlit.app/)**

### Key Web App Features:
- 🎛️ **Interactive Design Panel:** Adjust connector geometry, concrete grade, steel yield strength, and temperature exposure in real-time.
- ⚡ **Instant AI Predictions:** Get immediate predictions for Ultimate Shear Capacity ($V_u$ in kN) and Slip ($\delta$ in mm).
- 📈 **Dynamic Load-Slip Curves:** High-resolution Plotly charts showing non-linear structural degradation across temperature ranges.
- 🔍 **Explainable AI (SHAP):** Live visual breakdown of how each physical parameter influences the final prediction.

---

## 📊 Performance Metrics

| Model | Shear Capacity $R^2$ | Shear Capacity RMSE (kN) | Slip $R^2$ | Slip RMSE (mm) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Best)** | **0.9935** | **17.89** | **0.9856** | **0.29** |
| **Random Forest** | 0.9892 | 23.14 | 0.9781 | 0.36 |
| **Decision Tree** | 0.9745 | 35.62 | 0.9610 | 0.48 |
| **Multi-Layer Perceptron** | 0.9412 | 52.80 | 0.9105 | 0.73 |

---

## 💻 Local Setup & Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harsh-kakadiya1/sheer-capacity.git
   cd sheer-capacity
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit app locally:**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📁 Repository Structure

```
sheer-capacity/
├── data/                                 # Master experimental & FEA simulation datasets (1,235 records)
├── models/                               # Serialized AI models (XGBoost, RF, preprocessor)
├── results/                              # Output metrics, scatter plots, & SHAP charts
├── src/                                  # Python source code for model training & explainability
├── streamlit_app.py                      # Interactive Streamlit Web Application
├── requirements.txt                      # Package dependencies for local & Streamlit Cloud deployment
├── PROJECT_GUIDE_AND_EXECUTIVE_REPORT.md # Detailed technical guide & executive non-technical report
└── README.md                             # Repository homepage (this file)
```

---

## 📄 Executive Report & Documentation
For full research background, feature engineering details, and non-technical explanations, refer to [PROJECT_GUIDE_AND_EXECUTIVE_REPORT.md](file:///d:/sheer-capacity/PROJECT_GUIDE_AND_EXECUTIVE_REPORT.md).
