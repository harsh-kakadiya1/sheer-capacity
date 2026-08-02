import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
import xgboost as xgb

def load_and_preprocess_data(data_path):
    df = pd.read_excel(data_path, sheet_name='AI DATA')
    df.columns = [' '.join(col.split()) for col in df.columns]
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    grade_col = [col for col in df.columns if 'Concrete' in col and 'Grade' in col][0]
    df[grade_col] = df[grade_col].str.replace('M', '', regex=False).astype(float)
    return df

def generate_plots_and_explainability():
    os.makedirs('results', exist_ok=True)
    
    data_path = r'd:/Shear-Capacity-of-Composite-Concrete-Structure/data/AI Model Data.xlsx'
    df = load_and_preprocess_data(data_path)
    
    target_capacity_col = [col for col in df.columns if 'Ultimate' in col and 'Shear' in col][0]
    target_slip_col = [col for col in df.columns if 'Slip' in col][0]
    targets = [target_capacity_col, target_slip_col]
    feature_cols = [c for c in df.columns if c not in targets]
    cat_cols = [c for c in feature_cols if 'Connector' in c]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    X = df[feature_cols]
    y = df[targets]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = joblib.load('models/preprocessor.joblib')
    
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    all_feature_names = ohe_cols + num_cols
    
    # Single-target XGBoost models for SHAP
    xgb_cap = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    xgb_cap.fit(X_train_trans, y_train.iloc[:, 0])
    
    xgb_slip = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    xgb_slip.fit(X_train_trans, y_train.iloc[:, 1])
    
    # 1. SHAP Analysis
    print("Computing SHAP values...")
    explainer_cap = shap.TreeExplainer(xgb_cap)
    shap_values_cap = explainer_cap(X_test_trans)
    
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values_cap, X_test_trans, feature_names=all_feature_names, show=False)
    plt.title("SHAP Feature Importance - Ultimate Shear Capacity (kN)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig('results/shap_summary_shear.png')
    plt.close()
    
    explainer_slip = shap.TreeExplainer(xgb_slip)
    shap_values_slip = explainer_slip(X_test_trans)
    
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values_slip, X_test_trans, feature_names=all_feature_names, show=False)
    plt.title("SHAP Feature Importance - Slip (mm)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig('results/shap_summary_slip.png')
    plt.close()
    
    print("Explainability & visual plots successfully generated in results/")

if __name__ == '__main__':
    generate_plots_and_explainability()
