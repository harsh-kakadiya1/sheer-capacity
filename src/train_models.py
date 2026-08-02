import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

def load_and_preprocess_data(data_path):
    df = pd.read_excel(data_path, sheet_name='AI DATA')
    
    # Normalize column names across all scripts (remove internal newlines & extra spaces)
    df.columns = [' '.join(col.split()) for col in df.columns]
    
    # Strip whitespace from string values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Process Concrete Grade: 'M20' -> 20.0
    grade_col = [col for col in df.columns if 'Concrete' in col and 'Grade' in col][0]
    df[grade_col] = df[grade_col].astype(str).str.replace('M', '', regex=False).astype(float)
    
    target_capacity_col = [col for col in df.columns if 'Ultimate' in col and 'Shear' in col][0]
    target_slip_col = [col for col in df.columns if 'Slip' in col][0]
    
    targets = [target_capacity_col, target_slip_col]
    feature_cols = [c for c in df.columns if c not in targets]
    
    X = df[feature_cols].copy()
    y = df[targets].copy()
    
    cat_cols = [c for c in feature_cols if 'Connector' in c]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )
    
    return X, y, preprocessor, feature_cols, cat_cols, num_cols, targets

def train_and_evaluate():
    data_path = r'd:/Shear-Capacity-of-Composite-Concrete-Structure/data/AI Model Data.xlsx'
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("Loading and preprocessing dataset...")
    X, y, preprocessor, feature_cols, cat_cols, num_cols, targets = load_and_preprocess_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset split: Train shape = {X_train.shape}, Test shape = {X_test.shape}")
    
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
        'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'MLP_NeuralNet': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42, early_stopping=True)
    }
    
    results = {}
    best_model_name = None
    best_r2_avg = -float('inf')
    best_fitted_model = None

    for name, model in models.items():
        print(f"\nTraining model: {name}...")
        model.fit(X_train_trans, y_train)
        
        y_train_pred = model.predict(X_train_trans)
        y_test_pred = model.predict(X_test_trans)
        
        r2_cap_train = r2_score(y_train.iloc[:, 0], y_train_pred[:, 0])
        rmse_cap_train = np.sqrt(mean_squared_error(y_train.iloc[:, 0], y_train_pred[:, 0]))
        mae_cap_train = mean_absolute_error(y_train.iloc[:, 0], y_train_pred[:, 0])

        r2_slip_train = r2_score(y_train.iloc[:, 1], y_train_pred[:, 1])
        rmse_slip_train = np.sqrt(mean_squared_error(y_train.iloc[:, 1], y_train_pred[:, 1]))
        mae_slip_train = mean_absolute_error(y_train.iloc[:, 1], y_train_pred[:, 1])

        r2_cap_test = r2_score(y_test.iloc[:, 0], y_test_pred[:, 0])
        rmse_cap_test = np.sqrt(mean_squared_error(y_test.iloc[:, 0], y_test_pred[:, 0]))
        mae_cap_test = mean_absolute_error(y_test.iloc[:, 0], y_test_pred[:, 0])

        r2_slip_test = r2_score(y_test.iloc[:, 1], y_test_pred[:, 1])
        rmse_slip_test = np.sqrt(mean_squared_error(y_test.iloc[:, 1], y_test_pred[:, 1]))
        mae_slip_test = mean_absolute_error(y_test.iloc[:, 1], y_test_pred[:, 1])

        avg_test_r2 = (r2_cap_test + r2_slip_test) / 2.0
        
        results[name] = {
            'train': {
                'capacity': {'r2': float(r2_cap_train), 'rmse': float(rmse_cap_train), 'mae': float(mae_cap_train)},
                'slip': {'r2': float(r2_slip_train), 'rmse': float(rmse_slip_train), 'mae': float(mae_slip_train)}
            },
            'test': {
                'capacity': {'r2': float(r2_cap_test), 'rmse': float(rmse_cap_test), 'mae': float(mae_cap_test)},
                'slip': {'r2': float(r2_slip_test), 'rmse': float(rmse_slip_test), 'mae': float(mae_slip_test)},
                'avg_r2': float(avg_test_r2)
            }
        }
        
        print(f"  [{name}] Test Shear Capacity R^2: {r2_cap_test:.4f}, RMSE: {rmse_cap_test:.4f} kN, MAE: {mae_cap_test:.4f} kN")
        print(f"  [{name}] Test Slip R^2: {r2_slip_test:.4f}, RMSE: {rmse_slip_test:.4f} mm, MAE: {mae_slip_test:.4f} mm")

        if avg_test_r2 > best_r2_avg:
            best_r2_avg = avg_test_r2
            best_model_name = name
            best_fitted_model = model

    print(f"\nBest Model: {best_model_name} with Average Test R^2 = {best_r2_avg:.4f}")

    with open('results/metrics_summary.json', 'w') as f:
        json.dump(results, f, indent=4)

    joblib.dump(best_fitted_model, 'models/best_model.joblib')
    joblib.dump(models['XGBoost'], 'models/xgboost_model.joblib')
    joblib.dump(models['RandomForest'], 'models/randomforest_model.joblib')
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    meta_info = {
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'targets': targets,
        'best_model_name': best_model_name
    }
    with open('models/metadata.json', 'w') as f:
        json.dump(meta_info, f, indent=4)

    test_df = X_test.copy()
    test_df['Actual_Capacity_kN'] = y_test.iloc[:, 0].values
    test_df['Actual_Slip_mm'] = y_test.iloc[:, 1].values
    
    best_pred = best_fitted_model.predict(X_test_trans)
    test_df['Predicted_Capacity_kN'] = best_pred[:, 0]
    test_df['Predicted_Slip_mm'] = best_pred[:, 1]
    
    test_df.to_csv('results/test_predictions.csv', index=False)
    print("\nModel training & evaluation complete! Artifacts saved to models/ and results/.")

if __name__ == '__main__':
    train_and_evaluate()
