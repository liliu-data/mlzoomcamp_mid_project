#!/usr/bin/env python
# coding: utf-8

"""
Enhanced training script that compares multiple machine learning models:
- Regression models: Logistic Regression
- Tree-based models: Random Forest, XGBoost, LightGBM

This script trains all models, evaluates them, and saves the best performing model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
import pickle
import json
from datetime import datetime

def load_data():
    """Load and preprocess the stroke prediction dataset."""
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')
    
    bmi_mean = df.bmi.mean()
    df.bmi = df.bmi.fillna(bmi_mean)
    
    return df

def prepare_data(df):
    """Prepare data for training."""
    numerical = ['age', 'avg_glucose_level', 'bmi']
    categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                   'work_type', 'residence_type', 'smoking_status']
    
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
    
    df_full_train = df_full_train.reset_index(drop=True)
    y_full_train = (df_full_train.stroke == 1).astype(int).values
    del df_full_train['stroke']
    
    y_test = (df_test.stroke).astype('int').values
    del df_test['stroke']
    del df_test['id']
    
    # Prepare training data
    dicts_full_train = df_full_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dicts_full_train)
    
    # Handle class imbalance with SMOTE-Tomek
    smt = SMOTETomek(random_state=42)
    X_full_train_smt, y_full_train_smt = smt.fit_resample(X_full_train, y_full_train)
    
    # Prepare test data
    dicts_test = df_test.to_dict(orient='records')
    X_test = dv.transform(dicts_test)
    
    return X_full_train_smt, y_full_train_smt, X_test, y_test, dv

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression model."""
    print("\n" + "="*60)
    print("Training Logistic Regression Model")
    print("="*60)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0
    )
    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"AUC Score: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return model, {
        'model_name': 'Logistic Regression',
        'model_type': 'Regression',
        'auc': float(auc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    print("\n" + "="*60)
    print("Training Random Forest Model")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"AUC Score: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return model, {
        'model_name': 'Random Forest',
        'model_type': 'Tree-based',
        'auc': float(auc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }

def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    """Train XGBoost model."""
    print("\n" + "="*60)
    print("Training XGBoost Model")
    print("="*60)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 3,
        'min_child_weight': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'lambda': 1,
        'alpha': 0.1,
        'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train),
        'random_state': 42
    }
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=70, 
        verbose_eval=False
    )
    
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"AUC Score: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return model, {
        'model_name': 'XGBoost',
        'model_type': 'Tree-based',
        'auc': float(auc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM model."""
    print("\n" + "="*60)
    print("Training LightGBM Model")
    print("="*60)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train)
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"AUC Score: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return model, {
        'model_name': 'LightGBM',
        'model_type': 'Tree-based',
        'auc': float(auc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }

def compare_models(results):
    """Compare all models and select the best one."""
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('auc', ascending=False)
    
    print("\nModel Performance Summary (sorted by AUC):")
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_idx = comparison_df['auc'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'model_name']
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"AUC Score: {comparison_df.loc[best_model_idx, 'auc']:.4f}")
    print(f"{'='*60}")
    
    return best_model_name, comparison_df

def save_models(models_dict, dv, results, best_model_name):
    """Save all models and comparison results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best model
    best_model = models_dict[best_model_name]
    with open('model.bin', 'wb') as f_out:
        pickle.dump((best_model, dv, best_model_name), f_out)
    print(f"\nBest model ({best_model_name}) saved to model.bin")
    
    # Save all models
    all_models_file = f'models_all_{timestamp}.bin'
    with open(all_models_file, 'wb') as f_out:
        pickle.dump((models_dict, dv, results), f_out)
    print(f"All models saved to {all_models_file}")
    
    # Also save as latest for easy access
    latest_file = 'models_all_latest.bin'
    with open(latest_file, 'wb') as f_out:
        pickle.dump((models_dict, dv, results), f_out)
    print(f"All models also saved to {latest_file} (for API use)")
    
    # Save comparison results as JSON
    results_file = f'model_comparison_{timestamp}.json'
    with open(results_file, 'w') as f_out:
        json.dump(results, f_out, indent=2)
    print(f"Comparison results saved to {results_file}")
    
    # Save comparison as CSV
    comparison_df = pd.DataFrame(results)
    csv_file = f'model_comparison_{timestamp}.csv'
    comparison_df.to_csv(csv_file, index=False)
    print(f"Comparison results saved to {csv_file}")

def main():
    """Main training pipeline."""
    print("="*60)
    print("STROKE PREDICTION MODEL TRAINING & COMPARISON")
    print("="*60)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    df = load_data()
    X_train, y_train, X_test, y_test, dv = prepare_data(df)
    
    feature_names = list(dv.get_feature_names_out())
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Train all models
    models_dict = {}
    results = []
    
    # Regression model
    lr_model, lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
    models_dict['Logistic Regression'] = lr_model
    results.append(lr_results)
    
    # Tree-based models
    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    models_dict['Random Forest'] = rf_model
    results.append(rf_results)
    
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test, feature_names)
    models_dict['XGBoost'] = xgb_model
    results.append(xgb_results)
    
    lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_test, y_test)
    models_dict['LightGBM'] = lgb_model
    results.append(lgb_results)
    
    # Compare models
    best_model_name, comparison_df = compare_models(results)
    
    # Save models and results
    save_models(models_dict, dv, results, best_model_name)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == '__main__':
    main()
