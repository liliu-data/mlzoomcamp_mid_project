#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import xgboost as xgb
import pickle

def load_data():

    df = pd.read_csv('healthcare-dataset-stroke-data.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')
    
    bmi_mean = df.bmi.mean()
    df.bmi = df.bmi.fillna(bmi_mean)

    return df

def train_model(df):
    numerical = ['age', 'avg_glucose_level', 'bmi']
    categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)

    df_full_train = df_full_train.reset_index(drop=True)
    y_full_train = (df_full_train.stroke == 1).astype(int).values
    del df_full_train['stroke']
    y_test = (df_test.stroke).astype('int').values

    del df_test['stroke']
    del df_test['id']

    dicts_full_train = df_full_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dicts_full_train)
    smt = SMOTETomek(random_state=42)
    X_full_train_smt, y_full_train_smt = smt.fit_resample(X_full_train, y_full_train)

    feature_names = list(dv.get_feature_names_out())

    dfulltrain = xgb.DMatrix(X_full_train_smt, label=y_full_train_smt, 
                            feature_names=feature_names)

    dicts_test = df_test.to_dict(orient='records')
    X_test = dv.transform(dicts_test)
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
        'random_state': 42
    }

    model = xgb.train(params, dfulltrain, num_boost_round=70, verbose_eval=5)

    return model, dv

def save_model(filename, model, dv):
    with open(filename, 'wb') as f_out:
        pickle.dump((model, dv), f_out)
    print(f"Model and vectorizer saved to {filename}")


df = load_data()
model, dv = train_model(df)
save_model('model.bin', model, dv)








