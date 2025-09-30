# src/features.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_features(df, fit_encoder=None):
    # Simple numeric features
    X_num = df[['age','past_opens','past_clicks','avg_session_minutes']].fillna(0)

    # One-hot encode location + gender
    cat_cols = ['gender','location']
    if fit_encoder is None:
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_cat = enc.fit_transform(df[cat_cols])
        return np.hstack([X_num.values, X_cat]), enc
    else:
        enc = fit_encoder
        X_cat = enc.transform(df[cat_cols])
        return np.hstack([X_num.values, X_cat]), enc
