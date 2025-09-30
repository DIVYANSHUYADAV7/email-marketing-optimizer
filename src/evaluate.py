# src/evaluate.py
import joblib
import pandas as pd
import numpy as np
from .features import build_features, load_data

def score_contacts(model_path, contacts_csv, out_csv='scored_contacts.csv'):
    artifact = joblib.load(model_path)
    enc = artifact['enc']
    clf_open = artifact['clf_open']
    clf_click = artifact['clf_click']

    df = load_data(contacts_csv)
    X, _ = build_features(df, fit_encoder=enc)
    prob_open = clf_open.predict_proba(X)[:,1]
    prob_click = clf_click.predict_proba(X)[:,1]

    df['pred_open_prob'] = np.round(prob_open, 4)
    df['pred_click_prob'] = np.round(prob_click, 4)
    df['combined_score'] = (df['pred_open_prob']*0.6 + df['pred_click_prob']*0.4)
    df = df.sort_values('combined_score', ascending=False)
    df.to_csv(out_csv, index=False)
    print("Wrote scored contacts to", out_csv)
    return df
