# src/trainer.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from .features import load_data, build_features

def train_models(csv_path, out_model_prefix="model"):
    df = load_data(csv_path)

    # Predict 'opened' and 'clicked' independently (two labels)
    X, enc = build_features(df)
    y_open = df['opened'].values
    y_click = df['clicked'].values

    X_train, X_test, y_open_train, y_open_test = train_test_split(X, y_open, test_size=0.2, random_state=42)
    _, _, y_click_train, y_click_test = train_test_split(X, y_click, test_size=0.2, random_state=42)

    # Model for open
    clf_open = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_open.fit(X_train, y_open_train)
    preds_open = clf_open.predict_proba(X_test)[:,1]
    print("Open ROC AUC:", roc_auc_score(y_open_test, preds_open))
    print(classification_report(y_open_test, clf_open.predict(X_test)))

    # Model for click
    clf_click = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_click.fit(X_train, y_click_train)
    preds_click = clf_click.predict_proba(X_test)[:,1]
    print("Click ROC AUC:", roc_auc_score(y_click_test, preds_click))
    print(classification_report(y_click_test, clf_click.predict(X_test)))

    # Save models + encoder together
    artifact = {
        'enc': enc,
        'clf_open': clf_open,
        'clf_click': clf_click,
    }
    joblib.dump(artifact, f"{out_model_prefix}.joblib")
    print("Saved model to", f"{out_model_prefix}.joblib")
    return artifact
