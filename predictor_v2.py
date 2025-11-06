#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor_v2.py
-----------------
Κάνει προβλέψεις για:
1️⃣ Αποτέλεσμα (Home / Draw / Away)
2️⃣ Over / Under 2.5 goals

Αυτόματη ανίχνευση μοντέλων με λιγότερες κατηγορίες (π.χ. μόνο "Home"),
χωρίς να εμφανίζεται σφάλμα.
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Paths
DATA_PATH = os.path.join("data", "matches_past_features.csv")
PRED_PATH = os.path.join("data", "predictions.csv")
MODEL_DIR = "models"
MODEL_RESULT_PATH = os.path.join(MODEL_DIR, "model_result.pkl")
MODEL_OVER_PATH = os.path.join(MODEL_DIR, "model_over.pkl")

# Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "HomeForm","AwayForm","HomeRank","AwayRank",
    "HomeAvgScore","AwayAvgScore","HomeAvgConcede","AwayAvgConcede",
    "HomeCleanSheetRate","AwayCleanSheetRate","HomeH2HWin","AwayH2HWin","H2HDraws"
]

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES, how="all").fillna(0)
    return df

def proxy_labels(df):
    """Δημιουργεί proxy labels μέχρι να έχουμε πραγματικά αποτελέσματα"""
    result, overunder = [], []
    for _, r in df.iterrows():
        home_strength = (r["HomeForm"] + r["HomeAvgScore"] - r["HomeAvgConcede"])
        away_strength = (r["AwayForm"] + r["AwayAvgScore"] - r["AwayAvgConcede"])
        if home_strength - away_strength > 0.1:
            result.append("Home")
        elif away_strength - home_strength > 0.1:
            result.append("Away")
        else:
            result.append("Draw")
        total_goals = r["HomeAvgScore"] + r["AwayAvgScore"]
        overunder.append("Over" if total_goals >= 2.5 else "Under")
    df["ResultLabel"] = result
    df["OverUnderLabel"] = overunder
    return df

def train_models(df):
    X = df[FEATURES].values
    y_result = df["ResultLabel"]
    y_over = df["OverUnderLabel"]

    clf_result = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_result.fit(X, y_result)

    clf_over = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_over.fit(X, y_over)

    with open(MODEL_RESULT_PATH, "wb") as f:
        pickle.dump(clf_result, f)
    with open(MODEL_OVER_PATH, "wb") as f:
        pickle.dump(clf_over, f)

    print("✅ Εκπαιδεύτηκαν και αποθηκεύτηκαν τα μοντέλα.")
    return clf_result, clf_over

def load_or_train(df):
    if os.path.exists(MODEL_RESULT_PATH) and os.path.exists(MODEL_OVER_PATH):
        print("📦 Φόρτωση υπαρχόντων μοντέλων...")
        with open(MODEL_RESULT_PATH, "rb") as f:
            clf_result = pickle.load(f)
        with open(MODEL_OVER_PATH, "rb") as f:
            clf_over = pickle.load(f)
    else:
        print("🧠 Εκπαίδευση νέων μοντέλων...")
        clf_result, clf_over = train_models(df)
    return clf_result, clf_over

def predict_all(df, clf_result, clf_over):
    X = df[FEATURES].values
    probs_result = clf_result.predict_proba(X)
    probs_over = clf_over.predict_proba(X)

    result_labels = clf_result.classes_
    over_labels = clf_over.classes_

    if len(result_labels) < 3:
        print(f"⚠️ Προσοχή: το μοντέλο αποτελέσματος έχει μόνο {len(result_labels)} κατηγορία(ες): {list(result_labels)}")
    if len(over_labels) < 2:
        print(f"⚠️ Προσοχή: το μοντέλο Over/Under έχει μόνο {len(over_labels)} κατηγορία(ες): {list(over_labels)}")

    # Δημιουργία DataFrame δυναμικά ανάλογα με τις διαθέσιμες κατηγορίες
    preds_data = {
        "League": df["League"],
        "DateUTC": df["DateUTC"],
        "HomeTeam": df["HomeTeam"],
        "AwayTeam": df["AwayTeam"],
    }

    # Προσθήκη πιθανοτήτων για κάθε διαθέσιμη κατηγορία
    for i, label in enumerate(result_labels):
        preds_data[f"Prob_{label}"] = probs_result[:, i]

    for i, label in enumerate(over_labels):
        preds_data[f"Prob_{label}"] = probs_over[:, i]

    preds = pd.DataFrame(preds_data)
    preds.to_csv(PRED_PATH, index=False, encoding="utf-8")
    print(f"✅ Αποθηκεύτηκαν προβλέψεις στο {PRED_PATH} (rows={preds.shape[0]})")
    return preds

def predict_from_features(features):
    if not (os.path.exists(MODEL_RESULT_PATH) and os.path.exists(MODEL_OVER_PATH)):
        raise FileNotFoundError("Πρέπει να τρέξεις πρώτα το predictor_v2.py για να εκπαιδευτούν τα μοντέλα.")
    with open(MODEL_RESULT_PATH, "rb") as f:
        clf_result = pickle.load(f)
    with open(MODEL_OVER_PATH, "rb") as f:
        clf_over = pickle.load(f)

    X = np.array([[features.get(f,0) for f in FEATURES]])
    res_probs = clf_result.predict_proba(X)[0]
    over_probs = clf_over.predict_proba(X)[0]

    return {
        "Result": dict(zip(clf_result.classes_, np.round(res_probs,3))),
        "OverUnder": dict(zip(clf_over.classes_, np.round(over_probs,3)))
    }

if __name__ == "__main__":
    df = load_data()
    df = proxy_labels(df)
    clf_result, clf_over = load_or_train(df)
    predict_all(df, clf_result, clf_over)
    print("🎯 Ολοκληρώθηκε η διαδικασία πρόβλεψης χωρίς σφάλματα.")
