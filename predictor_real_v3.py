#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor_real_v3.py
--------------------
Εκπαιδεύει μοντέλα πρόβλεψης με ΠΡΑΓΜΑΤΙΚΑ labels (και proxy fallback όπου λείπουν)
και γράφει πιθανοτητες για αποτέλεσμα & Over/Under σε data/predictions_real.csv

Παράγει/χρησιμοποιεί:
- Input:  data/matches_past_features_labeled.csv
- Output: data/predictions_real.csv
- Models: models/model_result_real.pkl, models/model_over_real.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# ---------- Ρυθμίσεις ----------
DATA_IN  = os.path.join("data", "matches_past_features_labeled.csv")
DATA_OUT = os.path.join("data", "predictions_real.csv")
MODEL_DIR = "models"
MODEL_RES = os.path.join(MODEL_DIR, "model_result_real.pkl")
MODEL_OU  = os.path.join(MODEL_DIR, "model_over_real.pkl")

# Αν True, θα εκπαιδεύει κάθε φορά από την αρχή (πιο “φρέσκο”)
# Αν False, θα φορτώνει έτοιμα μοντέλα αν υπάρχουν
FORCE_TRAIN = True

# Τα features που χρησιμοποιούμε (ταιριάζουν με αυτά που φτιάξαμε νωρίτερα)
FEATURES = [
    "HomeForm","AwayForm","HomeRank","AwayRank",
    "HomeAvgScore","AwayAvgScore","HomeAvgConcede","AwayAvgConcede",
    "HomeCleanSheetRate","AwayCleanSheetRate",
    "HomeH2HWin","AwayH2HWin","H2HDraws",
]

# ---------- Βοηθητικά ----------
def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_df():
    if not os.path.exists(DATA_IN):
        raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο: {DATA_IN}")
    df = pd.read_csv(DATA_IN)
    df.columns = df.columns.str.strip()
    # Γέμισε ό,τι feature λείπει με 0
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURES] = df[FEATURES].fillna(0.0)
    # Βεβαιώσου ότι βασικές στήλες υπάρχουν
    for base in ["League","DateUTC","HomeTeam","AwayTeam"]:
        if base not in df.columns:
            df[base] = ""
    return df

def make_proxy_labels(df):
    """Proxy labels όταν λείπουν τα πραγματικά."""
    res_proxy, ou_proxy = [], []
    for _, r in df.iterrows():
        home_strength = r["HomeForm"] + r["HomeAvgScore"] - r["HomeAvgConcede"]
        away_strength = r["AwayForm"] + r["AwayAvgScore"] - r["AwayAvgConcede"]
        if home_strength - away_strength > 0.10:
            res_proxy.append("Home")
        elif away_strength - home_strength > 0.10:
            res_proxy.append("Away")
        else:
            res_proxy.append("Draw")
        total_goals = r["HomeAvgScore"] + r["AwayAvgScore"]
        ou_proxy.append("Over" if total_goals >= 2.5 else "Under")
    return pd.Series(res_proxy, index=df.index), pd.Series(ou_proxy, index=df.index)

def build_training_labels(df):
    """
    Φτιάχνει στήλες:
      TrueResult, TrueOverUnder (από πραγματικά δεδομένα όπου υπάρχουν)
      TrainResult, TrainOverUnder (τα labels που θα χρησιμοποιήσει το μοντέλο: real ή proxy)
    """
    # Αν υπάρχουν πραγματικά, κράτα τα
    has_real_result = "Result" in df.columns
    has_real_ou     = "OverUnderLabel" in df.columns

    df["TrueResult"] = df["Result"] if has_real_result else np.nan
    df["TrueOverUnder"] = df["OverUnderLabel"] if has_real_ou else np.nan

    # Proxy όπου λείπουν
    proxy_res, proxy_ou = make_proxy_labels(df)
    df["TrainResult"] = df["TrueResult"].copy()
    df["TrainOverUnder"] = df["TrueOverUnder"].copy()
    df["TrainResult"] = df["TrainResult"].fillna(proxy_res)
    df["TrainOverUnder"] = df["TrainOverUnder"].fillna(proxy_ou)

    # Αναφορές
    real_res_count = df["TrueResult"].notna().sum()
    real_ou_count  = df["TrueOverUnder"].notna().sum()
    print(f"ℹ️  Πραγματικά labels (Result): {real_res_count} / {len(df)}")
    print(f"ℹ️  Πραγματικά labels (Over/Under): {real_ou_count} / {len(df)}")

    return df

def class_weight_for(y):
    classes = np.unique(y)
    if len(classes) <= 1:
        return None  # δεν έχει νόημα
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

def train_or_load(df, target_col, model_path):
    """
    Εκπαιδεύει ή φορτώνει RandomForest για το target_col.
    Χρησιμοποιεί μόνο γραμμές όπου υπάρχει label στο target_col.
    """
    # Φίλτρο: μόνο όπου υπάρχει label
    d = df[df[target_col].notna()].copy()
    if d.empty:
        raise RuntimeError(f"Δεν βρέθηκαν labels για {target_col}.")

    X = d[FEATURES].values
    y = d[target_col].astype(str).values
    labels = np.unique(y)

    if len(labels) < 2:
        # Έχουμε μόνο 1 κατηγορία → απλό "μοντέλο" που επιστρέφει πάντα αυτή την κατηγορία
        print(f"⚠️  Προσοχή: {target_col} έχει μόνο 1 κατηγορία: {labels.tolist()}")
        model = ("constant", labels[0])  # αποθηκεύουμε tuple ως “σταθερό” μοντέλο
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        return model, labels

    if (not FORCE_TRAIN) and os.path.exists(model_path):
        print(f"📦 Φόρτωση μοντέλου από {model_path} ...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model, labels

    print(f"🧠 Εκπαίδευση RandomForest για {target_col} σε {len(d)} δείγματα ...")
    cw = class_weight_for(y)
    # RandomForest με περισσότερα δέντρα για σταθερότητα
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight=cw,
        n_jobs=-1
    )
    clf.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"✅ Αποθηκεύτηκε μοντέλο: {model_path}")
    return clf, clf.classes_

def proba_from_model(model, X, wanted_labels):
    """
    Επιστρέφει πιθανότητες για τα wanted_labels με ασφάλεια (ακόμα κι αν το μοντέλο έχει λιγότερες κλάσεις).
    - Αν το "μοντέλο" είναι ("constant", label), γυρνά 1.0 στην label και 0.0 στις υπόλοιπες.
    """
    if isinstance(model, tuple) and model[0] == "constant":
        const = model[1]
        out = np.zeros((X.shape[0], len(wanted_labels)), dtype=float)
        if const in wanted_labels:
            j = wanted_labels.index(const)
            out[:, j] = 1.0
        return out

    probs = model.predict_proba(X)  # σχήμα: [n_samples, n_classes_model]
    model_labels = list(model.classes_)
    out = np.zeros((X.shape[0], len(wanted_labels)), dtype=float)
    for j, lbl in enumerate(wanted_labels):
        if lbl in model_labels:
            mj = model_labels.index(lbl)
            out[:, j] = probs[:, mj]
        else:
            out[:, j] = 0.0
    return out

def main():
    ensure_dirs()
    df = load_df()
    df = build_training_labels(df)

    # ---- Εκπαίδευση/Φόρτωση μοντέλων ----
    model_res, labels_res = train_or_load(df, target_col="TrainResult",    model_path=MODEL_RES)
    model_ou,  labels_ou  = train_or_load(df, target_col="TrainOverUnder", model_path=MODEL_OU)

    # Θέλουμε οι στήλες να είναι με αυτή τη σειρά αν υπάρχουν:
    desired_res_labels = ["Home","Draw","Away"]
    desired_ou_labels  = ["Over","Under"]

    # ---- Πρόβλεψη για ΟΛΕΣ τις γραμμές του αρχείου (για έλεγχο/σύγκριση) ----
    X_all = df[FEATURES].values
    res_probs = proba_from_model(model_res, X_all, desired_res_labels)
    ou_probs  = proba_from_model(model_ou,  X_all, desired_ou_labels)

    # ---- Χτίσιμο πίνακα αποτελεσμάτων ----
    out = pd.DataFrame({
        "League":   df["League"],
        "DateUTC":  df["DateUTC"],
        "HomeTeam": df["HomeTeam"],
        "AwayTeam": df["AwayTeam"],
    })

    # Αποτέλεσμα
    for j, lbl in enumerate(desired_res_labels):
        out[f"Prob_{lbl}"] = res_probs[:, j]
    # Over/Under
    for j, lbl in enumerate(desired_ou_labels):
        out[f"Prob_{lbl}"] = ou_probs[:, j]

    # Αληθινά labels (αν υπάρχουν) για σύγκριση
    out["TrueResult"]     = df.get("TrueResult",    pd.Series([np.nan]*len(df)))
    out["TrueOverUnder"]  = df.get("TrueOverUnder", pd.Series([np.nan]*len(df)))

    out.to_csv(DATA_OUT, index=False, encoding="utf-8")
    print(f"✅ Δημιουργήθηκε: {DATA_OUT} (rows={out.shape[0]}, cols={out.shape[1]})")

    # Μικρή σύνοψη
    used_res = Counter(df["TrainResult"])
    used_ou  = Counter(df["TrainOverUnder"])
    print("ℹ️  Κατανομή (TrainResult):", dict(used_res))
    print("ℹ️  Κατανομή (TrainOverUnder):", dict(used_ou))
    print("🎯 Ολοκληρώθηκε η διαδικασία με πραγματικά labels (και proxy όπου έλειπαν).")

# ----------- Συνάρτηση για χρήση στο app / μελλοντικά fixtures -----------
def predict_from_features(features_dict):
    """
    Χρήση:
        feats = {
          "HomeForm":..., "AwayForm":..., "HomeRank":..., "AwayRank":...,
          "HomeAvgScore":..., "AwayAvgScore":..., "HomeAvgConcede":..., "AwayAvgConcede":...,
          "HomeCleanSheetRate":..., "AwayCleanSheetRate":..., "HomeH2HWin":..., "AwayH2HWin":..., "H2HDraws":...
        }
        print(predict_from_features(feats))
    """
    ensure_dirs()
    # Φόρτωση μοντέλων
    if not (os.path.exists(MODEL_RES) and os.path.exists(MODEL_OU)):
        raise FileNotFoundError("Δεν βρέθηκαν τα εκπαιδευμένα μοντέλα. Τρέξε πρώτα: python predictor_real_v3.py")

    with open(MODEL_RES, "rb") as f:
        model_res = pickle.load(f)
    with open(MODEL_OU, "rb") as f:
        model_ou = pickle.load(f)

    X = np.array([[features_dict.get(f, 0.0) for f in FEATURES]], dtype=float)

    desired_res_labels = ["Home","Draw","Away"]
    desired_ou_labels  = ["Over","Under"]

    res_probs = proba_from_model(model_res, X, desired_res_labels)[0]
    ou_probs  = proba_from_model(model_ou,  X, desired_ou_labels)[0]

    return {
        "Result": dict(zip(desired_res_labels, np.round(res_probs, 3))),
        "OverUnder": dict(zip(desired_ou_labels,  np.round(ou_probs, 3)))
    }

# ---------------- Εκτέλεση ----------------
if __name__ == "__main__":
    main()
