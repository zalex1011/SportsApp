#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Streamlit dashboard for predictions
Run: streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

PRED_PATH = os.path.join("data", "predictions.csv")
FEAT_PATH = os.path.join("data", "matches_past_features.csv")

st.set_page_config(page_title="Sports Predictions Dashboard", layout="wide")

@st.cache_data
def load_data():
    if not os.path.exists(PRED_PATH):
        st.error(f"Δεν βρέθηκε το αρχείο προβλέψεων: {PRED_PATH}")
        return None, None
    pred = pd.read_csv(PRED_PATH)
    feats = pd.read_csv(FEAT_PATH) if os.path.exists(FEAT_PATH) else None
    # Normalize column names
    for col in ["League","DateUTC","HomeTeam","AwayTeam"]:
        if col in pred.columns:
            pred[col] = pred[col].astype(str)
    return pred, feats

def find_prob_columns(df):
    """Detect probability columns. Return (result_cols, ou_cols)."""
    prob_cols = [c for c in df.columns if c.startswith("Prob_")]
    # Over/Under labels (case-insensitive match)
    ou_cols = [c for c in prob_cols if c.lower() in ("prob_over","prob_under")]
    result_cols = [c for c in prob_cols if c not in ou_cols]
    return result_cols, ou_cols

def compute_best_columns(df, result_cols, ou_cols):
    df = df.copy()
    # Best Result
    if result_cols:
        df["BestResultProb"] = df[result_cols].max(axis=1)
        df["BestResult"] = df[result_cols].idxmax(axis=1).str.replace("Prob_","", regex=False)
    else:
        df["BestResultProb"] = np.nan
        df["BestResult"] = ""
    # Best OU
    if ou_cols:
        df["BestOUProb"] = df[ou_cols].max(axis=1)
        df["BestOU"] = df[ou_cols].idxmax(axis=1).str.replace("Prob_","", regex=False)
    else:
        df["BestOUProb"] = np.nan
        df["BestOU"] = ""
    return df

def filter_df(df, league_sel, team_sel, min_prob, consider_over_under=False):
    df_f = df.copy()
    if league_sel and league_sel != "Όλες οι λίγκες":
        df_f = df_f[df_f["League"] == league_sel]
    if team_sel and team_sel != "Όλες οι ομάδες":
        df_f = df_f[(df_f["HomeTeam"] == team_sel) | (df_f["AwayTeam"] == team_sel)]
    # Probability filter
    if consider_over_under:
        prob_col = "BestOUProb"
    else:
        prob_col = "BestResultProb"
    if prob_col in df_f.columns and not np.isnan(min_prob):
        df_f = df_f[df_f[prob_col] >= (min_prob/100.0)]
    return df_f

def plot_top_bars(df, value_col, label_col, title, subtitle, topn=10):
    if df.empty or value_col not in df.columns:
        st.info("Δεν υπάρχουν δεδομένα για γράφημα.")
        return
    dfx = df.sort_values(value_col, ascending=False).head(topn)
    labels = (dfx["HomeTeam"] + " vs " + dfx["AwayTeam"] + " (" + dfx[label_col] + ")").tolist()
    vals = dfx[value_col].values

    fig = plt.figure(figsize=(10, 6))
    plt.barh(range(len(vals))[::-1], vals[::-1])
    plt.yticks(range(len(labels))[::-1], labels[::-1])
    plt.xlabel("Probability")
    plt.title(title)
    st.caption(subtitle)
    st.pyplot(fig)

# ---------------- UI ----------------
st.title("🏟️ Sports Predictions Dashboard")
st.markdown(
    "Πίνακας προβλέψεων με **Home / Draw / Away** και **Over / Under 2.5**. "
    "Χρησιμοποίησε τα φίλτρα για να εντοπίσεις τις πιο ισχυρές επιλογές."
)

pred, feats = load_data()
if pred is None:
    st.stop()

result_cols, ou_cols = find_prob_columns(pred)
pred = compute_best_columns(pred, result_cols, ou_cols)

# Sidebar filters
st.sidebar.header("⚙️ Φίλτρα")
leagues = ["Όλες οι λίγκες"] + sorted(pred["League"].dropna().unique().tolist())
league_sel = st.sidebar.selectbox("Λίγκα", leagues)

teams = sorted(set(pred["HomeTeam"]).union(set(pred["AwayTeam"])))
teams = ["Όλες οι ομάδες"] + teams
team_sel = st.sidebar.selectbox("Ομάδα", teams)

min_prob = st.sidebar.slider("Ελάχιστη πιθανότητα (%)", 0, 100, 60)
view_mode = st.sidebar.radio("Τύπος πιθανότητας για το φίλτρο/πίνακα",
                             ["Αποτέλεσμα", "Over/Under"], index=0)
consider_over_under = (view_mode == "Over/Under")

# Filtered dataframe
df_view = filter_df(pred, league_sel, team_sel, min_prob, consider_over_under)
st.subheader("📋 Προβλέψεις (φιλτραρισμένες)")

# Build display columns
display_cols = ["League","DateUTC","HomeTeam","AwayTeam"]
display_cols += result_cols + ou_cols
extra_cols = ["BestResult","BestResultProb","BestOU","BestOUProb"]
display_cols += [c for c in extra_cols if c in df_view.columns]
show_df = df_view[display_cols].sort_values(
    "BestOUProb" if consider_over_under else "BestResultProb",
    ascending=False
)
st.dataframe(show_df.reset_index(drop=True), use_container_width=True)

# Charts
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Top Result Predictions")
    plot_top_bars(
        df_view,
        value_col="BestResultProb",
        label_col="BestResult",
        title="Top Result (Home / Draw / Away)",
        subtitle="Οι 10 πιο ισχυρές προβλέψεις αποτελέσματος",
        topn=10
    )

with col2:
    st.subheader("📈 Top Over/Under Predictions")
    plot_top_bars(
        df_view,
        value_col="BestOUProb",
        label_col="BestOU",
        title="Top Over/Under 2.5",
        subtitle="Οι 10 πιο ισχυρές προβλέψεις Over/Under",
        topn=10
    )

st.markdown("---")
st.caption("Δεδομένα από data/predictions.csv — ενημέρωσε τα τρέχοντας ξανά το predictor_v2.py.")
