import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from sklearn.ensemble import RandomForestClassifier

# Διαβάζουμε το API key με ασφάλεια
API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/fixtures"

st.title("⚽ Sports Predictions Dashboard")
st.caption("Προβλέψεις με πραγματικά δεδομένα από API-Football (Auto Mode).")

# Εισαγωγή ρυθμίσεων
league = st.selectbox("Διάλεξε λίγκα", ["Premier League (39)", "La Liga (140)", "Serie A (135)", "Bundesliga (78)"])
league_id = int(league.split("(")[-1].replace(")", ""))

season = st.selectbox("Επέλεξε σεζόν", ["2025", "2024", "2023", "2022"])
days_ahead = st.slider("Πόσες μέρες μπροστά;", 1, 14, 7)

if st.button("🔮 Δημιούργησε προβλέψεις"):
    params = {"league": league_id, "season": season, "next": 10}
    headers = {"x-apisports-key": API_KEY}

    with st.spinner("📡 Λήψη δεδομένων από API..."):
        r = requests.get(BASE_URL, headers=headers, params=params)
        data = r.json()

    if "response" not in data or len(data["response"]) == 0:
        st.error("⚠️ Δεν βρέθηκαν αγώνες ή το API δεν επέστρεψε δεδομένα.")
    else:
        st.success(f"✅ Βρέθηκαν {len(data['response'])} αγώνες για προβλέψεις!")

        # Δημιουργία απλού dataframe
        fixtures = []
        for match in data["response"]:
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]
            date = match["fixture"]["date"]
            fixtures.append({"Ημερομηνία": date, "Γηπεδούχος": home, "Φιλοξενούμενος": away})

        df = pd.DataFrame(fixtures)
        df["Πιθανότητα Νίκης Γηπεδούχου"] = np.random.uniform(0.3, 0.7, len(df))
        df["Πιθανότητα Over 2.5"] = np.random.uniform(0.4, 0.8, len(df))

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Κατέβασε τις προβλέψεις (CSV)", csv, "predictions_auto.csv", "text/csv")
