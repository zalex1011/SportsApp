import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle
import requests
import io
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": API_KEY}

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    # Διαβάζουμε τα base64 strings από τα .txt αρχεία
    with open("models/model_result_real.txt", "r") as f:
        result_bytes = base64.b64decode(f.read())
    with open("models/model_over_real.txt", "r") as f:
        over_bytes = base64.b64decode(f.read())

    model_result = pickle.load(io.BytesIO(result_bytes))
    model_over = pickle.load(io.BytesIO(over_bytes))
    return model_result, model_over

model_result, model_over = load_models()

# ---------------- STREAMLIT UI ----------------
st.title("⚽ Sports Predictions Dashboard (AI Mode)")
st.caption("Προβλέψεις για επερχόμενους αγώνες με Machine Learning μοντέλα και API-Football δεδομένα.")

league = st.selectbox("Διάλεξε λίγκα", [
    "Premier League (39)",
    "La Liga (140)",
    "Serie A (135)",
    "Bundesliga (78)",
    "Super League Greece (197)"
])
league_id = int(league.split("(")[-1].replace(")", ""))

season = st.selectbox("Επέλεξε σεζόν", ["2025", "2024", "2023", "2022"])
days_ahead = st.slider("Πόσες μέρες μπροστά;", 1, 14, 7)

# ---------------- FETCH FIXTURES ----------------
if st.button("🔮 Δημιούργησε προβλέψεις"):
    today = datetime.utcnow().date()
    future_date = today + timedelta(days=days_ahead)

    params = {
        "league": league_id,
        "season": season,
        "from": str(today),
        "to": str(future_date)
    }

    with st.spinner("📡 Ανάκτηση αγώνων από API..."):
        response = requests.get(BASE_URL, headers=HEADERS, params=params)
        data = response.json()

    if "response" not in data or len(data["response"]) == 0:
        st.error("⚠️ Δεν βρέθηκαν επερχόμενοι αγώνες.")
    else:
        st.success(f"✅ Βρέθηκαν {len(data['response'])} αγώνες.")
        fixtures = []

        for m in data["response"]:
            fixture = m["fixture"]
            teams = m["teams"]

            home = teams["home"]["name"]
            away = teams["away"]["name"]
            date = fixture["date"]

            # Δημιουργούμε dummy features για το παράδειγμα
            # (θα συνδεθούν με τα πραγματικά features στα επόμενα βήματα)
            X = np.random.rand(1, model_result.n_features_in_)
            pred_result = model_result.predict_proba(X)[0]
            pred_over = model_over.predict_proba(X)[0]

            fixtures.append({
                "Ημερομηνία": date[:10],
                "Γηπεδούχος": home,
                "Φιλοξενούμενος": away,
                "Πιθανότητα Home": round(pred_result[0]*100, 1),
                "Πιθανότητα Draw": round(pred_result[1]*100, 1),
                "Πιθανότητα Away": round(pred_result[2]*100, 1),
                "Πιθανότητα Over 2.5": round(pred_over[1]*100, 1),
                "Πιθανότητα Under 2.5": round(pred_over[0]*100, 1)
            })

        df = pd.DataFrame(fixtures)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Κατέβασε τις προβλέψεις (CSV)", csv, "predictions_ai.csv", "text/csv")
