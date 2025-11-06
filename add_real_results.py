#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_real_results.py
-------------------
Ενημερώνει το matches_past_features.csv με πραγματικά αποτελέσματα και Over/Under.
Χρησιμοποιεί το fixtures endpoint για κάθε FixtureID και αποθηκεύει νέο αρχείο:
matches_past_features_labeled.csv
"""

import os
import time
import requests
import pandas as pd

# ---------------- ΡΥΘΜΙΣΕΙΣ ----------------
API_KEY = "227cdea05de943bf04fcab225cec1457"   # <--- Βάλε εδώ το δικό σου key
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
RATE_DELAY = 0.8  # δευτερόλεπτα ανά αίτημα

# ---------------- ΣΥΝΑΡΤΗΣΕΙΣ ----------------
def get_fixture_details(fixture_id):
    """Παίρνει τα goals για συγκεκριμένο FixtureID"""
    url = f"{BASE_URL}/fixtures"
    params = {"id": fixture_id}
    time.sleep(RATE_DELAY)
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        print(f"⚠️ Fixture {fixture_id} HTTP {resp.status_code}")
        return None
    try:
        data = resp.json().get("response", [])
        if not data:
            return None
        fixture = data[0]
        goals = fixture.get("goals", {})
        status = fixture.get("fixture", {}).get("status", {}).get("short")
        return {
            "FixtureID": fixture_id,
            "HomeGoals": goals.get("home"),
            "AwayGoals": goals.get("away"),
            "Status": status
        }
    except Exception as e:
        print(f"❌ Σφάλμα στο Fixture {fixture_id}: {e}")
        return None


def determine_result(home_goals, away_goals):
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "Home"
    elif away_goals > home_goals:
        return "Away"
    else:
        return "Draw"


def determine_over_under(home_goals, away_goals, threshold=2.5):
    if home_goals is None or away_goals is None:
        return None
    total = home_goals + away_goals
    return "Over" if total > threshold else "Under"


# ---------------- ΚΥΡΙΑ ΡΟΗ ----------------
def main():
    path_matches = os.path.join("data", "matches_past.csv")
    path_features = os.path.join("data", "matches_past_features.csv")
    out_path = os.path.join("data", "matches_past_features_labeled.csv")

    if not os.path.exists(path_matches):
        print(f"❌ Δεν βρέθηκε {path_matches}")
        return
    if not os.path.exists(path_features):
        print(f"❌ Δεν βρέθηκε {path_features}")
        return

    matches = pd.read_csv(path_matches)
    features = pd.read_csv(path_features)
    features.columns = features.columns.str.strip()  # καθάρισε spaces

    results = []

    for i, row in matches.iterrows():
        fid = row.get("FixtureID")
        if pd.isna(fid):
            continue
        data = get_fixture_details(int(fid))
        if not data or data.get("Status") != "FT":
            continue

        home_goals = data["HomeGoals"]
        away_goals = data["AwayGoals"]
        result = determine_result(home_goals, away_goals)
        ou = determine_over_under(home_goals, away_goals)

        # --- εδώ προσθέτουμε και τις ομάδες / ημερομηνία για σωστό merge ---
        results.append({
            "FixtureID": fid,
            "League": row.get("League"),
            "DateUTC": row.get("DateUTC"),
            "HomeTeam": row.get("HomeTeam"),
            "AwayTeam": row.get("AwayTeam"),
            "HomeGoals": home_goals,
            "AwayGoals": away_goals,
            "Result": result,
            "OverUnderLabel": ou
        })

        print(f"✅ Fixture {fid}: {home_goals}-{away_goals} ({result}, {ou})")

    if not results:
        print("⚠️ Δεν βρέθηκαν αποτελέσματα για ενημέρωση.")
        return

    df_results = pd.DataFrame(results)

    # συγχώνευση με βάση HomeTeam, AwayTeam και DateUTC
    merged = pd.merge(
        features,
        df_results,
        on=["HomeTeam", "AwayTeam", "DateUTC"],
        how="left"
    )

    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Αποθηκεύτηκε: {out_path} (rows={merged.shape[0]}, cols={merged.shape[1]})")
    print("🎯 Προστέθηκαν πραγματικά αποτελέσματα & Over/Under.")


if __name__ == "__main__":
    main()
