# data_collector_real.py
# Κατεβάζει ΠΡΑΓΜΑΤΙΚΑ αποτελέσματα 3 σεζόν για όλες τις λίγκες σου

import os
import requests
import pandas as pd
from datetime import datetime
from time import sleep

# ========= ΡΥΘΜΙΣΕΙΣ ΧΡΗΣΤΗ =========
API_KEY = "ΒΑΛΕ_ΕΔΩ_ΤΟ_ΠΡΑΓΜΑΤΙΚΟ_API_KEY"   # <-- άλλαξέ το

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Λίγκες που θες (ίδιες με το app)
FOOTBALL_LEAGUES = [
    39,   # Premier League
    197,  # Super League Greece
    140,  # La Liga
    135,  # Serie A
    145,  # Challenger Pro League
    144,  # Pro League Belgium (A)
    203,  # Super Lig Turkey
    88,   # Eredivisie
    94,   # Primeira Liga
    61,   # Ligue 1
    78,   # Bundesliga
    494,  # Super League 2 Greece
]

# Τελευταίες 3 σεζόν (μπορείς να τις αλλάξεις)
SEASONS = [2022, 2023, 2024]

OUT_PATH = os.path.join("data", "matches_past_real.csv")


def _get(url, params):
    """Κλήση στο API με βασικό error handling + μικρό delay."""
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    return data


def fetch_fixtures_for_league_season(league_id, season):
    """Φέρνει ΟΛΑ τα τελειωμένα ματς για μια λίγκα & σεζόν."""
    print(f"  ➜ League {league_id}, Season {season} ...", flush=True)

    url = f"{BASE_URL}/fixtures"
    page = 1
    all_rows = []

    while True:
        params = {
            "league": league_id,
            "season": season,
            "status": "FT",   # finished matches μόνο
            "page": page,
        }
        data = _get(url, params)

        response = data.get("response", [])
        if not response:
            break

        for fx in response:
            fix = fx["fixture"]
            league = fx["league"]
            teams = fx["teams"]
            goals = fx["goals"]

            row = {
                "LeagueID": league["id"],
                "League": league["name"],
                "Season": league["season"],
                "DateUTC": fix["date"],
                "HomeTeam": teams["home"]["name"],
                "AwayTeam": teams["away"]["name"],
                "HomeGoals": goals["home"],
                "AwayGoals": goals["away"],
            }
            all_rows.append(row)

        paging = data.get("paging", {})
        current = paging.get("current", 1)
        total = paging.get("total", 1)
        if current >= total:
            break
        page += 1
        sleep(0.25)  # μικρό delay για να είμαστε gentle στο API

    return all_rows


def main():
    os.makedirs("data", exist_ok=True)
    all_data = []

    print("🚀 Ξεκινάω κατέβασμα ιστορικών fixtures (3 σεζόν)...\n")
    for league_id in FOOTBALL_LEAGUES:
        for season in SEASONS:
            try:
                rows = fetch_fixtures_for_league_season(league_id, season)
                print(f"     ➕ Βρέθηκαν {len(rows)} αγώνες.")
                all_data.extend(rows)
            except Exception as e:
                print(f"❌ Σφάλμα σε league={league_id}, season={season}: {e}")

    if not all_data:
        print("❌ Δεν βρέθηκαν δεδομένα. Έλεγξε το API_KEY ή τα όρια του πλάνου.")
        return

    df = pd.DataFrame(all_data)
    # καθάρισμα ημερομηνίας
    df["DateUTC"] = pd.to_datetime(df["DateUTC"])
    df = df.sort_values(["LeagueID", "Season", "DateUTC"]).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Αποθηκεύτηκαν {len(df)} αγώνες στο {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
