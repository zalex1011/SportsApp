#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_collector_v2.py
--------------------
Φέρνει δεδομένα ποδοσφαίρου από το API-Sports.

Δουλεύει σε 2 modes:
- MODE = "past"   → φέρνει τελειωμένους αγώνες (για training)
- MODE = "future" → φέρνει επερχόμενους αγώνες (για προβλέψεις)

Αποθηκεύει τα αποτελέσματα σε:
- data/matches_past.csv  ή
- data/matches_future.csv
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime

# ---------------- ΡΥΘΜΙΣΕΙΣ ----------------
API_KEY = "227cdea05de943bf04fcab225cec1457"

SEASON = "2022"           # σεζόν με διαθέσιμους αγώνες
MODE = "past"             # "past" ή "future"
FOOTBALL_LEAGUES = [145]  # μόνο μία λίγκα για δοκιμή

DAYS_BEFORE = 365
DAYS_AHEAD = 14
RATE_LIMIT_SLEEP = 0.75

BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# ---------------- ΣΥΝΑΡΤΗΣΕΙΣ ----------------
def _get(url, params=None):
    """Ασφαλές GET με καθυστέρηση για να μην χτυπήσουμε rate limits."""
    time.sleep(RATE_LIMIT_SLEEP)
    r = requests.get(url, headers=HEADERS, params=params or {})
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Invalid JSON: {r.text[:200]}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} -> {data}")
    if "response" not in data:
        raise RuntimeError(f"Unexpected payload: {data.keys()}")
    return data["response"]

def get_fixtures_range(league_id, season, finished_only=False):
    """Φέρνει fixtures για τη συγκεκριμένη λίγκα και σεζόν."""
    url = f"{BASE}/fixtures"
    params = {"league": int(league_id), "season": season}
    if finished_only:
        params["status"] = "FT"

    print(f"🔎 Requesting League={league_id}, Season={season}")
    resp = _get(url, params)
    print(f"📦 Results: {len(resp)} fixtures received")
    return resp

# ---------------- ΒΑΣΙΚΗ ΡΟΗ ----------------
def build_dataset(mode="past", season=SEASON, leagues=None):
    leagues = leagues or FOOTBALL_LEAGUES
    print("🎯 Εκτελώ build_dataset()")
    print(f"🔢 Λίγκες που θα τραβήξω: {leagues}")

    if mode == "past":
        finished_only = True
        out_path = os.path.join("data", "matches_past.csv")
    else:
        finished_only = False
        out_path = os.path.join("data", "matches_future.csv")

    rows = []
    os.makedirs("data", exist_ok=True)

    for league_id in leagues:
        try:
            fixtures = get_fixtures_range(league_id, season, finished_only=finished_only)
            if not fixtures:
                print(f"⚠️ Καμία απάντηση για λίγκα {league_id}")
                continue

            for f in fixtures:
                league_name = f["league"]["name"]
                fixture_id = f["fixture"]["id"]
                when_iso = f["fixture"]["date"]
                home = f["teams"]["home"]
                away = f["teams"]["away"]

                rows.append({
                    "LeagueID": league_id,
                    "League": league_name,
                    "FixtureID": fixture_id,
                    "DateUTC": when_iso,
                    "HomeTeam": home["name"],
                    "AwayTeam": away["name"],
                    "HomeWinner": home["winner"],
                    "AwayWinner": away["winner"],
                    "Status": f["fixture"]["status"]["short"],
                })
        except Exception as e:
            print(f"❌ Σφάλμα στη λίγκα {league_id}: {e}")
            rows.append({"Error": str(e), "LeagueID": league_id})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Αποθηκεύτηκε: {out_path} (rows={df.shape[0]}, cols={df.shape[1]})")

# ---------------- ΕΚΤΕΛΕΣΗ ----------------
if __name__ == "__main__":
    build_dataset(mode=MODE, season=SEASON, leagues=FOOTBALL_LEAGUES)
