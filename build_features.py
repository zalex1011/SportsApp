#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features.py
-----------------
Δημιουργεί features από το data/matches_past.csv για χρήση στο predictor_v2.py.
Σχεδιασμένο για 1 λίγκα (π.χ. FOOTBALL_LEAGUES = [145]).
Μειώνει τις κλήσεις στο API επαναχρησιμοποιώντας απαντήσεις (cache per team).
Χρησιμοποιεί τελευταίους 5 αγώνες για φόρμα.

Χρήση:
    python build_features.py

Απαιτεί: αρχείο data/matches_past.csv
Παράγει: data/matches_past_features.csv
"""

import os, time, json
import requests
import pandas as pd
from collections import defaultdict

# ---------------- ΡΥΘΜΙΣΕΙΣ ----------------
API_KEY = "227cdea05de943bf04fcab225cec1457"
BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
RATE_LIMIT_SLEEP = 0.8  # ασφαλής καθυστέρηση ανά αίτημα

SEASON = "2022"  # σεζόν που χρησιμοποιήθηκε στο matches_past.csv
FORM_LAST_N = 5  # τελευταίοι 5 αγώνες για φόρμα

# ---------------- ΒΟΗΘΗΤΙΚΕΣ ----------------
def safe_get(url, params=None):
    time.sleep(RATE_LIMIT_SLEEP)
    r = requests.get(url, headers=HEADERS, params=params or {})
    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from {url}: {e} :: response_text={r.text[:200]}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url} -> {data}")
    return data

def ensure_data_folder():
    os.makedirs("data", exist_ok=True)

# ---------------- ΚΥΡΙΟ ----------------
def load_matches(path="data/matches_past.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Input matches_past.csv is empty")
    return df

def find_team_id_by_name(name, league_id, season=SEASON):
    """Χρησιμοποιεί /teams?search=name&league=... ώστε να βρει team id. Επιστρέφει int id ή None"""
    url = f"{BASE}/teams"
    params = {"search": name, "league": league_id, "season": season}
    data = safe_get(url, params=params)
    resp = data.get("response") if isinstance(data, dict) else data
    if isinstance(resp, list) and len(resp) > 0:
        # προσπάθησε να επιλέξεις το καλύτερο match: exact name if possible
        for item in resp:
            team = item.get("team") or {}
            if team.get("name","").lower() == name.lower():
                return team.get("id")
        # else take first
        first = resp[0].get("team",{})
        return first.get("id")
    return None

def get_team_statistics(team_id, league_id, season=SEASON):
    url = f"{BASE}/teams/statistics"
    params = {"league": league_id, "season": season, "team": team_id}
    data = safe_get(url, params=params)
    return data.get("response")

def get_last_matches(team_id, last_n=FORM_LAST_N, season=SEASON):
    url = f"{BASE}/fixtures"
    params = {"team": team_id, "season": season, "last": last_n, "status": "FT"}
    data = safe_get(url, params=params)
    return data.get("response")

# Compute form index from last matches response
def compute_form_from_fixtures(fixtures, team_id):
    if not fixtures:
        return None
    points = 0.0
    cnt = 0
    for m in fixtures:
        cnt += 1
        goals = m.get("goals") or {}
        gh = goals.get("home", 0)
        ga = goals.get("away", 0)
        hid = m.get("teams",{}).get("home",{}).get("id")
        aid = m.get("teams",{}).get("away",{}).get("id")
        if hid == team_id:
            if gh > ga: points += 1.0
            elif gh == ga: points += 0.5
        elif aid == team_id:
            if ga > gh: points += 1.0
            elif gh == ga: points += 0.5
    return round(points / max(cnt,1), 3)

def avg_goals_from_stats(stats_resp, side="for"):
    try:
        goals = stats_resp["goals"][side]["total"]["total"]
        played = stats_resp["fixtures"]["played"]["total"]
        return round(goals / max(played,1), 3)
    except Exception:
        return None

def clean_sheet_rate_from_stats(stats_resp):
    try:
        cs = stats_resp["clean_sheet"]["total"]
        played = stats_resp["fixtures"]["played"]["total"]
        return round(cs / max(played,1), 3)
    except Exception:
        return None

# ---------------- ΕΡΓΑΛΕΙΑ CACHE ----------------
class APICache:
    def __init__(self):
        self.team_id = {}          # name -> id
        self.team_stats = {}       # id -> stats payload
        self.team_last = {}        # id -> last matches list

# ---------------- MAIN FEATURE BUILDER ----------------
def build_features(league_id_list):
    ensure_data_folder()
    matches = load_matches("data/matches_past.csv")
    # Filter to chosen leagues only (safety)
    matches = matches[matches["LeagueID"].isin(league_id_list)]
    # Normalize team names (strip)
    matches["HomeTeam"] = matches["HomeTeam"].astype(str).str.strip()
    matches["AwayTeam"] = matches["AwayTeam"].astype(str).str.strip()

    # unique teams
    teams = sorted(pd.unique(matches[["HomeTeam","AwayTeam"]].values.ravel()))
    print(f"Found {len(teams)} unique teams in CSV.")

    cache = APICache()
    features_rows = []

    # Resolve team ids and collect per-team stats (minimize calls)
    for t in teams:
        try:
            tid = find_team_id_by_name(t, league_id_list[0], SEASON)
            if tid is None:
                print(f"⚠️ Δεν βρέθηκε team id για '{t}'. Θα χρησιμοποιήσουμε None.")
            cache.team_id[t] = tid
        except Exception as e:
            print(f"❌ Σφάλμα fetching id for {t}: {e}")
            cache.team_id[t] = None

    # For each team fetch statistics and last matches (if id available)
    for t, tid in cache.team_id.items():
        if tid is None:
            cache.team_stats[tid] = None
            cache.team_last[tid] = []
            continue
        try:
            stats = get_team_statistics(tid, league_id_list[0], SEASON)
            cache.team_stats[tid] = stats
        except Exception as e:
            print(f"⚠️ Team stats error for {t} ({tid}): {e}")
            cache.team_stats[tid] = None
        try:
            last = get_last_matches(tid, FORM_LAST_N, SEASON)
            cache.team_last[tid] = last or []
        except Exception as e:
            print(f"⚠️ Last matches error for {t} ({tid}): {e}")
            cache.team_last[tid] = []

    # Build features per fixture using cached info + local H2H from matches df
    # Precompute H2H counts from matches CSV (use names)
    h2h_counts = defaultdict(lambda: {"home_w":0, "away_w":0, "draw":0})
    for _, row in matches.iterrows():
        h = row["HomeTeam"]; a = row["AwayTeam"]
        hw = row.get("HomeWinner")
        aw = row.get("AwayWinner")
        # Sometimes CSV stores booleans or strings; normalize
        if str(hw).lower() in ("true","1","1.0"):
            h2h_counts[(h,a)]["home_w"] += 1
        elif str(aw).lower() in ("true","1","1.0"):
            h2h_counts[(h,a)]["away_w"] += 1
        else:
            h2h_counts[(h,a)]["draw"] += 1

    # Iterate fixtures to compute features
    for _, row in matches.iterrows():
        league = row.get("League"); league_id = row.get("LeagueID")
        when = row.get("DateUTC")
        home = row.get("HomeTeam"); away = row.get("AwayTeam")
        hid = cache.team_id.get(home)
        aid = cache.team_id.get(away)

        # team stats payloads
        h_stats = cache.team_stats.get(hid)
        a_stats = cache.team_stats.get(aid)

        # compute features
        home_form = compute_form_from_fixtures(cache.team_last.get(hid, []), hid) if hid else None
        away_form = compute_form_from_fixtures(cache.team_last.get(aid, []), aid) if aid else None

        home_avg_for = avg_goals_from_stats(h_stats, "for") if h_stats else None
        home_avg_against = avg_goals_from_stats(h_stats, "against") if h_stats else None
        away_avg_for = avg_goals_from_stats(a_stats, "for") if a_stats else None
        away_avg_against = avg_goals_from_stats(a_stats, "against") if a_stats else None

        home_cs = clean_sheet_rate_from_stats(h_stats) if h_stats else None
        away_cs = clean_sheet_rate_from_stats(a_stats) if a_stats else None

        # For rank: try to extract from stats payload (some payloads include league position)
        def extract_rank(stats_payload, team_id):
            try:
                league = stats_payload.get("league",{})
                if isinstance(league, dict):
                    # some responses include standings-like position under 'standings' or 'position'
                    pos = stats_payload.get("league",{}).get("standings")
                    # fallback: not all endpoints include rank -> return None
                return None
            except Exception:
                return None

        home_rank = None
        away_rank = None

        # H2H from precomputed counts (use direct key if available)
        h2h = h2h_counts.get((home,away), {"home_w":0,"away_w":0,"draw":0})
        h2h_home_wins = h2h["home_w"]
        h2h_away_wins = h2h["away_w"]
        h2h_draws = h2h["draw"]

        features_rows.append({
            "LeagueID": league_id,
            "League": league,
            "DateUTC": when,
            "HomeTeam": home,
            "AwayTeam": away,
            "HomeForm": home_form,
            "AwayForm": away_form,
            "HomeRank": home_rank,
            "AwayRank": away_rank,
            "HomeAvgScore": home_avg_for,
            "AwayAvgScore": away_avg_for,
            "HomeAvgConcede": home_avg_against,
            "AwayAvgConcede": away_avg_against,
            "HomeCleanSheetRate": home_cs,
            "AwayCleanSheetRate": away_cs,
            "HomeH2HWin": h2h_home_wins,
            "AwayH2HWin": h2h_away_wins,
            "H2HDraws": h2h_draws
        })

    # Save features
    out = os.path.join("data","matches_past_features.csv")
    df_out = pd.DataFrame(features_rows)
    df_out.to_csv(out, index=False, encoding="utf-8")
    print(f"✅ Features saved: {out} (rows={df_out.shape[0]}, cols={df_out.shape[1]})")

# ---------------- ΕΚΤΕΛΕΣΗ ----------------
if __name__ == "__main__":
    # Use single league 145 as requested
    build_features([145])
