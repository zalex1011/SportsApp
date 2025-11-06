#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_future_matches.py
---------------------
1) Τραβά επερχόμενα fixtures από API-Sports (για συγκεκριμένες λίγκες & ημέρες)
2) Υπολογίζει features για κάθε αγώνα (φόρμα, avg γκολ, clean sheets, H2H)
3) Φορτώνει τα εκπαιδευμένα μοντέλα (model_result_real.pkl, model_over_real.pkl)
4) Γράφει προβλέψεις στο data/predictions_future.csv

Απαιτήσεις:
- Έχεις ήδη τρέξει predictor_real_v3.py (ώστε να υπάρχουν τα models/)
- Υπάρχει ιστορικό dataset: data/matches_past_features_labeled.csv
"""

import os
import sys
import time
import math
import json
import datetime as dt
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import requests
import pickle

# ============== ΡΥΘΜΙΣΕΙΣ ΧΡΗΣΤΗ ==============
API_KEY = "227cdea05de943bf04fcab225cec1457"   # <- ΒΑΛΕ ΕΔΩ το API key σου
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_KEY}
RATE_DELAY = 0.7  # καθυστέρηση ανά κλήση για ασφάλεια

# Πόσες μέρες μπροστά να κοιτάξει για επερχόμενα ματς
DAYS_AHEAD = 7

# Λίγκες που θέλεις (IDs του API-Sports). Παράδειγμα: Premier 39, La Liga 140, Serie A 135, Ligue 1 61, Bundesliga 78, Greece 197, κλπ.
FOOTBALL_LEAGUES = [145]  # <- ΒΑΛΕ εδώ τις δικές σου. Μπορείς να προσθέσεις όσες θέλεις.

# Πόσα τελευταία παιχνίδια μετράμε για "φόρμα"
FORM_LAST_K = 5

# Paths
DATA_HIST_PATH = os.path.join("data", "matches_past_features_labeled.csv")
PRED_FUTURE_PATH = os.path.join("data", "predictions_future.csv")
MODEL_DIR = "models"
MODEL_RES = os.path.join(MODEL_DIR, "model_result_real.pkl")
MODEL_OU  = os.path.join(MODEL_DIR, "model_over_real.pkl")

# Features που περιμένουν τα μοντέλα
FEATURES = [
    "HomeForm","AwayForm","HomeRank","AwayRank",
    "HomeAvgScore","AwayAvgScore","HomeAvgConcede","AwayAvgConcede",
    "HomeCleanSheetRate","AwayCleanSheetRate","HomeH2HWin","AwayH2HWin","H2HDraws"
]

# ============== ΒΟΗΘΗΤΙΚΑ ==============
def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

def today_utc():
    return dt.datetime.utcnow().date()

def guess_season_for_date(date_obj):
    """
    Απλή υπόθεση για Ευρωπαϊκές λίγκες:
    Αν μήνας >= Αύγουστος → season = έτος
    Αλλιώς → season = έτος - 1
    π.χ. 2025-11-04 → season 2025
         2025-03-01 → season 2024
    """
    y = date_obj.year
    return y if date_obj.month >= 8 else y - 1

def daterange(days):
    start = today_utc()
    end = start + dt.timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def api_get(path, params):
    time.sleep(RATE_DELAY)
    r = requests.get(f"{BASE_URL}/{path}", headers=HEADERS, params=params, timeout=30)
    if r.status_code != 200:
        print(f"⚠️ HTTP {r.status_code} for {path} params={params}")
        return None
    try:
        return r.json().get("response", [])
    except Exception as e:
        print("❌ JSON parse error:", e)
        return None

def load_history():
    if not os.path.exists(DATA_HIST_PATH):
        raise FileNotFoundError(f"Δεν βρέθηκε {DATA_HIST_PATH}. Τρέξε πρώτα το pipeline ιστορικού.")
    df = pd.read_csv(DATA_HIST_PATH)
    df.columns = df.columns.str.strip()
    # Προαιρετική κανονικοποίηση
    basic = ["League","DateUTC","HomeTeam","AwayTeam","HomeGoals","AwayGoals","Result","OverUnderLabel"]
    for c in basic:
        if c not in df.columns:
            df[c] = np.nan
    # Μετατροπή DateUTC -> datetime
    try:
        df["DateUTC"] = pd.to_datetime(df["DateUTC"])
    except Exception:
        pass
    return df

def rolling_form_features(hist_df, team, upto_dt, k=FORM_LAST_K):
    """
    Υπολογίζει φόρμα/στατιστικά για 'team' μέχρι και πριν από την upto_dt.
    Επιστρέφει: avgScore, avgConcede, cleanSheetRate, formPointsAvg
    """
    df = hist_df[
        ((hist_df["HomeTeam"] == team) | (hist_df["AwayTeam"] == team))
        & (hist_df["DateUTC"] < pd.Timestamp(upto_dt) + pd.Timedelta(days=0))
        & (hist_df["HomeGoals"].notna()) & (hist_df["AwayGoals"].notna())
    ].sort_values("DateUTC", ascending=False).head(k)

    if df.empty:
        return 0.0, 0.0, 0.0, 0.0

    scored, conceded, clean_sheets, points = [], [], 0, []

    for _, r in df.iterrows():
        hg, ag = float(r["HomeGoals"]), float(r["AwayGoals"])
        if r["HomeTeam"] == team:
            scored.append(hg); conceded.append(ag)
            if ag == 0: clean_sheets += 1
            # points
            if hg > ag: points.append(3)
            elif hg == ag: points.append(1)
            else: points.append(0)
        else:
            scored.append(ag); conceded.append(hg)
            if hg == 0: clean_sheets += 1
            # points
            if ag > hg: points.append(3)
            elif ag == hg: points.append(1)
            else: points.append(0)

    n = len(scored)
    avg_scored = float(np.mean(scored)) if n else 0.0
    avg_concede = float(np.mean(conceded)) if n else 0.0
    cs_rate = clean_sheets / n if n else 0.0
    form_pts = float(np.mean(points)) if points else 0.0
    return avg_scored, avg_concede, cs_rate, form_pts

def h2h_features(hist_df, home, away, upto_dt, k=10):
    """
    Head-to-Head τελευταία k αναμετρήσεις μεταξύ home-away πριν την upto_dt.
    Επιστρέφει: homeWins, awayWins, draws
    """
    m = hist_df[
        (((hist_df["HomeTeam"] == home) & (hist_df["AwayTeam"] == away)) |
         ((hist_df["HomeTeam"] == away) & (hist_df["AwayTeam"] == home)))
        & (hist_df["DateUTC"] < pd.Timestamp(upto_dt))
        & (hist_df["HomeGoals"].notna()) & (hist_df["AwayGoals"].notna())
    ].sort_values("DateUTC", ascending=False).head(k)

    if m.empty:
        return 0, 0, 0

    hw = aw = dr = 0
    for _, r in m.iterrows():
        hg, ag = float(r["HomeGoals"]), float(r["AwayGoals"])
        if hg > ag:
            win_team = r["HomeTeam"]
        elif ag > hg:
            win_team = r["AwayTeam"]
        else:
            win_team = "DRAW"

        if win_team == "DRAW":
            dr += 1
        elif win_team == home:
            hw += 1
        else:
            aw += 1
    return hw, aw, dr

def simple_league_rank(hist_df, league, upto_dt, window_matches=20):
    """
    Χονδρικό ranking ομάδων ανά λίγκα, πριν από μια ημερομηνία:
    - Μαζεύει ~τελευταίους 'window_matches' αγώνες κάθε ομάδας στη λίγκα
    - Υπολογίζει μέσο όρο πόντων/αγώνα
    - Μικρό rank = καλύτερη ομάδα
    Επιστρέφει dict: team -> rank (1..N)
    """
    df = hist_df[(hist_df["League"] == league)
                 & (hist_df["DateUTC"] < pd.Timestamp(upto_dt))
                 & (hist_df["HomeGoals"].notna()) & (hist_df["AwayGoals"].notna())]

    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True).dropna())
    if len(teams) == 0:
        return {}

    pts_per_team = []
    for t in teams:
        sub = df[((df["HomeTeam"] == t) | (df["AwayTeam"] == t))].sort_values("DateUTC", ascending=False).head(window_matches)
        if sub.empty:
            pts_per_team.append((t, 0.0))
            continue
        pts = []
        for _, r in sub.iterrows():
            hg, ag = float(r["HomeGoals"]), float(r["AwayGoals"])
            if r["HomeTeam"] == t:
                if hg > ag: pts.append(3)
                elif hg == ag: pts.append(1)
                else: pts.append(0)
            else:
                if ag > hg: pts.append(3)
                elif ag == hg: pts.append(1)
                else: pts.append(0)
        avg_pts = float(np.mean(pts)) if pts else 0.0
        pts_per_team.append((t, avg_pts))

    # ταξινόμηση: μεγαλύτερο avg_pts -> καλύτερη θέση (rank 1)
    pts_sorted = sorted(pts_per_team, key=lambda x: x[1], reverse=True)
    ranks = {team: (i+1) for i, (team, _) in enumerate(pts_sorted)}
    return ranks

def load_models():
    if not os.path.exists(MODEL_RES) or not os.path.exists(MODEL_OU):
        raise FileNotFoundError("Δεν βρέθηκαν τα μοντέλα. Τρέξε πρώτα: python predictor_real_v3.py")
    with open(MODEL_RES, "rb") as f:
        model_res = pickle.load(f)
    with open(MODEL_OU, "rb") as f:
        model_ou = pickle.load(f)
    return model_res, model_ou

def proba_from_model(model, X, wanted_labels):
    """Βλέπε ίδιο helper στο predictor_real_v3.py – συμβατό με constant-model fallback."""
    if isinstance(model, tuple) and model[0] == "constant":
        const = model[1]
        out = np.zeros((X.shape[0], len(wanted_labels)), dtype=float)
        if const in wanted_labels:
            j = wanted_labels.index(const)
            out[:, j] = 1.0
        return out
    probs = model.predict_proba(X)
    model_labels = list(model.classes_)
    out = np.zeros((X.shape[0], len(wanted_labels)), dtype=float)
    for j, lbl in enumerate(wanted_labels):
        if lbl in model_labels:
            mj = model_labels.index(lbl)
            out[:, j] = probs[:, mj]
        else:
            out[:, j] = 0.0
    return out

# ============== ΛΗΨΗ ΕΠΕΡΧΟΜΕΝΩΝ FIXTURES ==============
def fetch_future_fixtures(leagues, days_ahead):
    start, end = daterange(days_ahead)
    today = today_utc()
    current_season = guess_season_for_date(today)

    rows = []
    for lg in leagues:
        params = {"league": lg, "season": current_season, "from": start, "to": end}
        resp = api_get("fixtures", params)
        if resp is None:
            continue
        for item in resp:
            status = item.get("fixture", {}).get("status", {}).get("short")
            if status not in ("NS", "TBD", "PST", "SUSP"):  # κρατάμε μόνο μη-παιγμένα
                continue
            fid = item.get("fixture", {}).get("id")
            dt_utc = item.get("fixture", {}).get("date")  # ISO
            try:
                dt_parsed = pd.to_datetime(dt_utc)
                date_utc = dt_parsed.strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_utc = dt_utc
            home = item.get("teams", {}).get("home", {}).get("name")
            away = item.get("teams", {}).get("away", {}).get("name")
            league_name = item.get("league", {}).get("name")
            rows.append({
                "LeagueID": lg,
                "League": league_name,
                "FixtureID": fid,
                "DateUTC": date_utc,
                "HomeTeam": home,
                "AwayTeam": away,
                "Status": status
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        # Μετατροπή DateUTC σε datetime για υπολογισμούς features
        try:
            df["DateUTC"] = pd.to_datetime(df["DateUTC"])
        except Exception:
            pass
    return df

# ============== FEATURE ENGINEERING ΓΙΑ ΜΕΛΛΟΝΤΙΚΑ ==============
def build_features_for_future(hist_df, fut_df):
    """
    Για κάθε μέλλον fixture στο fut_df, υπολογίζει τα FEATURES που περιμένει το μοντέλο.
    Επιστρέφει νέο DataFrame με τις στήλες των features + info αγώνα.
    """
    if fut_df.empty:
        return pd.DataFrame()

    records = []
    # Προ-υπολογισμός rank ανά λίγκα & ώρα αγώνα
    rank_cache = {}

    for _, row in fut_df.iterrows():
        league = row.get("League", "")
        dtt    = row.get("DateUTC")
        home   = row.get("HomeTeam", "")
        away   = row.get("AwayTeam", "")

        # rolling form features (ανά ομάδα)
        h_avg_scored, h_avg_conc, h_cs, h_form = rolling_form_features(hist_df, home, dtt, k=FORM_LAST_K)
        a_avg_scored, a_avg_conc, a_cs, a_form = rolling_form_features(hist_df, away, dtt, k=FORM_LAST_K)

        # simple league rank (χοντρικά)
        key = (league, pd.Timestamp(dtt).date())
        if key not in rank_cache:
            rank_cache[key] = simple_league_rank(hist_df, league, dtt, window_matches=20)
        ranks = rank_cache.get(key, {})
        home_rank = float(ranks.get(home, 0))
        away_rank = float(ranks.get(away, 0))

        # h2h
        h2h_hw, h2h_aw, h2h_dr = h2h_features(hist_df, home, away, dtt, k=10)

        rec = {
            "League": league,
            "DateUTC": dtt,
            "HomeTeam": home,
            "AwayTeam": away,
            "HomeForm": h_form, "AwayForm": a_form,
            "HomeRank": home_rank, "AwayRank": away_rank,
            "HomeAvgScore": h_avg_scored, "AwayAvgScore": a_avg_scored,
            "HomeAvgConcede": h_avg_conc, "AwayAvgConcede": a_avg_conc,
            "HomeCleanSheetRate": h_cs, "AwayCleanSheetRate": a_cs,
            "HomeH2HWin": h2h_hw, "AwayH2HWin": h2h_aw, "H2HDraws": h2h_dr,
        }
        records.append(rec)

    feats = pd.DataFrame(records)
    # Συμπλήρωση τυχόν κενών features με 0
    for c in FEATURES:
        if c not in feats.columns:
            feats[c] = 0.0
    feats[FEATURES] = feats[FEATURES].fillna(0.0)
    return feats

# ============== ΠΡΟΒΛΕΨΕΙΣ ==============
def run_predictions(future_feats, model_res, model_ou):
    if future_feats.empty:
        return pd.DataFrame()

    X = future_feats[FEATURES].values
    desired_res = ["Home","Draw","Away"]
    desired_ou  = ["Over","Under"]

    res_probs = proba_from_model(model_res, X, desired_res)
    ou_probs  = proba_from_model(model_ou,  X, desired_ou)

    out = pd.DataFrame({
        "League": future_feats["League"].astype(str),
        "DateUTC": future_feats["DateUTC"].astype(str),
        "HomeTeam": future_feats["HomeTeam"].astype(str),
        "AwayTeam": future_feats["AwayTeam"].astype(str),
        "Prob_Home": res_probs[:, 0],
        "Prob_Draw": res_probs[:, 1],
        "Prob_Away": res_probs[:, 2],
        "Prob_Over": ou_probs[:, 0],
        "Prob_Under": ou_probs[:, 1],
    })
    return out

# ============== MAIN ==============
def main():
    ensure_dirs()

    if API_KEY == "227cdea05de943bf04fcab225cec1457":
        print("❌ Βάλε το API key σου στη μεταβλητή API_KEY στην αρχή του αρχείου.")
        sys.exit(1)

    print(f"🔎 Φόρτωση ιστορικού από: {DATA_HIST_PATH}")
    hist = load_history()

    print(f"🌤️ Λήψη επερχόμενων fixtures για {DAYS_AHEAD} ημέρες, λίγκες: {FOOTBALL_LEAGUES}")
    fut = fetch_future_fixtures(FOOTBALL_LEAGUES, DAYS_AHEAD)
    if fut is None or fut.empty:
        print("⚠️ Δεν βρέθηκαν επερχόμενοι αγώνες στο διάστημα αυτό.")
        return

    print(f"🧮 Υπολογισμός features για {len(fut)} αγώνες ...")
    fut_feats = build_features_for_future(hist, fut)

    print("📦 Φόρτωση εκπαιδευμένων μοντέλων ...")
    model_res, model_ou = load_models()

    print("🤖 Υπολογισμός πιθανοτήτων ...")
    preds = run_predictions(fut_feats, model_res, model_ou)

    if preds.empty:
        print("⚠️ Δεν προέκυψαν προβλέψεις.")
        return

    preds.sort_values(["League","DateUTC"], inplace=True)
    preds.to_csv(PRED_FUTURE_PATH, index=False, encoding="utf-8")
    print(f"✅ Αποθηκεύτηκαν προβλέψεις μελλοντικών αγώνων: {PRED_FUTURE_PATH} (rows={preds.shape[0]})")
    print("🎯 Τέλος διαδικασίας.")

if __name__ == "__main__":
    main()
