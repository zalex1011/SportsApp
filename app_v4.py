import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle
import requests
import io
from datetime import datetime, timedelta

# =============== SETTINGS ===============
st.set_page_config(page_title="Sports Predictions - Card View", layout="wide")

API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": API_KEY}

LEAGUES = [
    ("Premier League (England)", 39),
    ("Super League Greece", 197),
    ("La Liga (Spain)", 140),
    ("Serie A (Italy)", 135),
    ("Challenger Pro League (Belgium - B Division)", 145),
    ("Pro League (Belgium - A Division)", 144),
    ("Super Lig (Turkey)", 203),
    ("Eredivisie (Netherlands)", 88),
    ("Primeira Liga (Portugal)", 94),
    ("Ligue 1 (France)", 61),
    ("Bundesliga (Germany)", 78),
    ("Super League 2 (Greece - B Division)", 494)
]


SEASONS = ["2025", "2024", "2023", "2022"]


# =============== LOAD MODELS ===============
@st.cache_resource
def load_models():
    try:
        with open("models/model_result_real.txt", "r") as f:
            result_bytes = base64.b64decode(f.read())
        with open("models/model_over_real.txt", "r") as f:
            over_bytes = base64.b64decode(f.read())
    except FileNotFoundError as e:
        st.error("Λείπουν τα base64 αρχεία μοντέλων μέσα στο φάκελο `models/` "
                 "(model_result_real.txt, model_over_real.txt).")
        raise e
    model_result = pickle.load(io.BytesIO(result_bytes))
    model_over = pickle.load(io.BytesIO(over_bytes))
    return model_result, model_over

model_result, model_over = load_models()


# =============== HELPERS ===============
def fetch_fixtures(league_id: int, season: str, days_ahead: int):
    today = datetime.utcnow().date()
    future_date = today + timedelta(days=days_ahead)
    params = {"league": league_id, "season": season, "from": str(today), "to": str(future_date)}
    r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
    data = r.json()
    return data.get("response", [])


def softmax_like_topgap(probs: np.ndarray) -> float:
    """
    Φερεγγυότητα για RESULT: όσο πιο πολύ ξεχωρίζει η μεγαλύτερη πιθανότητα, τόσο μεγαλύτερη.
    Επιστρέφει 0..1.
    """
    if probs.ndim == 1:
        p = probs
    else:
        p = probs[0]
    top = float(np.max(p))
    second = float(np.sort(p)[-2]) if len(p) >= 2 else 0.0
    gap = max(0.0, top - second)  # 0..1
    # ελαφρά "έμφαση" στο gap
    confidence = min(1.0, 0.75 * gap + 0.25 * (top - 1/len(p)))
    return float(np.clip(confidence, 0.0, 1.0))


def ou_confidence(probs_under_over: np.ndarray) -> float:
    """
    Φερεγγυότητα για Over/Under (δυαδικό): 2*|p-0.5| -> 0..1
    """
    p_over = float(probs_under_over[1])
    return float(np.clip(2.0 * abs(p_over - 0.5), 0.0, 1.0))


def color_for_prob(pct: float) -> str:
    if pct >= 70:
        return "#16a34a"  # green-600
    if pct >= 55:
        return "#f59e0b"  # amber-500
    return "#ef4444"      # red-500


def suggestion_from_result(probs: np.ndarray) -> str:
    labels = ["Home", "Draw", "Away"]
    idx = int(np.argmax(probs))
    return labels[idx]


def suggestion_from_ou(probs: np.ndarray) -> str:
    return "Over 2.5" if probs[1] >= probs[0] else "Under 2.5"


def make_dummy_features(n_features: int) -> np.ndarray:
    # Μέχρι να συνδέσουμε full features από API, κρατάμε dummy X με seed ανά παιχνίδι για σταθερότητα UI.
    X = np.random.rand(1, n_features)
    return X


# =============== UI HEADER ===============
st.title("🎴 Sports Predictions — Card View (AI Mode)")
st.caption("Προβλέψεις αποτελέσματος & Over/Under 2.5 με Machine Learning μοντέλα. "
           "Το UI δείχνει κάρτες ανά αγώνα με ποσοστά και φερεγγυότητα.")

colA, colB, colC = st.columns([2,1,1])
with colA:
    league_label = st.selectbox("Λίγκα", [f"{name} ({lid})" for name, lid in LEAGUES])
    league_id = int(league_label.split("(")[-1].rstrip(")"))
with colB:
    season = st.selectbox("Σεζόν", SEASONS, index=0)
with colC:
    days_ahead = st.slider("Μέρες μπροστά", 1, 14, 7)

go = st.button("🔮 Δημιούργησε προβλέψεις", use_container_width=True)

st.markdown(
    """
    <style>
    .card {
        border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px; margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        background: white;
    }
    .title {
        font-weight: 700; font-size: 1.05rem;
    }
    .subtle {
        color: #6b7280; font-size: 0.9rem;
    }
    .pill {
        display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 0.85rem;
        background: #f1f5f9; margin-right: 6px;
    }
    .bar-wrap {
        width: 100%; height: 10px; background: #f3f4f6; border-radius: 999px; overflow: hidden;
    }
    .bar {
        height: 100%;
        background: linear-gradient(90deg, #22c55e, #16a34a);
    }
    .label {
        font-size: 0.85rem; color: #374151; font-weight: 600;
    }
    .pct {
        font-weight: 700;
    }
    .muted { color:#6b7280; font-size:0.8rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============== MAIN ===============
if go:
    with st.spinner("📡 Λήψη αγώνων από API..."):
        fixtures = fetch_fixtures(league_id, season, days_ahead)

    if not fixtures:
        st.warning("Δεν βρέθηκαν επερχόμενοι αγώνες για τις επιλεγμένες ρυθμίσεις.")
        st.stop()

    cards_data = []
    # Για σταθερότητα προβολής, seed ανά fixture id όταν έχουμε.
    for m in fixtures:
        home = m["teams"]["home"]["name"]
        away = m["teams"]["away"]["name"]
        date = m["fixture"]["date"]
        short_date = date[:16].replace("T", " ")

        # ======= PREDICT (χρησιμοποιούμε προσωρινά dummy features μέχρι την πλήρη ενσωμάτωση features) =======
        n_res = getattr(model_result, "n_features_in_", 20)
        n_ou = getattr(model_over, "n_features_in_", 20)

        # σταθερό seed ανά matchup για να μην "πηδάει" το UI σε κάθε rerun
        seed_val = abs(hash((home, away))) % (2**32)
        rng_res = np.random.default_rng(seed_val)
        rng_ou = np.random.default_rng(seed_val + 13)

        X_res = rng_res.random((1, n_res))
        X_ou = rng_ou.random((1, n_ou))

        probs_res = model_result.predict_proba(X_res)[0]   # [Home, Draw, Away]
        probs_ou  = model_over.predict_proba(X_ou)[0]      # [Under, Over]

        # ======= Suggestions =======
        pick_res = suggestion_from_result(probs_res)
        pick_ou  = suggestion_from_ou(probs_ou)

        # ======= Confidences =======
        conf_res = softmax_like_topgap(probs_res)           # 0..1
        conf_ou  = ou_confidence(probs_ou)                  # 0..1
        conf_combined = float(np.clip((conf_res + conf_ou) / 2, 0.0, 1.0))

        cards_data.append({
            "date": short_date,
            "home": home,
            "away": away,
            "p_home": round(probs_res[0] * 100, 1),
            "p_draw": round(probs_res[1] * 100, 1),
            "p_away": round(probs_res[2] * 100, 1),
            "p_over": round(probs_ou[1] * 100, 1),
            "p_under": round(probs_ou[0] * 100, 1),
            "pick_res": pick_res,
            "pick_ou": pick_ou,
            "conf_res": round(conf_res * 100, 1),
            "conf_ou": round(conf_ou * 100, 1),
            "conf_combined": round(conf_combined * 100, 1),
        })

    # ======= Overall reliability bar (μέσος όρος) =======
    overall = round(np.mean([c["conf_combined"] for c in cards_data]), 1)
    st.markdown("**Συνολική Φερεγγυότητα (Result + Over/Under)**")
    st.progress(min(1.0, overall/100.0), text=f"{overall}%")

    st.divider()

    # ======= Cards Rendering =======
    # διαιρούμε την οθόνη σε 2 ή 3 columns ανάλογα με πλάτος
    cols = st.columns(2)
    for i, c in enumerate(cards_data):
        with cols[i % 2]:
            color_home = color_for_prob(c["p_home"])
            color_draw = color_for_prob(c["p_draw"])
            color_away = color_for_prob(c["p_away"])
            color_ov   = color_for_prob(c["p_over"])
            color_un   = color_for_prob(c["p_under"])

            st.markdown(f"""
            <div class="card">
              <div class="subtle">{c["date"]}</div>
              <div class="title">{c["home"]} <span class="subtle">vs</span> {c["away"]}</div>
              <div style="margin-top:6px; margin-bottom:10px;">
                <span class="pill">{c["pick_res"]} • Result</span>
                <span class="pill">{c["pick_ou"]} • O/U</span>
              </div>

              <div style="display:flex; gap:14px; flex-wrap:wrap;">
                <div><span class="label">Home:</span> <span class="pct" style="color:{color_home}">{c["p_home"]}%</span></div>
                <div><span class="label">Draw:</span> <span class="pct" style="color:{color_draw}">{c["p_draw"]}%</span></div>
                <div><span class="label">Away:</span> <span class="pct" style="color:{color_away}">{c["p_away"]}%</span></div>
              </div>

              <div style="display:flex; gap:14px; flex-wrap:wrap; margin-top:6px;">
                <div><span class="label">Over 2.5:</span> <span class="pct" style="color:{color_ov}">{c["p_over"]}%</span></div>
                <div><span class="label">Under 2.5:</span> <span class="pct" style="color:{color_un}">{c["p_under"]}%</span></div>
              </div>

              <div style="margin-top:10px;" class="muted">Φερεγγυότητα Αποτελέσματος: {c["conf_res"]}% • Φερεγγυότητα O/U: {c["conf_ou"]}%</div>
              <div style="margin-top:6px;">Συνδυαστική Φερεγγυότητα</div>
              <div class="bar-wrap" title="{c["conf_combined"]}%">
                <div class="bar" style="width:{c["conf_combined"]}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ======= CSV Download =======
    out_df = pd.DataFrame(cards_data)
    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Κατέβασε CSV", csv, "predictions_cards.csv", "text/csv", use_container_width=True)
else:
    st.info("Επίλεξε λίγκα, σεζόν και μέρες, μετά πάτα **🔮 Δημιούργησε προβλέψεις** για να δεις τις κάρτες.")
